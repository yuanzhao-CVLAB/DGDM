import json
import os
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
from accelerate import Accelerator
from diffusers.schedulers import DDIMScheduler

# Dataset and model imports
from data.mydataset import MVTecDataset, VisaDataset
from data.accelerate_metric import cal_metric
from models.unet_model import UNetModel, get_unet
from models.changedetection_model import SegmentationSubNetwork
from two_denoise_model import two_step_inference
from utils.constants import datasets_classes


os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

# Set random seed
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(1024)

def defaultdict_from_json(jsonDict):
    """Convert dictionary to defaultdict."""
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def print_param_sum(parameters, model_name):
    """Print total number of parameters in the model."""
    total_params = sum(p.numel() for p in parameters)
    accelerator.print(f"Total parameters in {model_name}: {total_params/1e6}M")

def evalute_single(accelerator, testing_dataset_loader, args, unet_model, seg_model, sub_class):
    """Evaluate model on a single class."""
    unet_model.eval()
    seg_model.eval()

    total_image_pred, total_image_gt = np.array([]), np.array([])
    total_pixel_gt, total_pixel_pred = np.array([]), np.array([])

    for sample in testing_dataset_loader:
        image = sample["augmented_image"]
        target = sample['has_anomaly']
        gt_mask = sample["anomaly_mask"]
        control = sample
        path = sample["path"]
        class_name = sample["class_name"][0]

        # Inference with two-step denoising
        normal_t_tensor = torch.tensor(args["eval_normal_t"], device=image.device).repeat(image.shape[0])
        first_x0, second_x0 = two_step_inference(
            noise_scheduler, unet_model, image, gt_mask, control,             timesteps=normal_t_tensor)

        pred_mask = seg_model(image, second_x0)

        # Compute image-level anomaly score
        topk_out_mask = torch.topk(torch.flatten(pred_mask[0], start_dim=1), (256**2)//400, dim=1)[0]#(256**2)//1000
        image_score = torch.mean(topk_out_mask)

        # Apply Gaussian smoothing
        out_mask = gaussian_filter(pred_mask.detach().cpu().numpy(), sigma=4)
        out_mask = torch.tensor(out_mask)


        # Accumulate image and pixel-level metrics
        total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
        total_image_gt = np.append(total_image_gt, target[0].detach().cpu().numpy())
        total_pixel_gt = np.append(total_pixel_gt, gt_mask[0].flatten().detach().cpu().numpy().astype(int))
        total_pixel_pred = np.append(total_pixel_pred, out_mask[0].flatten().detach().cpu().numpy())

    # Gather metrics across devices
    total_image_gt = accelerator.gather_for_metrics(torch.tensor(total_image_gt).cuda())
    total_pixel_gt = accelerator.gather_for_metrics(torch.tensor(total_pixel_gt).cuda())
    total_image_pred = accelerator.gather_for_metrics(torch.tensor(total_image_pred).cuda())
    total_pixel_pred = accelerator.gather_for_metrics(torch.tensor(total_pixel_pred).cuda())

    return cal_metric(total_image_gt.cuda(), total_pixel_gt.cuda(), total_image_pred, total_pixel_pred, (256, 256))

def evalute(args, unet_model, seg_model, datasets_type, accelerator):
    """Evaluate model over all classes in the dataset."""
    results = []

    for sub_class in tqdm(class_list, desc="Evaluating..."):
        testing_dataset = test_datasets_func[datasets_type](
            args[f"{datasets_type}_root_path"], [sub_class], img_size=args["img_size"], is_train=False)

        test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=args["num_workers"])
        test_loader = accelerator.prepare(test_loader)

        with torch.no_grad():
            results.append(evalute_single(accelerator, test_loader, args, unet_model, seg_model, sub_class))

    # Aggregate and display results
    col_names = ["objects", "Pixel Auroc", "Sample Auroc", "pixel ap", "Sample ap", "pixel aupr", "Sample aupr",
                 "F1max_px", "F1max_sp", "Pixel Aupro"]

    logs = [[cls] + list(res) for cls, res in zip(class_list, results)]
    logs.append(["all"] + np.mean(results, axis=0).tolist())

    pd_data = pd.DataFrame(logs, columns=col_names)
    new_col_order = ["objects", "Sample Auroc", "Sample aupr", "F1max_sp", "Pixel Auroc",
                     "Pixel Aupro", "F1max_px", "pixel ap", "Sample ap", "pixel aupr"]
    pd_data = pd_data.reindex(columns=new_col_order)

    accelerator.print("\n" + tabulate(pd_data.values, headers=new_col_order, tablefmt="pipe"))

def main(datasets_type):
    """Main evaluation pipeline."""

    unet_model = get_unet(args, classes=len(class_list))
    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1, base_channels=64)

    print_param_sum(seg_model.parameters(), "seg_model")
    print_param_sum(unet_model.parameters(), "unet_model")

    unet_model, seg_model = accelerator.prepare(unet_model, seg_model)

    if args["checkpoints"]:
        print("Loading checkpoints:", args["checkpoints"])
        accelerator.load_state(args["checkpoints"], strict=True)
    evalute(args, unet_model, seg_model, datasets_type, accelerator)

if __name__ == '__main__':
    with open('config/config.json', 'r') as f:
        args = json.load(f)

    args = defaultdict_from_json(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16"
    )
    accelerator.print("Running with args:", args)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args["T"],
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args["beta_schedule"]
    )
    noise_scheduler.set_timesteps(num_inference_steps=10)

    # Dataset setup
    datasets_type = "mvtec"
    train_datasets_func = {"visa": VisaDataset, "mvtec": MVTecDataset, "mvtec3d": MVTecDataset}
    test_datasets_func = {"visa": VisaDataset, "mvtec": MVTecDataset, "mvtec3d": MVTecDataset}
    class_list = datasets_classes[datasets_type]

    main(datasets_type)
