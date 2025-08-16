import os
import random
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
from data.perlin import rand_perlin_2d_np
import imgaug.augmenters as iaa
import glob
import math

class MVTecDataset(Dataset):
    def __init__(self, root='/home/data/mvtec', all_class_names=None, is_train=True,
                 img_size=320, shot=-1,syn_anomaly = False):

        self.syn_anomaly = syn_anomaly
        self.root_path = root
        self.is_train = is_train
        self.resize = img_size[0]
        self.cropsize = img_size[0]
        self.mvtec_folder_path = root
        # download dataset if not exist
        # self.download()
        ssl = True
        self.good, self.x, self.y, self.mask = {}, [], [], []
        self.classes = []
        # self.x, self.y, self.mask = self.load_dataset_folder()
        # All_CLASS_NAMES = [class_name]
        if not isinstance(all_class_names, list): all_class_names = [all_class_names]
        self.all_class_names = all_class_names
        class_index = [
            "screw",
            'carpet',
            'grid',
            'leather', 'tile', 'wood',
            'bottle',
            'cable',
            'capsule',
            'hazelnut',
            # 'screw',
            'metal_nut',
            'pill',
            'toothbrush',
            'transistor',
            'zipper'
        ]
        self.class_names = []
        for item, class_name in enumerate(all_class_names):
            self.class_name = class_name
            good1, x1, y1, mask1 = self.load_dataset_folder()
            self.class_names.extend([class_name]*len(x1))
            self.good[class_name] = good1
            self.x.extend(x1)
            self.classes.extend([class_index.index(class_name)] * len(x1))
            # self.classes.extend(len(x1) * [item])
            self.y.extend(y1)
            self.mask.extend(mask1)
            if shot != -1:
                self.good[class_name] = good1[:shot]
                self.x = self.x[:item * shot + 1]
                self.y = self.y[:item * shot + 1]
                self.mask = self.mask[:item * shot + 1]

        self.init_transformer(self.resize, self.cropsize)
        self.class_name_index = -4

    def perlin_synthetic(self, image, path):
        anomaly_source_path = self.anomaly_source_paths[random.randint(0, len(self.anomaly_source_paths) - 1)]
        thresh_path = path.replace('train', 'DISthresh')

        if os.path.exists(thresh_path):
            thresh = cv2.imread(thresh_path, 0)
            thresh = cv2.resize(thresh, dsize=(self.cropsize, self.cropsize))
            thresh = cv2.threshold(thresh, thresh=127, maxval=255, type=cv2.THRESH_BINARY)[1]
            thresh = np.array(thresh).astype(np.float32) / 255.0
        else:
            thresh = np.ones((self.cropsize, self.cropsize))
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                            perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                            perlin_scale, (1,)).numpy()[0])

        has_anomaly = 0
        try_cnt = 0
        while (has_anomaly == 0):
            perlin_noise = rand_perlin_2d_np(
                (self.cropsize, self.cropsize), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            object_perlin = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

            object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)
            object_perlin = object_perlin * thresh[:, :, None]
            if np.sum(object_perlin) != 0:

                has_anomaly = 1
            try_cnt += 1
            # print(try_cnt)

        aug = self.randAugmenter_anomaly()

        anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path), cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.cropsize, self.cropsize))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        img_object_thr = anomaly_img_augmented.astype(
            np.float32) * object_perlin / 255

        beta = torch.rand(1).numpy()[0] * 0.2 + 0.3

        object_perlin = object_perlin

        augmented_image = image * (1 - object_perlin) + beta * img_object_thr * (object_perlin) + (1 - beta) * image * (
            object_perlin)

        augmented_image = augmented_image

        return augmented_image, msk.transpose(2, 0, 1)

    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def init_transformer(self, resize, cropsize):
        self.anomaly_source_paths = sorted(glob.glob(
            "./datasets/dtd-r1.0.1/dtd/images/*/*.jpg"))

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                                   iaa.MultiplyAndAddToBrightness(
                                       mul=(0.8, 1.2), add=(-30, 30)),
                                   iaa.pillike.EnhanceSharpness(),
                                   iaa.AddToHueAndSaturation(
                                       (-50, 50), per_channel=True),
                                   iaa.Solarize(0.5, threshold=(32, 128)),
                                   iaa.Posterize(),
                                   iaa.Invert(),
                                   iaa.pillike.Autocontrast(),
                                   iaa.pillike.Equalize(),
                                   ]

        self.transform_test = T.Compose([
            T.Resize((resize, resize), Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])])
        self.transform_mask = T.Compose([
            T.Resize((self.cropsize, self.cropsize), Image.NEAREST),
            # T.Resize(resize//8),
            # T.CenterCrop((cropsize, cropsize)),
            T.ToTensor()])

    def get_ref_images(self, class_name):
        return random.choice(self.good[class_name])

    def input_transform(self, image):

        image = image.astype(np.float32)
        image = image
        image -= [0.5, 0.5, 0.5]
        image /= [0.5, 0.5, 0.5]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image

    def __getitem__(self, idx):
        path, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
            mask[mask > 0] = 1
        if self.is_train and self.syn_anomaly:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.cropsize, self.cropsize)) / 255

            has_anomaly = 1 if random.random() > 0.3 else 0
            if has_anomaly == 1:
                img, mask = self.perlin_synthetic(img, path)
                mask = torch.tensor(mask)


        else:
            has_anomaly = y
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.cropsize, self.cropsize)) / 255
        img = self.input_transform(img)

        y = self.classes[idx] if self.is_train else y
        return {"augmented_image": img, "class_idx": y, "anomaly_mask": mask, "path": path, "has_anomaly": has_anomaly,
                "class_name": self.class_names[idx]}

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        # phase =  'test'
        phase = 'train' if self.is_train else 'test'
        good, x, y, mask = [], [], [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.jpg')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                good.extend(img_fpath_list)
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                # gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.jpg')
                #                  for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(good), list(x), list(y), list(mask)

    def download(self):
        """Download dataset if not exist"""

        if not os.path.exists(self.mvtec_folder_path):
            tar_file_path = self.mvtec_folder_path + '.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        return
class VisaDataset(MVTecDataset):
    def __init__(self, root='path', all_class_names=None, is_train=True,
                 img_size=320, shot=-1, syn_anomaly=True):

        self.root_path = root
        self.syn_anomaly = syn_anomaly
        csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)
        self.is_train = is_train
        self.resize = img_size[0]
        self.cropsize = img_size[0]
        self.mvtec_folder_path = root
        # download dataset if not exist
        # self.download()
        ssl = True
        self.good, self.x, self.y, self.mask = {}, [], [], []
        self.classes = []
        # self.x, self.y, self.mask = self.load_dataset_folder()
        # All_CLASS_NAMES = [class_name]
        if not isinstance(all_class_names, list): all_class_names = [all_class_names]
        phase= 'train' if is_train else "test"
        cls_data =csv_data[[True if name in all_class_names else False for name in csv_data['object']]]
        data = cls_data[cls_data['split'] == phase]


        self.phase = phase
        self.img_paths = []
        self.gt_paths = []
        self.class_names = []
        for row in data.values:
            self.x.append(os.path.join(root, row[3]))
            self.y.append(int(isinstance(row[-1],str)))
            self.mask.append(os.path.join(root, str(row[4])))
            self.class_names.append(row[0])
        self.all_class_names = all_class_names
        class_index = ['candle', 'capsules', 'cashew',
                'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
                    'pcb3', 'pcb4', 'pipe_fryum']
        self.classes = [class_index.index(class_name) for class_name in self.class_names]
        self.init_transformer(self.resize, self.cropsize)
        self.class_name_index = -4

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

