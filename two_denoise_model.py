import torch
import torch.nn.functional as F
def denoise(noise_scheduler,unet_model,x_t,timesteps,control):
    estimate_noise, classes,control = unet_model(x_t, timesteps , control)
    pred_x_0_noisier = [noise_scheduler.step(estimate_noise[i], timesteps[i], x_t[i])
                        for i in range(len(timesteps))]
    prev_sample = list(map(lambda x: x.prev_sample, pred_x_0_noisier))
    pred_original_sample = list(map(lambda x: x.pred_original_sample.clamp(-1,1), pred_x_0_noisier))
    return prev_sample,pred_original_sample,estimate_noise,classes,control
def two_step_inference(noise_scheduler,unet_model, x_0,  anomaly_mask, control, timesteps):
    control["cond"] = True
    noise_steps = (noise_scheduler.config.num_train_timesteps // noise_scheduler.num_inference_steps)

    # first step
    control,inference = one_step(noise_scheduler, unet_model, x_0,None,anomaly_mask, control,  timesteps)
    first_x0 = control["pred_x0"] = inference["pred_x_0"].detach()
    timesteps = timesteps - noise_steps


    # second step
    x_t =   inference["pred_prev_sample"]
    control,inference = one_step(noise_scheduler, unet_model, x_0,x_t,anomaly_mask, control,  timesteps)
    second_x0 = inference["pred_x_0"].detach()

    return first_x0,second_x0

def one_step(noise_scheduler,unet_model, x_0,x_t,  anomaly_mask, control,timesteps):


    noise = torch.randn_like(x_0)
    if x_t is None:
        x_t = noise_scheduler.add_noise(x_0, noise, timesteps)

    pred_prev_sample,pred_x_0,estimate_noise,classes,control = denoise(noise_scheduler,unet_model, x_t, timesteps,control)
    loss = (estimate_noise - noise).square()[(anomaly_mask == 0).repeat(1, x_0.shape[1], 1, 1)].mean()
    class_loss = F.cross_entropy(classes, control["class_idx"], reduction="none")
    pred_x_0_noisier = torch.stack(pred_x_0, 0)
    pred_prev_sample = torch.stack(pred_prev_sample, 0)
    alpha = 1 / (1 + torch.exp(0.02 * timesteps - 10))
    class_loss = (class_loss * alpha).mean()
    return control,{"loss":loss,"x_t":x_t, "pred_x_0":pred_x_0_noisier.detach(),"pred_prev_sample":pred_prev_sample.detach(), "class_loss":class_loss, "timesteps":timesteps}
