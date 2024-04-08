from diffusers.models import AutoencoderKL
from utils.train_utils import instantiate_from_config
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.img_related import save_image_tensor, tensor2img
from models.animate.unet_2d_condition_spatial_attn_v2 import UNet2DConditionModelSpatialAttn
from models.animate.unet_2d_condition_spatial_attn_v2_temporal import UNet2DConditionModelSpatialAttn as Temporal
from PIL import Image

"""
"""


class AnimateTrainer:
    def __init__(
            self,
            vae_cfg,
            ref_net_cfg,
            unet_cfg,
            scheduler_cfg,
            animate_cfg
    ):
        self.ref_net = UNet2DConditionModelSpatialAttn.from_pretrained(
            ref_net_cfg.pretrained_model, subfolder=ref_net_cfg.subfolder, in_channels=ref_net_cfg.in_channels,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        self.unet = Temporal.from_pretrained(
            unet_cfg.pretrained_model, subfolder=unet_cfg.subfolder, in_channels=unet_cfg.in_channels,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        self.scheduler = instantiate_from_config(scheduler_cfg)
        self.vae = AutoencoderKL.from_pretrained(vae_cfg.pretrained_model, subfolder=vae_cfg.subfolder)

        self.weight_dtype = torch.float32
        self.seq_num = animate_cfg.seq_num - 1

    def train(self, train_loader, val_loader, train_cfg, logdir):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        accelerator = Accelerator(mixed_precision=train_cfg.mixed_precision)

        device = accelerator.device
        if train_cfg.get('unet_pretrained_model', None):
            self.unet.load_state_dict(torch.load(train_cfg.unet_pretrained_model, map_location='cpu'), strict=False)
            print('load pretrained from {}'.format(train_cfg.unet_pretrained_model))
        if train_cfg.get('ref_net_pretrained_model', None):
            self.ref_net.load_state_dict(torch.load(train_cfg.ref_net_pretrained_model, map_location='cpu'), strict=False)
            print('load pretrained from {}'.format(train_cfg.ref_net_pretrained_model))

        # self.unet.enable_xformers_memory_efficient_attention()
        self.ref_net.enable_xformers_memory_efficient_attention()
        self.vae.enable_slicing()
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.ref_net.eval()
        self.ref_net.requires_grad_(False)

        if train_cfg.scale_lr:
            train_cfg.learning_rate = (
                    train_cfg.learning_rate * accelerator.num_processes
            )
            print('scale lr to {}'.format(train_cfg.learning_rate))

        params_to_optimize = []
        for n, p in self.unet.named_parameters():
            if 'temporal' in n:
                params_to_optimize.append(p)
            else:
                p.requires_grad_(False)
        print('trainable params in model: {}'.format(len(params_to_optimize)))
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=train_cfg.learning_rate
        )

        self.unet, optimizer, train_loader, val_loader = \
            accelerator.prepare(self.unet, optimizer, train_loader, val_loader)

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        self.weight_dtype = weight_dtype
        self.vae.to(device, dtype=self.weight_dtype)
        self.ref_net.to(device, dtype=self.weight_dtype)


        if accelerator.is_local_main_process:
            print("***** Running training *****")
            print(f"  Num examples = {len(train_loader.dataset)}")
            print(f"  Num batches each epoch = {len(train_loader)}")
            print(f"  Num Epochs = {train_cfg.num_train_epochs}")
        global_step = 0
        for epoch in range(train_cfg.num_train_epochs):
            loss_sum = 0
            pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(pbar):
                self.unet.train()
                with accelerator.accumulate(self.unet):
                    inputs = self.get_input(batch, accelerator, return_rec=False, return_uc=False)
                    ref = inputs['ref']
                    frames = inputs['frames']
                    target_pose = inputs['target_pose']

                    bsz, c, h, w = ref.shape
                    t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                    with torch.no_grad():
                        _, result_refs = self.ref_net(ref, t, encoder_hidden_states=None)

                    t_frames = t.unsqueeze(-1).repeat([1, self.seq_num]).reshape(bsz*self.seq_num,)
                    noise = torch.randn_like(frames)
                    noisy_latents = self.scheduler.add_noise(frames, noise, t_frames)
                    model_pred, _ = self.unet(torch.cat([noisy_latents, target_pose], dim=1),
                                              t_frames,encoder_hidden_states=None, hidden_states_refs=result_refs,
                                              seq_num=self.seq_num)

                    if self.scheduler.config.prediction_type == "epsilon":
                        target = noise
                        raise Exception

                    elif self.scheduler.config.prediction_type == "v_prediction":
                        target = self.scheduler.get_velocity(frames, noise, t)
                    else:
                        raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
                    loss = F.mse_loss(model_pred, target, reduction="mean")
                    loss_sum += loss.detach().item()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.unet.parameters(), train_cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    if global_step % train_cfg.log_batch_frequency == 0:
                        self.unet.eval()
                        sample_result = self.sample_image(batch, train_cfg.log_image_cfg, accelerator)
                        save_image_tensor(sample_result['result'], os.path.join(logdir, 'train', '{}_{}_result.jpg'.format(global_step, local_rank)))
                        save_image_tensor(sample_result['ref_rec'], os.path.join(logdir, 'train', '{}_{}_source.jpg'.format(global_step, local_rank)))
                    if accelerator.is_local_main_process:
                        pbar.set_description("epoch={}, step={}, loss={:.4f}".format(epoch, global_step, loss_sum/(step+1)))
                        if global_step % train_cfg.save_batch_frequency == 0 and global_step != 0:
                            save_path = os.path.join(train_cfg.log_dir, "step_{}_unet.pt".format(global_step))
                            self.save_model(save_path, accelerator.unwrap_model(self.unet), accelerator)
                    global_step += 1
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            save_path = os.path.join(train_cfg.log_dir, "final_unet.pt")
            self.save_model(save_path, accelerator.unwrap_model(self.unet), accelerator)


    def test(self, test_loader, test_cfg, logdir):
        accelerator = Accelerator(mixed_precision=test_cfg.mixed_precision)

        device = accelerator.device
        if test_cfg.get('unet_pretrained_model', None):
            self.unet.load_state_dict(torch.load(test_cfg.unet_pretrained_model, map_location='cpu'), strict=False)
            print('load pretrained from {}'.format(test_cfg.unet_pretrained_model))
        if test_cfg.get('ref_net_pretrained_model', None):
            self.ref_net.load_state_dict(torch.load(test_cfg.ref_net_pretrained_model, map_location='cpu'), strict=False)
            print('load pretrained from {}'.format(test_cfg.ref_net_pretrained_model))
        # self.unet.enable_xformers_memory_efficient_attention()
        self.ref_net.enable_xformers_memory_efficient_attention()
        self.vae.enable_slicing()
        self.vae.eval()
        self.ref_net.eval()
        self.unet.eval()

        test_loader = accelerator.prepare(test_loader)

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        self.weight_dtype = weight_dtype
        self.vae.to(device, dtype=self.weight_dtype)
        self.ref_net.to(device, dtype=self.weight_dtype)
        self.unet.to(device, dtype=self.weight_dtype)
        pbar = tqdm(test_loader, disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(pbar):
            sample_result = self.sample_image(batch, test_cfg.log_image_cfg, accelerator)
            save_image_tensor(sample_result['ref_rec'], os.path.join(logdir, '{}_source.jpg'.format(step)))
            result = sample_result['result']
            pils = []
            for i in range(result.shape[0]):
                pil = Image.fromarray(tensor2img(result[i], min_max=(-1, 1)))
                pils.append(pil)
            pils[0].save(os.path.join(logdir, '{}_result.gif'.format(step)),
                         save_all=True, append_images=pils[1:], duration=100, loop=0)



    def get_input(self, batch, accelerator, return_rec, return_uc):
        img_ref = batch['img_ref']
        target_pose = batch['pose']
        frames = batch['frames']
        bs, seq, c, h, w = frames.shape
        frames = frames.reshape(bs*seq, c, h, w)
        target_pose = target_pose.reshape(bs*seq, c, h, w)
        weight_dtype = self.weight_dtype
        with torch.no_grad():
            vae = accelerator.unwrap_model(self.vae)
            z_ref = vae.encode(img_ref.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_frames = vae.encode(frames.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_pose = vae.encode(target_pose.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            if return_uc:
                z_ref_uc = vae.encode(torch.zeros_like(img_ref).to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                z_pose_uc = vae.encode(torch.zeros_like(target_pose).to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            inputs = {}
            inputs['ref'] = z_ref
            inputs['frames'] = z_frames
            inputs['target_pose'] = z_pose
            if return_uc:
                inputs['ref_uc'] = z_ref_uc
                inputs['target_pose_uc'] = z_pose_uc

            if return_rec:
                ref_rec = vae.decode(z_ref / vae.config.scaling_factor).sample
                inputs['ref_rec'] = ref_rec

        return inputs

    def save_model(self, path, model, accelerator):
        net_dict = accelerator.get_state_dict(model)
        accelerator.save(net_dict,path)



    def sample_image(self, batch, log_cfg, accelerator):
        self.scheduler.set_timesteps(log_cfg.steps)
        timesteps = self.scheduler.timesteps
        scale = log_cfg.classifier_free_scale
        with torch.no_grad():
            inputs = self.get_input(batch, accelerator, return_rec=True, return_uc=True)
            ref = inputs['ref']
            ref_uc = inputs['ref_uc']
            pose = inputs['target_pose']
            pose_uc = inputs['target_pose_uc']
            ref_all = torch.cat([ref, ref_uc], dim=0)
            pose_all = torch.cat([pose, pose_uc], dim=0)
            img = torch.randn_like(inputs['frames'])
            for t in timesteps:
                img_all = torch.cat([img, img], dim=0)
                _, result_refs = self.ref_net(ref_all, t, encoder_hidden_states=None)
                model_pred, _ = self.unet(torch.cat([img_all, pose_all], dim=1),
                                          t,encoder_hidden_states=None, hidden_states_refs=result_refs,seq_num=self.seq_num)
                pred, pred_uc = model_pred.chunk(2)
                noise_pred = pred_uc + scale * (pred - pred_uc)
                img = self.scheduler.step(noise_pred, t, img, eta=log_cfg.eta).prev_sample
            vae = accelerator.unwrap_model(self.vae)
            img = img.to(self.weight_dtype) * 1 / vae.config.scaling_factor
            img = vae.decode(img).sample.to(torch.float32)
            sample_result = {}
            sample_result['result'] = img
            sample_result['ref_rec'] = inputs['ref_rec']
        return sample_result





