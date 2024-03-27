from diffusers.models import AutoencoderKL
from utils.train_utils import instantiate_from_config
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.img_related import save_image_tensor
from models.animate.unet_2d_condition_spatial_attn_v2 import UNet2DConditionModelSpatialAttn
from utils.train_utils import log_GPU_memory_usage
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

"""
viton use hrviton
"""


class VitonTrainer:
    def __init__(
            self,
            vae_cfg,
            ref_net_cfg,
            unet_cfg,
            scheduler_cfg,
    ):
        ref_net = UNet2DConditionModelSpatialAttn.from_pretrained(
            ref_net_cfg.pretrained_model, subfolder=ref_net_cfg.subfolder, in_channels=ref_net_cfg.in_channels,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        unet = UNet2DConditionModelSpatialAttn.from_pretrained(
            unet_cfg.pretrained_model, subfolder=unet_cfg.subfolder, in_channels=unet_cfg.in_channels,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        self.scheduler = instantiate_from_config(scheduler_cfg)
        self.vae = AutoencoderKL.from_pretrained(vae_cfg.pretrained_model, subfolder=vae_cfg.subfolder)
        self.ucg_rng_generator = np.random.RandomState()
        self.weight_dtype = torch.float32
        self.model = UnitedModel(unet, ref_net)

    def train(self, train_loader, val_loader, train_cfg, logdir):
        print('train with ucg_prob={}'.format(train_cfg.ucg_prob))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        accelerator = Accelerator(mixed_precision=train_cfg.mixed_precision,
                                  gradient_accumulation_steps=train_cfg.gradient_accumulation_steps)
        # for ampere GPU
        torch.backends.cuda.matmul.allow_tf32 = True

        device = accelerator.device
        if train_cfg.get('unet_pretrained_model', None):
            self.model.unet.load_state_dict(torch.load(train_cfg.unet_pretrained_model, map_location='cpu'), strict=False)
            print('load pretrained from {}'.format(train_cfg.unet_pretrained_model))
        if train_cfg.get('ref_net_pretrained_model', None):
            self.model.ref_net.load_state_dict(torch.load(train_cfg.ref_net_pretrained_model, map_location='cpu'), strict=False)
            print('load pretrained from {}'.format(train_cfg.ref_net_pretrained_model))

        self.model.enable_xformer()
        self.vae.enable_slicing()
        self.vae.eval()
        self.vae.requires_grad_(False)


        if train_cfg.gradient_checkpointing:
            self.model.enable_gradient_checkpoint()

        if train_cfg.scale_lr:
            train_cfg.learning_rate = (
                    train_cfg.learning_rate * accelerator.num_processes
            )
            print('scale lr to {}'.format(train_cfg.learning_rate))

        params_to_optimize = self.model.parameters()

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=train_cfg.learning_rate
        )

        # lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader))


        self.model, optimizer, train_loader, val_loader = \
            accelerator.prepare(self.model, optimizer, train_loader, val_loader)

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        self.weight_dtype = weight_dtype
        self.vae.to(device, dtype=self.weight_dtype)


        if train_cfg.gradient_checkpointing:
            accelerator.unwrap_model(self.model).enable_gradient_checkpoint()


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
                accelerator.unwrap_model(self.model).train()
                with accelerator.accumulate(self.model):
                    inputs = self.get_input(batch, accelerator, ucg_prob=train_cfg.ucg_prob, return_rec=False)
                    cloth = inputs['cloth']
                    target = inputs['target']
                    target_pose = inputs['pose']
                    face = inputs['face']

                    noise = torch.randn_like(target)
                    bsz = noise.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device)
                    timesteps = timesteps.long()
                    noisy_latents = self.scheduler.add_noise(target, noise, timesteps)
                    model_pred = self.model(cloth, timesteps, noisy_latents, target_pose, face)


                    if self.scheduler.config.prediction_type == "epsilon":
                        target = noise
                        raise Exception

                    elif self.scheduler.config.prediction_type == "v_prediction":
                        target = self.scheduler.get_velocity(target, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
                    loss = F.mse_loss(model_pred, target, reduction="mean")
                    loss_sum += loss.detach().item()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    # log_GPU_memory_usage(local_rank)
                    if global_step % train_cfg.log_batch_frequency == 0:
                        self.model.eval()
                        sample_result = self.sample_image(batch, train_cfg.log_image_cfg, accelerator)
                        save_image_tensor(sample_result['result'], os.path.join(logdir, 'train', '{}_{}_result.jpg'.format(global_step, local_rank)))
                        save_image_tensor(sample_result['cloth_rec'], os.path.join(logdir, 'train', '{}_{}_cloth.jpg'.format(global_step, local_rank)))
                        save_image_tensor(sample_result['target_rec'], os.path.join(logdir, 'train', '{}_{}_target.jpg'.format(global_step, local_rank)))
                        save_image_tensor(sample_result['face'], os.path.join(logdir, 'train', '{}_{}_face.jpg'.format(global_step, local_rank)))
                    if accelerator.is_local_main_process:
                        pbar.set_description("epoch={}, step={}, loss={:.4f}".format(epoch, global_step, loss_sum/(step+1)))
                        if global_step % train_cfg.save_batch_frequency == 0 and global_step != 0:
                            save_path = os.path.join(train_cfg.log_dir, "step_{}_ref.pt".format(global_step))
                            self.save_model(save_path, accelerator.unwrap_model(self.model).ref_net, accelerator)
                            save_path = os.path.join(train_cfg.log_dir, "step_{}_unet.pt".format(global_step))
                            self.save_model(save_path, accelerator.unwrap_model(self.model).unet, accelerator)
                    global_step += 1

            # if accelerator.is_local_main_process:
            if epoch % train_cfg.log_epoch_frequency == 0:
                pbar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
                self.model.eval()
                for step, batch in enumerate(pbar):
                    sample_result = self.sample_image(batch, train_cfg.log_image_cfg, accelerator)
                    save_image_tensor(sample_result['result'], os.path.join(logdir, 'val', '{}_{}_{}_result.jpg'.format(global_step, step, local_rank)))
                    save_image_tensor(sample_result['cloth_rec'], os.path.join(logdir, 'val', '{}_{}_{}_cloth.jpg'.format(global_step, step, local_rank)))
                    save_image_tensor(sample_result['target_rec'], os.path.join(logdir, 'val', '{}_{}_{}_target.jpg'.format(global_step, step, local_rank)))

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            save_path = os.path.join(train_cfg.log_dir, "final_ref.pt")
            self.save_model(save_path, accelerator.unwrap_model(self.model).ref_net, accelerator)
            save_path = os.path.join(train_cfg.log_dir, "final_unet.pt")
            self.save_model(save_path, accelerator.unwrap_model(self.model).unet, accelerator)


    def test(self, test_loader, test_cfg, logdir):
        accelerator = Accelerator(mixed_precision=test_cfg.mixed_precision)
        device = accelerator.device
        self.model, test_loader = accelerator.prepare(self.model, test_loader)
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
        self.model.unet.load_state_dict(torch.load(test_cfg.unet_pretrained_model, map_location='cpu'), strict=True)
        print('load pretrained from {}'.format(test_cfg.unet_pretrained_model))
        self.model.ref_net.load_state_dict(torch.load(test_cfg.ref_net_pretrained_model, map_location='cpu'), strict=True)
        print('load pretrained from {}'.format(test_cfg.ref_net_pretrained_model))
        self.model.enable_xformer()
        self.model.eval()
        self.vae.enable_slicing()
        self.vae.eval()
        self.vae.to(device, dtype=self.weight_dtype)
        pbar = tqdm(test_loader)
        for step, batch in enumerate(pbar):
            with torch.autocast(device_type='cuda', dtype=weight_dtype):
                sample_result = self.sample_image(batch, test_cfg.log_image_cfg, accelerator)
            combined = torch.cat([sample_result['result'], sample_result['target_rec'], sample_result['cloth_rec']], dim=-1)
            save_image_tensor(combined, os.path.join(logdir, '{}.jpg'.format(step)))



    def get_input(self, batch, accelerator, ucg_prob, return_rec):
        cloth = batch['cloth']
        target = batch['img']
        pose = batch['pose']
        img_face = batch['img_ref']

        weight_dtype = self.weight_dtype
        if ucg_prob > 0:
            for i in range(target.shape[0]):
                if self.ucg_rng_generator.choice(2, p=[1 - ucg_prob, ucg_prob]):
                    cloth[i] = torch.zeros_like(cloth[i])
                    pose[i] = torch.zeros_like(pose[i])
                    # img_face[i] = torch.zeros_like(img_face[i])
        with torch.no_grad():
            vae = accelerator.unwrap_model(self.vae)
            z_cloth = vae.encode(cloth.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_target = vae.encode(target.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_pose = vae.encode(pose.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_face = vae.encode(img_face.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

            inputs = {}
            inputs['cloth'] = z_cloth.to(torch.float32)
            inputs['target'] = z_target.to(torch.float32)
            inputs['pose'] = z_pose.to(torch.float32)
            inputs['face'] = z_face.to(torch.float32)

            if return_rec:
                cloth_rec = vae.decode(z_cloth / vae.config.scaling_factor).sample
                target_rec = vae.decode(z_target / vae.config.scaling_factor).sample
                pose_rec = vae.decode(z_pose / vae.config.scaling_factor).sample
                face_rec = vae.decode(z_face / vae.config.scaling_factor).sample
                inputs['cloth_rec'] = cloth_rec
                inputs['target_rec'] = target_rec
                inputs['pose_rec'] = pose_rec
                inputs['face_rec'] = face_rec

        return inputs

    def save_model(self, path, model, accelerator):
        net_dict = accelerator.get_state_dict(model)
        accelerator.save(net_dict,path)



    def sample_image(self, batch, log_cfg, accelerator):
        self.scheduler.set_timesteps(log_cfg.steps)
        timesteps = self.scheduler.timesteps
        scale = log_cfg.classifier_free_scale
        with torch.no_grad():
            inputs = self.get_input(batch, accelerator, ucg_prob=0, return_rec=True)
            inputs_uc = self.get_input(batch, accelerator, ucg_prob=1, return_rec=False)
            cloth = inputs['cloth']
            cloth_uc = inputs_uc['cloth']
            target = inputs['target']
            pose = inputs['pose']
            pose_uc = inputs_uc['pose']
            face = inputs['face']
            face_uc = inputs_uc['face']
            cloth_all = torch.cat([cloth, cloth_uc], dim=0)
            pose_all = torch.cat([pose, pose_uc], dim=0)
            face_all = torch.cat([face, face_uc], dim=0)
            img = torch.randn_like(target)
            for t in timesteps:
                img_all = torch.cat([img, img], dim=0)
                model_output = self.model(cloth_all, t, img_all, pose_all, face_all)
                pred, pred_uc = model_output.chunk(2)
                noise_pred = pred_uc + scale * (pred - pred_uc)
                img = self.scheduler.step(noise_pred, t, img, eta=log_cfg.eta).prev_sample
            vae = accelerator.unwrap_model(self.vae)
            img = img.to(self.weight_dtype) * 1 / vae.config.scaling_factor
            img = vae.decode(img).sample.to(torch.float32)
            sample_result = {}
            sample_result['result'] = img
            sample_result['cloth_rec'] = inputs['cloth_rec']
            sample_result['target_rec'] = inputs['target_rec']
            sample_result['pose'] = inputs['pose_rec']
            sample_result['face'] = inputs['face_rec']
        return sample_result

    def log_captions(self, captions, file_name):
        with open(file_name, 'w', encoding='utf8') as fp:
            for idx, c in enumerate(captions):
                fp.write('{}: {}\n\n'.format(idx, c))



class UnitedModel(torch.nn.Module):
    def __init__(self, unet, ref_net):
        super().__init__()
        self.unet = unet
        self.ref_net = ref_net

    def forward(self, cloth, timesteps, noisy_latents, target_pose, face):
        _, result_refs = self.ref_net(cloth, timesteps, encoder_hidden_states=None)
        model_pred, _ = self.unet(torch.cat([noisy_latents, target_pose, face], dim=1),
                                  timesteps, encoder_hidden_states=None, hidden_states_refs=result_refs)
        return model_pred

    def enable_xformer(self):
        self.unet.enable_xformers_memory_efficient_attention()
        self.ref_net.enable_xformers_memory_efficient_attention()

    def enable_gradient_checkpoint(self):
        self.unet.enable_gradient_checkpointing()
        self.ref_net.enable_gradient_checkpointing()



