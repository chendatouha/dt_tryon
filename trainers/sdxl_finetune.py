from diffusers.models import AutoencoderKL, UNet2DConditionModel
from utils.train_utils import instantiate_from_config
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.img_related import save_image_tensor
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, get_constant_schedule_with_warmup
from utils.general_utls import log_captions
from diffusers.training_utils import compute_snr, EMAModel
"""
SDXL finetune
"""


class SDXLTrainer:
    def __init__(
            self,
            vae_cfg,
            unet_cfg,
            scheduler_cfg,
            tokenizer1_cfg,
            tokenizer2_cfg,
            text_encoder1_cfg,
            text_encoder2_cfg,

    ):
        self.unet = UNet2DConditionModel.from_pretrained(
            unet_cfg.pretrained_model, subfolder=unet_cfg.subfolder,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True, variant='fp16'
        )
        self.scheduler = instantiate_from_config(scheduler_cfg)
        self.vae = AutoencoderKL.from_pretrained(vae_cfg.pretrained_model, subfolder=vae_cfg.subfolder
                                                 ,low_cpu_mem_usage=False, ignore_mismatched_sizes=True,variant='fp16')
        self.ucg_rng_generator = np.random.RandomState()
        self.tokenizer1 = CLIPTokenizer.from_pretrained(
            tokenizer1_cfg.pretrained_model, subfolder=tokenizer1_cfg.subfolder,
        )
        self.tokenizer2 = CLIPTokenizer.from_pretrained(
            tokenizer2_cfg.pretrained_model, subfolder=tokenizer2_cfg.subfolder,
        )
        self.text_encoder1 = CLIPTextModel.from_pretrained(text_encoder1_cfg.pretrained_model,
                                                           subfolder=text_encoder1_cfg.subfolder, variant='fp16')
        self.text_encoder2 = CLIPTextModelWithProjection.from_pretrained(text_encoder2_cfg.pretrained_model,
                                                                         subfolder=text_encoder2_cfg.subfolder)

        self.weight_dtype = torch.float32

    def train(self, train_loader, val_loader, train_cfg, logdir):

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        accelerator = Accelerator(mixed_precision=train_cfg.mixed_precision,
                                  gradient_accumulation_steps=train_cfg.gradient_accumulation_steps)

        if train_cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        device = accelerator.device

        if train_cfg.get('vae_pretrained_model', None):
            self.vae.load_state_dict(torch.load(train_cfg.vae_pretrained_model, map_location='cpu'), strict=True)
            accelerator.print('load pretrained vae from {}'.format(train_cfg.vae_pretrained_model))
        if train_cfg.get('unet_pretrained_model', None):
            self.unet.load_state_dict(torch.load(train_cfg.unet_pretrained_model, map_location='cpu'), strict=True)
            accelerator.print('load pretrained vae from {}'.format(train_cfg.unet_pretrained_model))

        self.unet.enable_xformers_memory_efficient_attention()
        self.vae.enable_slicing()
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.text_encoder1.eval()
        self.text_encoder1.requires_grad_(False)
        self.text_encoder2.eval()
        self.text_encoder2.requires_grad_(False)

        if train_cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if train_cfg.scale_lr:
            train_cfg.learning_rate = (
                    train_cfg.learning_rate * accelerator.num_processes
            )
            accelerator.print('scale lr to {}'.format(train_cfg.learning_rate))

        params_to_optimize = self.unet.parameters()

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=train_cfg.learning_rate
        )

        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps)
        self.unet, optimizer, train_loader, lr_scheduler = \
            accelerator.prepare(self.unet, optimizer, train_loader, lr_scheduler)

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        self.weight_dtype = weight_dtype
        self.text_encoder1.to(device, dtype=self.weight_dtype)
        self.text_encoder2.to(device, dtype=self.weight_dtype)
        self.vae.to(device, dtype=self.weight_dtype)

        if train_cfg.use_ema:
            ema_unet = EMAModel(self.unet.parameters(), decay=train_cfg.ema_decay)
            ema_unet.to(device)


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
                    inputs = self.get_input(batch, accelerator, ucg_prob=train_cfg.ucg_prob, return_rec=False)
                    target = inputs['target']
                    encoder_states = inputs['encoder_states']
                    add_text_embeds = inputs['encoder_states_pooled']
                    add_time_ids = inputs['add_time_ids']
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise = torch.randn_like(target)
                    if train_cfg.noise_offset:
                        noise += train_cfg.noise_offset * torch.randn(
                            (noise.shape[0], noise.shape[1], 1, 1), device=noise.device
                        )

                    bsz = noise.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device)
                    noisy_latents = self.scheduler.add_noise(target, noise, timesteps)
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_states,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    prediction_type = self.scheduler.config.prediction_type
                    assert prediction_type in ['epsilon', 'v_prediction']
                    if prediction_type == "epsilon":
                        target = noise
                    else:
                        target = self.scheduler.get_velocity(target, noise, timesteps)

                    if train_cfg.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        snr = compute_snr(self.scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, train_cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        if prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        else:
                            mse_loss_weights = mse_loss_weights / (snr + 1)
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()
                    loss_sum += loss.detach().item()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.unet.parameters(), train_cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    if train_cfg.use_ema:
                        ema_unet.step(self.unet.parameters())
                    if global_step % train_cfg.log_batch_frequency == 0:
                        self.unet.eval()
                        if train_cfg.use_ema:
                            ema_unet.store(self.unet.parameters())
                            ema_unet.copy_to(self.unet.parameters())
                        sample_result = self.sample_image(batch, train_cfg.log_image_cfg, accelerator)
                        self.save_result(sample_result, global_step, local_rank, logdir, 'train')
                        if train_cfg.use_ema:
                            ema_unet.restore(self.unet.parameters())
                    if accelerator.is_local_main_process:
                        pbar.set_description("epoch={}, step={}, loss={:.4f}".format(epoch, global_step, loss_sum/(step+1)))
                        if global_step % train_cfg.save_batch_frequency == 0 and global_step != 0:
                            save_path = os.path.join(train_cfg.log_dir, "step_{}.pt".format(global_step))
                            self.save_model(save_path, accelerator.unwrap_model(self.unet), accelerator)
                            if train_cfg.use_ema:
                                ema_unet.store(self.unet.parameters())
                                ema_unet.copy_to(self.unet.parameters())
                                save_path = os.path.join(train_cfg.log_dir, 'ema_step_{}.pt'.format(global_step))
                                self.save_model(save_path, accelerator.unwrap_model(self.unet), accelerator)
                                ema_unet.restore(self.unet.parameters())
                    global_step += 1
        if accelerator.is_local_main_process:
            save_path = os.path.join(train_cfg.log_dir, "final.pt")
            self.save_model(save_path, accelerator.unwrap_model(self.unet), accelerator)
            if train_cfg.use_ema:
                ema_unet.store(self.unet.parameters())
                ema_unet.copy_to(self.unet.parameters())
                save_path = os.path.join(train_cfg.log_dir, 'ema_final.pt')
                self.save_model(save_path, accelerator.unwrap_model(self.unet), accelerator)
                ema_unet.restore(self.unet.parameters())

    def test(self, test_loader, test_cfg, logdir):
        raise NotImplemented


    def get_input(self, batch, accelerator, ucg_prob, return_rec):
        img = batch['img']
        caption = batch['caption']
        add_time_ids = batch['add_time_ids']
        weight_dtype = self.weight_dtype

        with torch.no_grad():
            encoder_states, encoder_states_pooled = self.encode_prompt(caption, caption, accelerator.device)
            if ucg_prob > 0:
                for i in range(encoder_states.shape[0]):
                    if self.ucg_rng_generator.choice(2, p=[1 - ucg_prob, ucg_prob]):
                        encoder_states[i] = torch.zeros_like(encoder_states[i])
                        encoder_states_pooled[i] = torch.zeros_like(encoder_states_pooled[i])
            vae = accelerator.unwrap_model(self.vae)
            z_target = vae.encode(img.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

            inputs = {}
            inputs['target'] = z_target
            inputs['encoder_states'] = encoder_states
            inputs['encoder_states_pooled'] = encoder_states_pooled
            inputs['add_time_ids'] = add_time_ids
            inputs['caption'] = caption

            if return_rec:
                target_rec = vae.decode(z_target / vae.config.scaling_factor).sample
                inputs['target_rec'] = target_rec
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
            target = inputs['target']
            encoder_states = inputs['encoder_states']
            encoder_states_uc = inputs_uc['encoder_states']
            pooled = inputs['encoder_states_pooled']
            pooled_uc = inputs_uc['encoder_states_pooled']
            add_time_ids = inputs['add_time_ids']
            add_time_ids_uc = inputs['add_time_ids']
            added_cond_kwargs = {"text_embeds": torch.cat([pooled, pooled_uc], dim=0),
                                 "time_ids":torch.cat([add_time_ids, add_time_ids_uc], dim=0)}
            encoder_states_all = torch.cat([encoder_states, encoder_states_uc], dim=0)
            img = torch.randn_like(target)
            img = img * self.scheduler.init_noise_sigma
            for t in timesteps:
                img_all = torch.cat([img, img], dim=0)
                img_all = self.scheduler.scale_model_input(img_all, t)
                model_pred = self.unet(
                    img_all,
                    t,
                    encoder_hidden_states=encoder_states_all,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                pred, pred_uc = model_pred.chunk(2)
                noise_pred = pred_uc + scale * (pred - pred_uc)
                img = self.scheduler.step(noise_pred, t, img).prev_sample
            vae = accelerator.unwrap_model(self.vae)
            img = img.to(self.weight_dtype) * 1 / vae.config.scaling_factor
            img = vae.decode(img).sample.to(torch.float32)
            sample_result = {}
            sample_result['result'] = img
            sample_result['target_rec'] = inputs['target_rec']
            sample_result['caption'] = inputs['caption']
        return sample_result


    def encode_prompt(self,prompt, prompt2, device):
        tokenizers = [self.tokenizer1, self.tokenizer2]
        text_encoders = [self.text_encoder1, self.text_encoder2]
        prompt_embeds_list = []
        prompts = [prompt, prompt2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def save_result(self, sample_result, step, local_rank, logdir, train_val='train'):
        result_path = os.path.join(logdir, train_val, '{}_{}_result.jpg'.format(step, local_rank))
        save_image_tensor(sample_result['result'], result_path)
        target_path = os.path.join(logdir, 'train', '{}_{}_target.jpg'.format(step, local_rank))
        save_image_tensor(sample_result['target_rec'], target_path)
        log_captions(sample_result['caption'], os.path.join(logdir, 'train', '{}_{}_cap.txt'.format(step, local_rank)))


