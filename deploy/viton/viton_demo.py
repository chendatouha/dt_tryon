import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from utils.img_related import pad_to_ratio
import torchvision.transforms.functional as TF
from models.animate.unet_2d_condition_spatial_attn_v2 import UNet2DConditionModelSpatialAttn
from utils.train_utils import instantiate_from_config
from diffusers.models import AutoencoderKL
from utils.img_related import tensor2img
import os


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


class WebDemo:
    def __init__(self):
        self.config = OmegaConf.load('configs/demo/viton.yaml')
        self.weight_dtype = torch.bfloat16
        self.model, self.scheduler, self.vae = self.initialize_model()
        self.samples = 1
        self.scale = 2
        self.size = (768, 1024)
        self.pad_ratio = (3, 4)
        self.background_value = 246
        self.batch_size = 1
        self.steps = 50
        self.eta = 1
        self.model_pics_path = 'deploy/viton/pics'
        self.cloth_path = 'deploy/viton/cloth'
        self.model_files, self.clothes = self.load_models_pics()
        self.ucg_rng_generator = np.random.RandomState()
    def load_models_pics(self):
        files = {}
        for file in os.listdir(self.model_pics_path):
            file_name, ext = os.path.splitext(file)
            if ext != '.jpg':
                continue
            files[file_name] = [Image.open(os.path.join(self.model_pics_path, file)),
                                Image.open(os.path.join(self.model_pics_path, '{}.png'.format(file_name))),
                                Image.open(os.path.join(self.model_pics_path, '{}_rendered.png'.format(file_name)))]
        cloth = {}
        for file in os.listdir(self.cloth_path):
            file_name, ext = os.path.splitext(file)
            cloth[file_name] = Image.open(os.path.join(self.cloth_path, file))

        return files, cloth



    def normalize(self, img):
        img = TF.to_tensor(img)
        img = (img - 0.5) / 0.5
        return img

    def preprocess(self, img_pil, background_color, resample, to_rgb=False):
        if to_rgb:
            img_pil = img_pil.convert('RGB')
        img_np = np.asarray(img_pil)
        img = Image.fromarray(pad_to_ratio(img_np, self.pad_ratio, background_color))
        img = img.resize(self.size, resample=resample)
        return img

    def initialize_model(self):
        config = self.config
        ref_net_cfg = config.trainer.params.ref_net_cfg
        unet_cfg = config.trainer.params.unet_cfg
        scheduler_cfg = config.trainer.params.scheduler_cfg
        vae_cfg = config.trainer.params.vae_cfg

        ref_net = UNet2DConditionModelSpatialAttn.from_pretrained(
            ref_net_cfg.pretrained_model, subfolder=ref_net_cfg.subfolder, in_channels=ref_net_cfg.in_channels,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        unet = UNet2DConditionModelSpatialAttn.from_pretrained(
            unet_cfg.pretrained_model, subfolder=unet_cfg.subfolder, in_channels=unet_cfg.in_channels,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        scheduler = instantiate_from_config(scheduler_cfg)
        vae = AutoencoderKL.from_pretrained(vae_cfg.pretrained_model, subfolder=vae_cfg.subfolder)
        model = UnitedModel(unet, ref_net)
        model.unet.load_state_dict(torch.load(config.train_cfg.unet_pretrained_model, map_location='cpu'), strict=True)
        print('load pretrained from {}'.format(config.train_cfg.unet_pretrained_model))
        model.ref_net.load_state_dict(torch.load(config.train_cfg.ref_net_pretrained_model, map_location='cpu'), strict=True)
        print('load pretrained from {}'.format(config.train_cfg.ref_net_pretrained_model))
        model.enable_xformer()

        model.to('cuda', dtype=self.weight_dtype)
        vae.to('cuda', dtype=self.weight_dtype)

        return model, scheduler, vae


    def predict(self, selected_model, cloth_pil):
        img_ref, parse, pose = self.model_files[selected_model]
        img_ref = self.preprocess(img_ref, self.background_value, Image.BILINEAR, to_rgb=True)
        parse = self.preprocess(parse, 0, Image.NEAREST)
        pose = self.preprocess(pose, 0, Image.BILINEAR, to_rgb=True)
        cloth = self.preprocess(cloth_pil, self.background_value, Image.BILINEAR, to_rgb=True)
        parse = np.asarray(parse)
        img_ref = remove_background(parse, np.asarray(img_ref))

        pose = self.normalize(pose).to('cuda')
        img_ref = self.normalize(img_ref).to('cuda')
        cloth = self.normalize(cloth).to('cuda')
        batch = {'pose': pose[None, ...], 'img_ref': img_ref[None, ...], 'cloth': cloth[None, ...]}
        log_cfg = self.config.train_cfg.log_image_cfg
        img = self.sample_image(batch, log_cfg)['result']
        img = tensor2img(img[0])
        return img

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown('虚拟试衣测试')
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(choices=list(self.model_files.keys()), label='选择模特(目前不支持自定义模特)',
                                                 multiselect=False)
                    model_img = gr.Image(label='模特图', interactive=True, type='pil')
                    model_dropdown.change(lambda x: self.model_files[x][0], inputs=[model_dropdown], outputs=[model_img])
                with gr.Column():
                    cloth_dropdown = gr.Dropdown(choices=list(self.clothes.keys()), label='内置衣服(支持自定义衣服)',  multiselect=False)
                    cloth_img = gr.Image(label='衣服平面图', interactive=True, type='pil')
                    cloth_dropdown.change(lambda x: self.clothes[x], inputs=[cloth_dropdown], outputs=[cloth_img])
            with gr.Row():
                result_img = gr.Image(label='试衣结果', interactive=False)
                btn_run = gr.Button("试一下")
            btn_run.click(fn=self.predict, inputs=[model_dropdown, cloth_img], outputs=[result_img])

        demo.launch(server_name="0.0.0.0", server_port=5664)

    def get_input(self, batch, ucg_prob):
        cloth = batch['cloth']
        pose = batch['pose']
        img_face = batch['img_ref']

        weight_dtype = self.weight_dtype
        if ucg_prob > 0:
            for i in range(cloth.shape[0]):
                if self.ucg_rng_generator.choice(2, p=[1 - ucg_prob, ucg_prob]):
                    cloth[i] = torch.zeros_like(cloth[i])
                    pose[i] = torch.zeros_like(pose[i])
        with torch.no_grad():
            vae = self.vae
            z_cloth = vae.encode(cloth.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_pose = vae.encode(pose.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            z_face = vae.encode(img_face.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

            inputs = {}
            inputs['cloth'] = z_cloth
            inputs['pose'] = z_pose
            inputs['face'] = z_face

        return inputs

    def sample_image(self, batch, log_cfg):
        self.scheduler.set_timesteps(log_cfg.steps)
        timesteps = self.scheduler.timesteps
        scale = log_cfg.classifier_free_scale
        with torch.no_grad():
            inputs = self.get_input(batch, ucg_prob=0)
            inputs_uc = self.get_input(batch, ucg_prob=1)
            cloth = inputs['cloth']
            cloth_uc = inputs_uc['cloth']
            pose = inputs['pose']
            pose_uc = inputs_uc['pose']
            face = inputs['face']
            face_uc = inputs_uc['face']
            cloth_all = torch.cat([cloth, cloth_uc], dim=0)
            pose_all = torch.cat([pose, pose_uc], dim=0)
            face_all = torch.cat([face, face_uc], dim=0)
            img = torch.randn_like(pose)
            for t in timesteps:
                img_all = torch.cat([img, img], dim=0)
                model_output = self.model(cloth_all, t, img_all, pose_all, face_all)
                pred, pred_uc = model_output.chunk(2)
                noise_pred = pred_uc + scale * (pred - pred_uc)
                img = self.scheduler.step(noise_pred, t, img, eta=log_cfg.eta).prev_sample
            vae = self.vae
            img = img.to(self.weight_dtype) * 1 / vae.config.scaling_factor
            img = vae.decode(img).sample.to(torch.float32)
            sample_result = {}
            sample_result['result'] = img
        return sample_result

def remove_background(seg_np, img_np):
    mask_face = (seg_np == 13).astype(np.uint8)
    mask_hair = (seg_np == 2).astype(np.uint8)
    mask_hair_cutoff = cutoff_hair(mask_face, mask_hair)
    mask_keep = mask_face + mask_hair - mask_hair_cutoff
    img = img_np * mask_keep[..., None] + (1 - mask_keep[..., None]) * 128
    return img
def cutoff_hair(mask_face, mask_hair):
    h, w = mask_face.shape
    idx = np.argwhere(mask_face)
    if idx.shape[0] == 0:
        return np.zeros_like(mask_hair)
    idx_y = idx[..., 0]
    lowest_y = max(idx_y)
    mask = np.zeros((lowest_y,w), dtype=np.uint8)
    mask = np.concatenate([mask, np.ones((h - lowest_y, w), dtype=np.uint8)], axis=0)
    mask = mask_hair * mask
    return mask



if __name__ == "__main__":
    demo = WebDemo()
    demo.launch()
