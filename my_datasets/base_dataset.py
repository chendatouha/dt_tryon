from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np
from utils.img_related import pad_to_ratio


class BaseDataset(Dataset):
    def __init__(self, dataset_dir=None, sample_file=None, size=(384, 512), pad_ratio=(3,4), mean=0.5, std=0.5, preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=220):
        self.dataset_dir = dataset_dir
        self.sample_file = sample_file
        self.samples = self.load_sample_file(os.path.join(dataset_dir, sample_file))
        if keep_num > 0:
            self.samples = self.samples[:keep_num]
        self.pad_ratio = pad_ratio
        self.background_color = background_color
        self.size = size
        self.mean = mean
        self.std = std
        self.preview_dir = preview_dir
        self.repeat_len = repeat_len

    def __len__(self):
        if self.repeat_len > 0:
            return self.repeat_len
        return len(self.samples)


    def __getitem__(self, idx):
        raise NotImplemented

    def load_sample_file(self, file_path):
        raise NotImplemented

    def load_img(self, img_path, pad_ratio, padvalue, resize=None, resample='bicubic', to_rgb=True):
        assert resample in ['bicubic', 'nearest', 'bilinear']
        if resample == 'bicubic':
            resample = Image.BICUBIC
        elif resample == 'bilinear':
            resample = Image.BILINEAR
        elif resample == 'nearest':
            resample = Image.NEAREST
        else:
            assert False, "resample must in ['bicubic', 'nearest', 'bilinear']"
        if to_rgb:
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path)
        img_np = np.asarray(img)
        if pad_ratio is not None:
            img = Image.fromarray(pad_to_ratio(img_np, pad_ratio, padvalue))
        else:
            img = Image.fromarray(img_np)
        if resize is not None:
            iw, ih = img.size
            if iw != resize[0] or ih != resize[1]:
                img = img.resize(resize, resample=resample)
        return img

    def normalize(self, img):
        img = TF.to_tensor(img)
        img = (img - self.mean) / self.std
        return img