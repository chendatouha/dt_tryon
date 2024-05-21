from my_datasets.base_dataset import BaseDataset
import os
import random
from PIL import Image
import json
import torch
from tqdm import tqdm
from torchvision.transforms.v2.functional import resize, center_crop
class SDFineTune(BaseDataset):
    def __init__(self, dataset_dir, sample_file, size=(384, 512), pad_ratio=(3, 4), preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=220):
        super(SDFineTune, self).__init__(dataset_dir=dataset_dir, sample_file=sample_file,pad_ratio=pad_ratio,
                                        size=size, preview_dir=preview_dir, repeat_len=repeat_len, keep_num=keep_num,
                                        background_color=background_color)
        self.img_dir = 'fine_tune'





    def load_sample_file(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf8') as fp:
            for bid, line in enumerate(fp.readlines()):
                sample = json.loads(line)
                samples.append(sample)
        print('load {} samples'.format(len(samples)))
        return samples


    def __getitem__(self, idx):
        if self.repeat_len > 0:
            idx = idx % len(self.samples)
        sample = self.samples[idx]
        file, caption = sample['file'], sample['cap']
        file_path = os.path.join(self.dataset_dir, self.img_dir, file)
        img = self.load_img(file_path, pad_ratio=None, padvalue=None, resize=self.size)
        img = self.normalize(img)
        add_time_id = (self.size + [0,0] + self.size)
        add_time_id = torch.tensor(add_time_id, dtype=torch.float32)
        example = {}
        example['img'] = img
        example['caption'] = caption
        example['add_time_ids'] = add_time_id
        return example


if __name__ == "__main__":
    pass