from my_datasets.base_dataset import BaseDataset
import os
import numpy as np
from PIL import Image
import torch

class AnimateDataset(BaseDataset):
    def __init__(self, dataset_dir, sample_file, seq_num, size=(384, 512), pad_ratio=(3, 4), preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=220):
        super(AnimateDataset, self).__init__(dataset_dir=dataset_dir, sample_file=sample_file,pad_ratio=pad_ratio,
                                              size=size, preview_dir=preview_dir, repeat_len=repeat_len, keep_num=keep_num,
                                              background_color=background_color)
        self.img_dir = 'frames'
        self.pose_dir = 'pose'
        self.seq_num = seq_num


    def load_sample_file(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf8') as fp:
            for line in fp.readlines():
                samples.append(eval(line.strip()))
        print('load {} samples'.format(len(samples)))
        return samples

    def __getitem__(self, idx):
        if self.repeat_len > 0:
            idx = idx % len(self.samples)
        sample = self.samples[idx]
        assert len(sample) >= self.seq_num
        sample = sample[:self.seq_num]
        poses = []
        imgs = []
        for s in sample:
            img, pose = self._load_files(s)
            img = self.normalize(img)
            pose = self.normalize(pose)
            poses.append(pose[None])
            imgs.append(img[None])
        imgs = torch.cat(imgs, dim=0)
        poses = torch.cat(poses, dim=0)

        example = {}
        example['img_ref'] = imgs[0]
        example['frames'] = imgs[1:]
        example['pose'] = poses[1:]
        return example




    def _load_files(self, file_name):
        person_path = os.path.join(self.dataset_dir, self.img_dir, file_name)
        pose_path = os.path.join(self.dataset_dir, self.pose_dir, file_name)
        person_img = self.load_img(person_path, self.pad_ratio, self.background_color, resize=self.size, resample='bilinear')
        pose_img = self.load_img(pose_path, self.pad_ratio, 0, resize=self.size, resample='bilinear', to_rgb=True)

        return person_img, pose_img


if __name__ == "__main__":
    dataset_dir = r'D:\workspace\data4ALL\animate\UBC fashion'
    dataset = AnimateDataset(dataset_dir=dataset_dir, sample_file='train_list.txt', preview_dir='preview')
    for d in dataset:
        print(d)