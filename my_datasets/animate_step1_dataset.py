from my_datasets.base_dataset import BaseDataset
import os
import random
import torch
import json
import numpy as np

class AnimateStep1(BaseDataset):
    def __init__(self, dataset_dir, sample_file, size=(384, 512), pad_ratio=(3, 4), preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=220, ref_size=(384, 512)):
        super(AnimateStep1, self).__init__(dataset_dir=dataset_dir, sample_file=sample_file,pad_ratio=pad_ratio,
                                              size=size, preview_dir=preview_dir, repeat_len=repeat_len, keep_num=keep_num,
                                              background_color=background_color)
        self.img_dir = 'image'
        self.pose_dir = 'pose'
        self.ref_size = ref_size


    def load_sample_file(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf8') as fp:
            for line in fp.readlines():
                line = line.strip()
                sample = json.loads(line)
                samples.append(sample)
        print('load {} samples'.format(len(samples)))
        return samples

    def __getitem__(self, idx):
        if self.repeat_len > 0:
            idx = idx % len(self.samples)
        sample = self.samples[idx]
        source, dest = random.sample(sample['files'], 2)
        source_file, source_direct = source.split('@D')
        dest_file, dest_direct = dest.split('@D')
        source_img_pil,_ = self._load_files(source_file, self.ref_size)
        dest_img, dest_pose = self._load_files(dest_file, self.size)

        if self.preview_dir:
            source_img_pil.save(os.path.join(self.preview_dir, 'source_{}'.format(source_file)))
            dest_img.save(os.path.join(self.preview_dir, 'dest_{}'.format(dest_file)))
            dest_pose.save(os.path.join(self.preview_dir, 'pose_{}'.format(dest_file)))

        source_img = self.normalize(source_img_pil)
        dest_img = self.normalize(dest_img)
        pose_img = self.normalize(dest_pose)

        example = {}
        example['source'] = source_img
        example['dest'] = dest_img
        example['dest_pose'] = pose_img

        return example




    def _load_files(self, file_name, size):
        person_path = os.path.join(self.dataset_dir, self.img_dir, file_name)
        pose_path = os.path.join(self.dataset_dir, self.pose_dir, file_name)
        person_img = self.load_img(person_path, self.pad_ratio, self.background_color, resize=size, resample='bilinear')
        pose_img = self.load_img(pose_path, self.pad_ratio, 0, resize=size, resample='bilinear')
        return person_img, pose_img



if __name__ == "__main__":
    from tqdm import tqdm
    data_root = r'path/to/dataset_root'
    sample_file = 'train.txt'
    dataset = AnimateStep1(dataset_dir=data_root, sample_file=sample_file)
    for d in tqdm(dataset):
        print(d)

