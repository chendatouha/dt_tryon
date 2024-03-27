from my_datasets.base_dataset import BaseDataset
import os
import numpy as np
from PIL import Image

class VITONDataset(BaseDataset):
    def __init__(self, dataset_dir, sample_file, mode, size=(384, 512), pad_ratio=(3, 4), preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=220, ref_size=(384, 512), pair=False):
        super(VITONDataset, self).__init__(dataset_dir=dataset_dir, sample_file=sample_file,pad_ratio=pad_ratio,
                                              size=size, preview_dir=preview_dir, repeat_len=repeat_len, keep_num=keep_num,
                                              background_color=background_color)
        self.img_dir = 'image'
        self.parse_dir = 'image-parse-v3'
        self.pose_dir = 'openpose_img'
        self.cloth_dir = 'cloth'
        self.ref_size = ref_size
        self.mode = mode
        self.pair = pair


    def load_sample_file(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf8') as fp:
            for line in fp.readlines():
                samples.append(line.strip())
        print('load {} samples'.format(len(samples)))
        return samples

    def __getitem__(self, idx):
        if self.repeat_len > 0:
            idx = idx % len(self.samples)
        sample = self.samples[idx]
        if not self.pair:
            img, pose, parse, img_ref, cloth = self._load_files(sample)
        else:
            sample1, sample2 = sample.split(' ')
            img, pose, parse, img_ref, _ = self._load_files(sample1)
            _, _, _, _, cloth = self._load_files(sample2)
        parse = np.asarray(parse)
        img_ref = self._remove_background(parse, np.asarray(img_ref))


        if self.preview_dir:
            Image.fromarray(img_ref).save(os.path.join(self.preview_dir, 'ref_{}'.format(sample)))
            pose.save(os.path.join(self.preview_dir, 'pose_{}'.format(sample)))
            cloth.save(os.path.join(self.preview_dir, 'cloth_{}'.format(sample)))
            img.save(os.path.join(self.preview_dir, 'img_{}'.format(sample)))

        img = self.normalize(img)
        pose = self.normalize(pose)
        img_ref = self.normalize(img_ref)
        cloth = self.normalize(cloth)


        example = {}
        example['img'] = img
        example['pose'] = pose
        example['img_ref'] = img_ref
        example['cloth'] = cloth
        return example




    def _load_files(self, file_name):
        file_name, _ = os.path.splitext(file_name)
        person_path = os.path.join(self.dataset_dir, self.mode, self.img_dir, file_name+'.jpg')
        pose_path = os.path.join(self.dataset_dir, self.mode, self.pose_dir, file_name+'_rendered.png')
        parse_path = os.path.join(self.dataset_dir, self.mode, self.parse_dir, file_name+'.png')
        cloth_path = os.path.join(self.dataset_dir, self.mode, self.cloth_dir, file_name+'.jpg')
        person_img = self.load_img(person_path, self.pad_ratio, self.background_color, resize=self.size, resample='bilinear')
        person_img_ref = person_img.resize(self.ref_size, resample=Image.BILINEAR)
        pose_img = self.load_img(pose_path, self.pad_ratio, 0, resize=self.size, resample='bilinear', to_rgb=True)
        parse_img = self.load_img(parse_path, self.pad_ratio, 0, resize=self.ref_size, resample='nearest', to_rgb=False)
        cloth_img = self.load_img(cloth_path, self.pad_ratio, self.background_color, resize=self.ref_size, resample='bilinear', to_rgb=True)

        return person_img, pose_img, parse_img, person_img_ref, cloth_img

    def _remove_background(self, seg_np, img_np):
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
    dataset_dir = r'D:\workspace\data4ALL\zalando-hd-resized'
    dataset = VITONDataset(dataset_dir=dataset_dir, sample_file='test_pairs.txt', mode='test', pair=True)
    for d in dataset:
        print(d)