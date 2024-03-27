import numpy as np
import cv2
import torchvision
import torchvision.transforms.functional as TF
import torch
from PIL import Image, ImageDraw


def l_to_p(img_l, label_colours=None):
    if label_colours is None:
        label_colours = [(0,0,0)
            , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0),
                         (0,85,85), (85,51,0), (52,86,128), (0,128,0)
            , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    label_colours = np.asarray(label_colours)
    img_p = img_l.convert(mode='P')
    img_p.putpalette(label_colours.astype(np.uint8).flatten())
    return img_p


def tensor2img(tensor, min_max=(-1, 1)):
    tensor = tensor.float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.numpy().transpose((1,2,0))
    img_np = (img_np * 255.0).round().astype(np.uint8)
    return img_np


def save_image_tensor(img_tensor, save_path, min_max=(-1, 1)):
    if img_tensor.dim() == 4:
        img_tensor = torchvision.utils.make_grid(img_tensor)
    img_np = tensor2img(img_tensor, min_max=min_max)
    if img_np.shape[-1] == 1:
        img_np = np.squeeze(img_np, -1)
    im = Image.fromarray(img_np)
    im.save(save_path)


def resize_with_pad(img_np, square_length, interp_mode,pad_value=255):
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if c == 1:
            pad = [(0, 0), (0,0)]
            img_np = img_np[..., 0]
        else:
            pad = [(0,0), (0,0), (0,0)]
    else:
        h, w = img_np.shape
        pad = [(0,0), (0,0)]
    if h == w:
        img_resized = cv2.resize(img_np, (square_length, square_length), interpolation=interp_mode)
        return img_resized, tuple(pad)
    elif h > w:
        scale = square_length / h
        target_w = int(w * scale)
        img_resized = cv2.resize(img_np, (target_w, square_length), interpolation=interp_mode)
        pad[1] = ((square_length - target_w) // 2, (square_length - target_w) - (square_length - target_w) // 2)
    else:
        scale = square_length / w
        target_h = int(h * scale)
        img_resized = cv2.resize(img_np, (square_length, target_h), interpolation=interp_mode)
        pad[0] = ((square_length - target_h) // 2, (square_length - target_h) - (square_length - target_h) // 2)

    pad = tuple(pad)
    img_pad = np.pad(img_resized, pad, mode='constant', constant_values=(pad_value, pad_value))

    return img_pad, pad

def resize_with_pad_back(img_np, pad, origin_size, interp_mode):
    if len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        h, w = img_np.shape
    assert h == w
    h_pad, w_pad = pad[:2]
    if any(h_pad):
        head, tail = h_pad
        img = img_np[head:h-tail, ...]
    elif any(w_pad):
        head, tail = w_pad
        img = img_np[:, head:h-tail, ...]
    else:
        img = img_np
    img_resized = cv2.resize(img, origin_size, interpolation=interp_mode)
    return img_resized


def pad_to_ratio(img_np, ratio, pad_value):
    ratio_w, ratio_h = ratio
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if c == 1:
            pad = [(0, 0), (0,0)]
            img_np = img_np[..., 0]
        else:
            pad = [(0,0), (0,0), (0,0)]
    else:
        h, w = img_np.shape
        pad = [(0,0), (0,0)]
    factor = ratio_h / ratio_w
    if h / w > factor:
        pad_w = int(h / factor - w)
        pad_head = pad_w // 2
        pad_tail = pad_w - pad_head
        pad[1] = (pad_head, pad_tail)
    else:
        pad_h = int(w * factor - h)
        pad_head = pad_h // 2
        pad_tail = pad_h - pad_head
        pad[0] = (pad_head, pad_tail)
    pad = tuple(pad)
    img_pad = np.pad(img_np, pad, mode='constant', constant_values=(pad_value, pad_value))
    return img_pad

def centralize_image(img, mask, size, resample='bilinear', boarder=0.1):
    assert resample in ['bicubic', 'nearest', 'bilinear']
    if resample == 'bicubic':
        resample = Image.BICUBIC
    elif resample == 'bilinear':
        resample = Image.BILINEAR
    elif resample == 'nearest':
        resample = Image.NEAREST
    else:
        assert False, "resample must in ['bicubic', 'nearest', 'bilinear']"
    indices = np.argwhere(mask)
    if indices.size == 0:
        img = img.resize(size, resample=resample)
        mask = np.asarray(Image.fromarray(mask).resize(size, resample=Image.NEAREST))
        return img, mask
    img_w, img_h = img.size
    xmin, xmax = min(indices[..., 1]), max(indices[..., 1])
    ymin, ymax = min(indices[..., 0]), max(indices[..., 0])
    w = xmax - xmin
    h = ymax - ymin
    xmin_boarder = max(xmin - int(w * boarder), 0)
    xmax_boarder = min(xmax + int(w * boarder), img_w-1)
    ymin_boarder = max(ymin - int(h * boarder), 0)
    ymax_boarder = min(ymax + int(h * boarder), img_h-1)
    img = img.crop(box=(xmin_boarder, ymin_boarder, xmax_boarder, ymax_boarder)).resize(size, resample=resample)
    mask = mask[ymin_boarder:ymax_boarder+1, xmin_boarder:xmax_boarder+1]
    mask = np.asarray(Image.fromarray(mask).resize(size, resample=Image.NEAREST))
    return img, mask


def normalize(img_pil, mean=0.5, std=0.5):
    img = TF.to_tensor(img_pil)
    if type(mean) is list:
        assert type(std) is list and len(mean) == 3
        mean = torch.tensor(mean, dtype=img.dtype).reshape([3, 1, 1])
        std = torch.tensor(std, dtype=img.dtype).reshape([3, 1, 1])
    img = (img - mean) / std
    return img


def log_txt_as_img(size_wh, text_list, size=10):
    b = len(text_list)
    imgs = list()
    for bi in range(b):
        img = Image.new("RGB", size_wh, color="white")
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(font='DejaVuSans.ttf', size=size)
        nc = int(40 * (size_wh[0] / 256))
        lines = "\n".join(text_list[bi][start:start + nc] for start in range(0, len(text_list[bi]), nc))
        draw.text((10, 10), lines, fill="black", size=size)
        imgs.append(torchvision.transforms.functional.to_tensor(img)[None])
    img_tensor = torchvision.utils.make_grid(torch.cat(imgs, dim=0))
    return img_tensor