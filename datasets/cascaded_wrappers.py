import random
import math
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

def sample_system_scale(scale_factor, scale_base):
    scale_it = []
    s = copy.copy(scale_factor)

    if s <= scale_base:
        scale_it.append(s)
    else:
        scale_it.append(scale_base)
        s = s / scale_base

        while s > 1:
            if s >= scale_base:
                scale_it.append(scale_base)
            else:
                scale_it.append(s)
            s = s / scale_base

    return scale_it

@register('sr-implicit-paired-cascaded')
class SRImplicitPairedCascaded(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        batch_size=16,
        scale_base=2,
        sample_q=None,
        window_size=0,
        augment=False
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.batch_size = batch_size
        self.scale_base = scale_base
        self.sample_q = sample_q
        self.window_size = window_size
        self.augment = augment

    def collate_fn(self, datas):
        s = datas[0]['img_hr'].shape[-2] // datas[0]['img_lr'].shape[-2]

        self.scale_it = sample_system_scale(s, self.scale_base)

        coords = []
        hr_list = []
        lr_list = []

        if self.inp_size is None:
            # batch_size: 1
            for data in (datas):
                lr_h = data['img_lr'].shape[-2]
                lr_w = data['img_lr'].shape[-1]

                lr_list.append(data['img_lr'])
                hr_list.append(data['img_hr'])
        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['img_lr'].shape[-2] - self.inp_size)
                w0 = random.randint(0, data['img_lr'].shape[-1] - self.inp_size)
                crop_lr = data['img_lr'][:, h0:h0 + self.inp_size, w0:w0 + self.inp_size]
                hr_size = self.inp_size * s
                h1 = h0 * s
                w1 = w0 * s
                crop_hr = data['img_hr'][:, h1:h1 + hr_size, w1:w1 + hr_size]

                lr_list.append(crop_lr)
                hr_list.append(crop_hr)

        for idx in range(len(self.scale_it)):
            img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
            img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

            coord = make_coord((img_h, img_w),
                               flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
            coords.append(coord)

        hr_h = round(lr_h * np.prod(self.scale_it))
        hr_w = round(lr_w * np.prod(self.scale_it))

        for idx in range(len(hr_list)):
            hr_list[idx] = hr_list[idx][..., :hr_h, :hr_w]

        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                coords[idx] = coord

            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []
            for i in range(len(hr_list)):
                flatten_coord = coords[-1][i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]
                sample_coord.append(sample_flatten_coord)
            sample_coord = torch.stack(sample_coord, dim=0)

        return {'inp': inp, 'gt': hr_rgb, 'coords': coords, 'cell': cell, 'sample_coord': sample_coord}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        return {'img_lr': img_lr, 'img_hr': img_hr}


@register('sr-implicit-downsampled-cascaded')
class SRImplicitDownsampledCascaded(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        batch_size=16,
        scale_base=4,
        scale_min=1,
        scale_max=None,
        sample_q=None,
        k=1,
        window_size=0,
        augment=False,
        phase='train'
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.batch_size = batch_size
        self.scale_base = scale_base
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.k = k
        self.window_size = window_size
        self.augment = augment
        self.phase = phase

        self.counter = 0

    def collate_fn(self, datas):
        coords = []
        hr_list = []

        if self.inp_size is None:
            self.scale_it = sample_system_scale(self.scale_max, self.scale_base)
        else:
            self.scale_it = []
            for idx in range(len(self.scale_max)):
                self.scale_it.append(random.uniform(self.scale_min[idx], self.scale_max[idx]))

        if self.phase == 'train':
            if self.counter % self.k == 0:
                self.counter = 1
            else:
                del self.scale_it[self.counter:]
                self.counter += 1

        if self.inp_size is None:
            # batch_size: 1
            lr_h = math.floor(datas[0]['inp'].shape[-2] / np.prod(self.scale_it) + 1e-9)
            lr_w = math.floor(datas[0]['inp'].shape[-1] / np.prod(self.scale_it) + 1e-9)

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w), flatten=False) \
                                    .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

                coords.append(coord)

            for idx, data in enumerate(datas):
                crop_hr = data['inp'][:, :img_h, :img_w]
                hr_list.append(crop_hr)
        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            img_size_min = 9999

            for idx, data in enumerate(datas):
                img_size_min = min(img_size_min, data['inp'].shape[-2], data['inp'].shape[-1])

            if np.prod(self.scale_it) * self.inp_size > img_size_min:
                img_ratio = img_size_min / (np.prod(self.scale_it) * self.inp_size)
                self.scale_it[0] *= img_ratio

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w),
                                   flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                coords.append(coord)

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['inp'].shape[-2] - img_h)
                w0 = random.randint(0, data['inp'].shape[-1] - img_w)
                crop_hr = data['inp'][:, h0:h0 + img_h, w0:w0 + img_w]
                hr_list.append(crop_hr)

        lr_list = [resize_fn(hr_list[i], (lr_h, lr_w)) for i in range(len(hr_list))]
        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                coords[idx] = coord

            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []
            for i in range(len(hr_list)):
                flatten_coord = coords[-1][i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]
                sample_coord.append(sample_flatten_coord)
            sample_coord = torch.stack(sample_coord, dim=0)

        return {'inp': inp, 'gt': hr_rgb, 'coords': coords, 'cell': cell, 'sample_coord': sample_coord}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)

        return {'inp': img}
