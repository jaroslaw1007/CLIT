import argparse
import pathlib
import copy
import os
import cv2

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms

import models
import utils

def parse_indices(img_path):
    coord_indices = []

    img = cv2.imread(img_path)
    h, w, c = img.shape

    def draw_circle(event, x, y, flags, param):
        global mouseX, mouseY

        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 0, (255, 0, 0), -1)
            mouseX, mouseY = x, y
            coord_indices.append(mouseY * w + mouseX)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', draw_circle)

    while (1):
        cv2.imshow('Image', img)
        cv2.resizeWindow('Image', 1200, 1000)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            break

    cv2.destroyAllWindows()

    return coord_indices

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

def read_img(img_path):
    return transforms.ToTensor()(
        transforms.ToPILImage()(transforms.ToTensor()(Image.open(img_path).convert('RGB')))
    )

def save_img(img, img_path):
    plt.imshow(img, cmap='viridis')
    plt.colorbar()
    plt.savefig(img_path)
    plt.close()

def system_scale(scale, scale_base=4):
    scale_list = []
    s = copy.copy(scale)

    if s <= scale_base:
        scale_list.append(s)
    else:
        scale_list.append(scale_base)
        s = s / scale_base
        while s > 1:
            if s >= scale_base:
                scale_list.append(scale_base)
            else:
                scale_list.append(s)
            s = s / scale_base

    return scale_list

def visualize(model, inp, coords, cell, save_dir, visual_indices):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    attn_dir = save_dir / 'attn_maps'
    attn_dir.mkdir(exist_ok=True)

    inp = inp.cuda()

    for idx in range(len(coords)):
        coords[idx] = coords[idx].cuda()
        
    cell = cell.cuda()

    model.eval()

    data_norm = {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    with torch.no_grad():
        attn_maps = model.chop_forward(
            (inp - inp_sub) / inp_div, coords, cell, 30000, visual_indices
        )

    for i in range(len(attn_maps)):
        for j in range(attn_maps[i].shape[0]):
            for k in range(attn_maps[i][j].shape[-1]):
                save_img(attn_maps[i][j, ..., k].cpu().numpy(), os.path.join(attn_dir, \
                        'att_map_iter{}_idx{}_head{}.png'.format(i, j, k)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--model')
    parser.add_argument('--scale', type=float)
    args = parser.parse_args()

    img_path = args.img_path # 'visual_imgs/0370.png'
    scale = args.scale

    visual_indices = parse_indices(img_path)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    hr_img = read_img(img_path)
    hr_h, hr_w = hr_img.shape[-2:]
    lr_img = resize_fn(hr_img, (round(hr_h / scale), round(hr_w / scale)))

    inp = lr_img.unsqueeze(0)

    inp_h, inp_w = inp.shape[-2:]

    scale_list = system_scale(scale)

    coords = []

    for idx in range(len(scale_list)):
        h = round(inp_h * np.prod(scale_list[:idx + 1]))
        w = round(inp_w * np.prod(scale_list[:idx + 1]))

        coord = utils.make_coord((h, w), flatten=False).unsqueeze(0)

        coords.append(coord)

    cell = torch.ones(2).unsqueeze(0)
    cell[:, 0] *= 2. / hr_h
    cell[:, 1] *= 2. / hr_w

    save_dir = args.model.split('.pth')[0]

    visualize(model, inp, coords, cell, save_dir, visual_indices)
