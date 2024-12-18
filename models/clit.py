import numpy as np

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

@register('clit')
class CLIT(nn.Module):
    def __init__(
        self,
        encoder_spec,
        imnet_spec,
        pb_spec=None,
        base_dim=192,
        head=8,
        r=3,
        imnet_num=1,
        conv_num=1,
        is_cell=True,
        local_attn=True,
        verbose=False,
    ):
        super().__init__()

        self.dim = base_dim
        self.head = head
        self.r = r
        self.imnet_num = imnet_num
        self.conv_num = conv_num
        self.is_cell = is_cell
        self.local_attn = local_attn
        self.verbose = verbose

        self.encoder = models.make(encoder_spec)

        self.conv_ch = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)

        self.conv_vs = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1) for _ in range(self.conv_num)
        ])

        if self.local_attn:
            self.conv_qs = nn.ModuleList([
                nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                for _ in range(self.conv_num)
            ])

            self.conv_ks = nn.ModuleList([
                nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                for _ in range(self.conv_num)
            ])

            self.is_pb = True if pb_spec else False

            if self.is_pb:
                self.pb_encoder = models.make(pb_spec, args={'head': self.head}).cuda()
        else:
            self.r = 0

        self.r_area = (2 * self.r + 1)**2

        imnet_in_dim = self.dim * self.r_area + 2 if self.is_cell else self.dim * self.r_area

        self.imnets = nn.ModuleList([
            models.make(
                imnet_spec,
                args={'in_dim': imnet_in_dim}
            ) for _ in range(self.imnet_num)
        ])

    def gen_feat(self):
        if self.prev_feat is None:
            self.feat = self.encoder(self.inp)
            self.feat = self.conv_ch(self.feat)
            self.init_feat = self.feat.clone()
        else:
            self.feat = self.prev_feat.clone()

        if self.local_attn:
            self.feat_q = self.conv_qs[self.conv_idx](self.feat)
            self.feat_k = self.conv_ks[self.conv_idx](self.feat)

        self.feat_v = self.conv_vs[self.conv_idx](self.feat)

        return self.feat

    def query_rgb(self, sample_coord, cell=None):
        feat = self.feat

        bs, q_sample, _ = sample_coord.shape

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        # field radius (global: [-1, 1])
        rh = 2 / feat.shape[-2]
        rw = 2 / feat.shape[-1]

        r = self.r

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
            coord_lr, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                self.feat_q, sample_coord_.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            sample_feat_q = sample_feat_q.reshape(
                bs, q_sample, 1, self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                self.feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1)
            sample_feat_k = sample_feat_k.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            ).permute(0, 1, 3, 4, 2)

        sample_feat_v = F.grid_sample(
            self.feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        # b, q, 49, 2
        rel_coord = sample_coord_ - sample_coord_k
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        if self.local_attn:
            # b, q, h, 1, r_area -> b, q, r_area, h
            attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                bs, q_sample, self.head, self.r_area
            ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

            if self.is_pb:
                _, pb = self.pb_encoder(rel_coord)
                attn = F.softmax(torch.add(attn, pb), dim=-2)
            else:
                attn = F.softmax(attn, dim=-2)

            attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1)
            sample_feat_v = sample_feat_v.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            )
            sample_feat_v = torch.mul(sample_feat_v, attn).reshape(bs, q_sample, self.r_area, -1)

        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets[self.im_idx](feat_in)

        if self.prev_pred is None:
            self.prev_pred = pred
        else:
            pred = pred + self.prev_pred * 0.75
            self.prev_pred = pred

        pred = pred + F.grid_sample(self.inp, sample_coord_.flip(-1), mode='bilinear',\
                                    padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        if self.local_attn and self.verbose:
            return pred, attn[0, :, :, :, 0].reshape(-1, 2 * r + 1, 2 * r + 1, self.head)
            
        return pred

    def cascaded_forward(self, inp, coords, sample_coord, cell):
        preds = []

        self.im_idx = 0
        self.conv_idx = 0
        self.prev_feat = None
        self.prev_pred = None

        for idx in range(len(coords)):
            bs, h, w, _ = coords[idx].shape

            self.gen_feat()
            if idx < len(coords) - 1:
                # b, c, q, 1
                coord = coords[idx].clone()
                coord = coord.reshape(bs, -1, 2).unsqueeze(2)
                prev_feat = F.grid_sample(
                    self.init_feat, coord.flip(-1), mode='bilinear', align_corners=False
                )
                self.prev_feat = prev_feat.reshape(bs, -1, h, w)

            pred = self.query_rgb(sample_coord, cell)

            self.im_idx += 1
            self.conv_idx += 1

            preds.append(pred)

        return preds

    def forward(self, inp, coords, sample_coord, cell):
        self.inp = inp

        return self.cascaded_forward(inp, coords, sample_coord, cell)

    def chop_forward(self, inp, coords, cell, eval_bsize=100000, visual_indices=None):
        self.inp = inp

        bs = coords[-1].shape[0]
        hr_coord = coords[-1].clone()
        hr_coord = hr_coord.reshape(bs, -1, 2)
        n = hr_coord.shape[1]

        self.prev_feat = None
        self.prev_pred = None
        prev_pred = None

        preds = []
        attns = []

        self.im_idx = 0
        self.conv_idx = 0

        for idx in range(len(coords)):
            self.gen_feat()
            if idx < len(coords) - 1:
                h, w = coords[idx].shape[1:3]
                coord = coords[idx].clone()
                coord = coord.reshape(bs, -1, 2).unsqueeze(2)
                prev_feat = F.grid_sample(
                    self.init_feat, coord.flip(-1), mode='bilinear', align_corners=False
                )
                self.prev_feat = prev_feat.reshape(bs, -1, h, w)

            if visual_indices is not None:
                self.verbose = True
                sample_coord = hr_coord[:, visual_indices, :]

                pred, attn = self.query_rgb(sample_coord, cell)

                attns.append(attn)
            else:
                cur_preds = []
                prev_preds = []
                ql = 0

                while ql < n:
                    qr = min(ql + eval_bsize, n)

                    sample_coord = hr_coord[:, ql:qr, :]

                    if prev_pred is None:
                        self.prev_pred = None
                    else:
                        self.prev_pred = prev_pred[:, ql:qr, :]

                    pred = self.query_rgb(sample_coord, cell)
                    cur_preds.append(pred)
                    prev_preds.append(self.prev_pred)

                    ql = qr

                cur_pred = torch.cat(cur_preds, dim=1)
                prev_pred = torch.cat(prev_preds, dim=1)
                preds.append(cur_pred)

            self.im_idx += 1
            self.conv_idx += 1

        if len(attns) > 0:
            return attns
        else:
            return preds[-1]
