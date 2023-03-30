import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

@register('lit')
class LIT(nn.Module):
    def __init__(
        self,
        encoder_spec,
        imnet_spec=None,
        is_cell=True,
        local_ensemble=False,
        local_attn=True,
        base_dim=192,
        head=8,
        pe_spec=None,
        pb_spec=None
    ):
        super().__init__()
        self.is_cell = is_cell
        self.local_ensemble = local_ensemble
        self.local_attn = local_attn

        self.encoder = models.make(encoder_spec)

        self.dim = base_dim
        self.head = head
        self.conv_v = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)
        if self.local_attn:
            self.conv_q = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)
            self.conv_k = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)
            self.is_pb = True if pb_spec else False
            if self.is_pb:
                self.pb_encoder = models.make(pb_spec, args={'head': self.head}).cuda()
            self.r = 3
        else:
            self.r = 0
        self.r_area = (2 * self.r + 1)**2

        self.pe_encoder = models.make(pe_spec).cuda()
        self.conv_freq = nn.Conv2d(
            self.encoder.out_dim, self.pe_encoder.enc_dims // 2, kernel_size=3, padding=1
        )

        if self.is_cell:
            self.imnet = models.make(
                imnet_spec,
                args={'in_dim': (self.dim + self.pe_encoder.enc_dims) * self.r_area + 2}
            )
        else:
            self.imnet = models.make(
                imnet_spec, args={'in_dim': (self.dim + self.pe_encoder.enc_dims) * self.r_area}
            )

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)

        self.feat_v = self.conv_v(self.feat)
        if self.local_attn:
            self.feat_q = self.conv_q(self.feat)
            self.feat_k = self.conv_k(self.feat)

        self.feat_freq = self.conv_freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        bs, q = coord.shape[:2]

        # b, q, 1, 2
        coord = coord.unsqueeze(2)

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # local ensamble - field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        r = self.r
        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)

        areas = []
        preds = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
                coord_k = F.grid_sample(
                    coord_lr, coord_.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)

                # local ensamble
                ensamble_coord = coord - coord_k
                ensamble_coord[:, :, :, 0] *= feat.shape[-2]
                ensamble_coord[:, :, :, 1] *= feat.shape[-1]
                area = torch.abs(ensamble_coord[:, :, 0, 0] * ensamble_coord[:, :, 0, 1])
                areas.append(area + 1e-9)

                if self.local_attn:
                    # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
                    feat_q = F.grid_sample(
                        self.feat_q,
                        coord_.flip(-1),
                        mode='nearest' if self.local_ensemble else 'bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                    feat_q = feat_q.reshape(bs, q, 1, self.head,
                                            self.dim // self.head).permute(0, 1, 3, 2, 4)

                    # b, q, 1, 2 -> b, q, r_area, 2
                    coord_k = coord_k + delta

                    # K - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c -> b, q, r_area, h, c -> b, q, h, c, r_area
                    feat_k = F.grid_sample(
                        self.feat_k, coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
                    feat_k = feat_k.reshape(bs, q, self.r_area, self.head,
                                            self.dim // self.head).permute(0, 1, 3, 4, 2)

                    # V - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c
                    feat_v = F.grid_sample(
                        self.feat_v, coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)

                else:
                    feat_v = F.grid_sample(
                        self.feat_v, coord_.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)

                feat_freq = F.grid_sample(
                    self.feat_freq, coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)

                # b, q, r_area, 2
                rel_coord = coord - coord_k
                rel_coord[..., 0] *= feat.shape[-2]
                rel_coord[..., 1] *= feat.shape[-1]

                # b, q, 2
                rel_cell = cell.clone()
                rel_cell[..., 0] *= feat.shape[-2]
                rel_cell[..., 1] *= feat.shape[-1]

                if self.local_attn:
                    # b, q, h, 1, r_area -> b, q, r_area, h
                    similarity = torch.matmul(feat_q, feat_k).reshape(
                        bs, q, self.head, self.r_area
                    ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)
                    if self.is_pb:
                        _, pb = self.pb_encoder(rel_coord)
                        attn = F.softmax(similarity + pb, dim=-2)
                    else:
                        attn = F.softmax(similarity, dim=-2)
                    attn = attn.reshape(bs, q, self.r_area, self.head, 1)
                    feat_v = feat_v.reshape(bs, q, self.r_area, self.head, self.dim // self.head)
                    feat_v = torch.mul(feat_v, attn).reshape(bs, q, self.r_area, -1)

                    attn_map = attn[0, 10, :, 0, :].reshape(2 * r + 1, 2 * r + 1, 1)

                feat_freq = feat_freq.reshape(bs, q, 2 * r + 1, 2 * r + 1, -1)
                feat_freq = torch.fft.fft2(feat_freq, dim=(2, 3), norm='ortho')
                feat_freq = feat_freq.reshape(bs, q, self.r_area, -1)

                rel_enc, _ = self.pe_encoder(rel_coord)
                rel_enc.mul_(feat_freq)
                rel_enc = torch.view_as_real(rel_enc)
                rel_enc = rel_enc.reshape(bs, q, self.r_area, -1)

                out = torch.cat([feat_v, rel_enc], dim=-1)

                out = out.reshape(bs, q, -1)

                if self.is_cell:
                    out = torch.cat([out, rel_cell], dim=-1)

                pred = self.imnet(out)
                preds.append(pred)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret +=  F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                        padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)
        return ret, attn_map

    def forward(self, inp, coord, cell):

        self.gen_feat(inp)
        return self.query_rgb(coord, cell)