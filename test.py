import argparse
import pathlib
import time
import yaml

from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
import models
import utils

def eval_psnr(
    loader,
    model,
    name=None,
    data_norm=None,
    eval_type=None,
    eval_bsize=None,
    scale_max=4,
    save_dir=None,
    verbose=False
):
    model.eval()

    if save_dir:
        save_dir = pathlib.Path(save_dir)
        tmp_dir = name + '-results'
        img_dir = save_dir / tmp_dir
        img_dir.mkdir(parents=True, exist_ok=True)

        val_res = utils.Averager()
    else:
        val_res = [utils.Averager() for _ in range(len(loader.dataset.scale_max))]

    if data_norm is None:
        data_norm = {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    log_name = 'PSNR.txt'

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('ssim'):
        log_name = 'SSIM.txt'
        scale = int(eval_type.split('-')[1])
        metric_fn = utils.ssim
    else:
        raise NotImplementedError

    val_time = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')

    IDX = 1

    for batch in pbar:
        for k, v in batch.items():
            if isinstance(batch[k], list):
                for idx in range(len(batch[k])):
                    batch[k][idx] = v[idx].cuda()
            else:
                if v is not None:
                    batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        bs = batch['inp'].shape[0]
        batch_index = torch.arange(bs).unsqueeze(1)

        coords = batch['coords']
        sample_coord = batch['sample_coord']
        cell = batch['cell']

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            if eval_bsize is None:
                preds = model(inp, coords, sample_coord, cell)
            else:
                # cell *= max(scale / scale_max, 1)
                preds = model.chop_forward(inp, coords, cell, eval_bsize)

        end = time.time()
        torch.cuda.synchronize()

        val_time.add(end - start, bs)

        if eval_type is not None:
            hr_h, hr_w = batch['gt'].shape[-2:]
            coord_h, coord_w = batch['coords'][-1].shape[1:3]

            final_pred = preds[-1]
            final_pred = final_pred * gt_div + gt_sub
            final_pred.clamp_(0, 1)

            shape = [bs, coord_h, coord_w, 3]
            final_pred = final_pred.view(*shape).permute(0, 3, 1, 2)
            final_pred = final_pred[..., :hr_h, :hr_w]

            res = metric_fn(final_pred, batch['gt'])
            val_res.add(res.item(), bs)
        else:
            for idx in range(len(preds)):
                pred = preds[idx] * gt_div + gt_sub
                pred = pred.clamp_(0, 1)
                sample_coord = batch['sample_coord']
                sample_coord = sample_coord.unsqueeze(2)

                gt = F.grid_sample(batch['gt'], sample_coord.flip(-1), \
                        mode='nearest', align_corners=False).permute(0, 2, 3, 1)

                gt = gt.reshape(bs, -1, 3)
                res = metric_fn(pred, gt)
                val_res[idx].add(res.item(), bs)

        if save_dir:
            # transforms.ToPILImage()(final_pred.squeeze().cpu()
            #                        ).save(img_dir / f'{str(IDX).zfill(4)}_x{scale}.png')
            IDX += 1
            with open(img_dir / log_name, mode='a') as f:
                print('result: {:.6f} | time: {:.6f}'.format(res.item(), end - start), file=f)

            if verbose:
                des = 'val: {:.4f}'.format(val_res.item())
                pbar.set_description(des)

        else:
            if verbose:
                des = 'val 1:{:.4f}'.format(val_res[0].item())

                for idx in range(1, len(val_res)):
                    des += ' {}:{:.4f}'.format(idx + 1, val_res[idx].item())

                pbar.set_description(des)

    if save_dir:
        with open(img_dir / log_name, mode='a') as f:
            print('AVG-result: {:.6f}'.format(val_res.item()), file=f)
            print('AVG-Time: {:.6f}'.format(val_time.item()), file=f)

        return val_res.item()
    else:
        return [val_res[idx].item() for idx in range(len(val_res))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--name', default='div2k')
    parser.add_argument('--scale_max', default='4')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(
        loader,
        model,
        args.name,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max=int(args.scale_max),
        save_dir=args.model.split('.pth')[0],
        verbose=True
    )

    print('result: {:.4f}'.format(res))
