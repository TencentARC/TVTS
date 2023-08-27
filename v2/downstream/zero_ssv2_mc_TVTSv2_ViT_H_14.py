import sys

sys.path.append('/path/to/TVTS/v2')

import argparse
import torch
from tqdm import tqdm
import data_loader.data_loader as module_data
import model.metric as module_metric

from parse_config import ConfigParser
import pandas as pd
import numpy as np
from sacred import Experiment
import transformers
from utils.util import state_dict_data_parallel_fix
from trainer.trainer import verbose
import OpenCLIP
import json
import os

import downstream.model_TVTSv2_ViT_H_14_mc as module_arch
from downstream.model_TVTSv2_ViT_H_14_mc import sim_matrix


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


ex = Experiment('test')


@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['args']['split'] = 'test'
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['sliding_window_stride'] = config._config['sliding_window_stride']
    data_loader = config.initialize('data_loader', module_data)
    tokenizer = OpenCLIP.get_tokenizer('ViT-H-14')

    # build model architecture
    model = config.initialize('arch', module_arch)
    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    print('Loading checkpoint: {}'.format(config._config['arch']['args']['load_checkpoint']))

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # meta_arr = []
    # text_embed_arr = []
    # vid_embed_arr = []
    print(len(data_loader))

    top1, top5, n = 0., 0., 0.
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            # meta_arr.append(data['meta'])
            if tokenizer is not None:
                data['text'] = [tokenizer(opt) for opt in data['text']]
                # 174 x B x 77
                data['text'] = torch.stack(data['text'], dim=0)
                data['text'] = data['text'].view(-1, data['text'].shape[-1])
            data['text'] = data['text'].to(device)
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)
            target = data['label'].to(device)

            text_features, image_features = model(data, return_embeds=True)

            # B x D => B x 1 x D
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.unsqueeze(1)

            # 174 x B x D => B x 174 x D
            text_features = text_features.permute(1, 0, 2)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # [B x 1 x D][B x D x 174]=>[B x 1 x 174]
            logits = 100. * torch.bmm(image_features, text_features.permute(0, 2, 1)).squeeze(1)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += image_features.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100

        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')

    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
