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
from CLIP import clip
from decord import VideoReader, cpu
from torchvision import transforms
from PIL import Image
import cv2
import os

import downstream.model_TVTSv2_ViT_B_16 as module_arch
from downstream.model_TVTSv2_ViT_B_16 import sim_matrix

ex = Experiment('test')


@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['args']['split'] = 'test'
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['sliding_window_stride'] = config._config['sliding_window_stride']
    tokenizer = clip.tokenize

    # build model architecture
    model = config.initialize('arch', module_arch)

    ckpt_path = 'TVTSv2_ViT_B_16.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(ckpt['state_dict'], strict=True)
    print('Loaded checkpoint from {}'.format(ckpt_path))

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    video_path = args.video_path

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n_frames = len(vr)
    n_frames_sample = 12
    sample_idx = np.linspace(0, n_frames - 1, n_frames_sample).astype(int)
    sample_frames = vr.get_batch(sample_idx)
    sample_frames = sample_frames.asnumpy()

    inputs = []
    for i in range(n_frames_sample):
        frame = sample_frames[i]
        frame = Image.fromarray(frame, mode='RGB')
        frame = transform(frame)
        inputs.append(frame)
    input = torch.stack(inputs, dim=0).unsqueeze(0)

    text = 'NULL'
    text = tokenizer([text], truncate=True)

    print('video shape', input.shape)
    # video shape torch.Size([1, 12, 3, 224, 224])

    data = {
        'video': input.to(device),
        'text': text.to(device),
        'keep_ind': torch.arange(196).unsqueeze(0).to(device)  # generate tube mask
    }

    with torch.no_grad():
        _, video_embeds = model(data, return_embeds=True)

    print('video embeds shape', video_embeds.shape)
    # video embeds shape torch.Size([1, 512])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default='configs/zero-msrvtt-vit-b-16.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--video_path', default=None, type=str)

    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
