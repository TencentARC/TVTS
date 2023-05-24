from base.base_dataset import TextImageDataset
import pandas as pd
import os
import json
import numpy as np
import random
import zlib
from torchvision import transforms


def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict


class ConceptualCaptions3M(TextImageDataset):
    """
    Conceptual Captions dataset. Split files are specific to my download regime.
    """

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data'
        split_files = {
            'train': 'cc3m_train.tsv',
            'val': 'cc3m_val.tsv',  # there is no test
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

        # important!!!! image transform is not equal to video transform!
        self.image_transforms = init_transform_dict()[self.split]

    def _get_video_path(self, sample):
        # conceptual captions uses this hashing to create the filename
        rel_dir = 'training'
        if self.split != 'train':
            rel_dir = 'validation'
        rel_fp = os.path.join(rel_dir, sample[1])
        # rel_fp = os.path.join(rel_dir, str(zlib.crc32(sample['thumbnailUrl'].encode('utf-8')) & 0xffffffff))
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        # for compatible with YT joint training
        return [sample[0]]
