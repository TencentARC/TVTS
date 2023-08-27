from base.base_dataset import TextVideoDataset
import pandas as pd
import os
import json
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms


class Kinetics400(TextVideoDataset):
    def _load_metadata(self):
        # download specific
        metadata_dir = 'meta_data'
        split_files = {
            'train': 'k400/prompt/kinetics400_train_list_videos.tsv',
            'val': 'k400/prompt/kinetics400_val_list_videos.tsv',  # there is no test
            'test': 'k400/prompt/kinetics400_val_list_videos.tsv',
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_table(os.path.join(metadata_dir, target_split_fp), sep='\t')

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

    def _get_video_path(self, sample):
        rel_path = sample[0]
        class_name, video_name = rel_path.split('/')
        if video_name.endswith('.mkv'):
            video_name = video_name[:-4] + '.mp4'
        rel_path = video_name
        if self.split == 'train':
            return os.path.join(self.data_dir, 'train/train', rel_path), rel_path
        else:
            return os.path.join(self.data_dir, 'videos_val', rel_path), rel_path

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        # caption = self._get_caption(sample)
        # assert os.path.exists(video_fp)
        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample, fix_start=fix_start)
        except:
            if video_loading == 'strict':
                raise ValueError(f'Video loading failed for {video_fp}, video loading for this dataset is strict.')
            else:
                print(f'Video loading failed for {video_fp}, video loading for this dataset is {video_loading}.')
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        # generate tube mask
        patches_per_frame, mask_ratio = self.patches_per_frame, self.mask_ratio

        n_keep = int(patches_per_frame * (1 - mask_ratio))
        ind = np.arange(patches_per_frame)
        np.random.shuffle(ind)
        keep_ind = ind[:n_keep]

        meta_arr = {'raw_captions': 'NULL', 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': 'NULL', 'label': sample[1], 'meta': meta_arr, 'keep_ind': keep_ind}
        return data
