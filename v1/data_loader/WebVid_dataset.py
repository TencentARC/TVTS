from base.base_dataset import TextVideoDataset
import pandas as pd
import os
import json
import numpy as np
import random


class WebVid(TextVideoDataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """

    def _load_metadata(self):
        # metadata_dir = os.path.join(self.metadata_dir, 'meta_data')
        metadata_dir = './meta_data'
        split_files = {
            'train': 'webvid_train.tsv',
            'val': 'webvid_val.tsv',  # there is no test
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        # metadata['caption'] = metadata['name']
        # del metadata['name']
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        # self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.mp4'
        # rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # return sample[0]
        # for compatible with YT joint training
        return [sample[0]]
