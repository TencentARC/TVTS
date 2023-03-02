from base.base_dataset_yt import BaseDataset, clean_subtitles, align_using_dtw
from base.base_dataset_yt import video_clip_reader_cat, video_clip_reader_cat_decord
import torch as th
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import json
import math
import ftfy
import itertools
from torchvision.transforms import ToPILImage
import time
import sys
import decord


class YTTemporal(BaseDataset):
    """YTTemporal Video-Text loader."""

    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 ):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["yttemporal_train"]
        elif split == "val":
            names = ["yttemporal_val"]
        elif split == "test":
            names = ["yttemporal_test"]
        super().__init__(data_dir=data_dir, transform_keys=["pixelbert"],
                         image_size=video_params['input_res'],
                         num_frames=video_params['num_frames'],
                         names=['yttemporal'], text_column_name="caption")

        self.metadata = None
        self._load_metadata()
        self.min_time = 4.0
        self.size = 224
        self.fps = 2
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = True
        if self.split == 'train':
            self.center_crop = False
        else:
            self.center_crop = True
        self.benchmark = False
        self.num_candidates = 1
        self.random_flip = True
        self._load_metadata()

        # New
        self.num_clips = 4
        self.n_trans = 4
        self.mask_ratio = 0.75

        self.reader = reader
        if self.reader == 'cv2':
            print('using OpenCV as video reader for %s' % dataset_name)
        elif self.reader == 'decord':
            print('using decord as video reader for %s' % dataset_name)
            decord.bridge.set_bridge('torch')
        elif self.reader == 'pims':
            print('using pims as video reader for %s' % dataset_name)
        else:
            raise Exception('Unknown reader')

    def _load_metadata(self):
        metadata_dir = './meta_data'
        split_files = {
            'train': 'yttemporal_train.csv',  # _1000000
            'val': 'yttemporal_val.csv',  # there is no test
            'test': 'yttemporal_val.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(
            metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata["Name"]

    def get_caption_multi(self, caption_csv, order):
        with open(caption_csv, 'r') as f:
            cap = json.load(f)

        # [{'word': 'hey', 'time': 0.0}, {'word': 'guys', 'time': 0.149}]
        all_text = cap["subtitles"]
        # clean noisy asr text
        all_text = clean_subtitles(all_text)
        vtt = pd.DataFrame(all_text)
        denoised_word_by_word = []
        for x in cap['denoised']:
            # Ftfy just in case
            cleanasr = ftfy.ftfy(x['cleanasr'])
            denoised_word_by_word += cleanasr.split(' ')
        # Align
        vtt['denoised'] = align_using_dtw(vtt['word'], denoised_word_by_word)

        # random choice 10s-15s video clips
        video_len = int(cap["info"]["duration"])

        interval = 1
        segm_length = random.randint(3, 5) * self.n_trans + interval * (self.n_trans - 1)

        try:
            start = random.randint(0, (video_len - segm_length - 1)) + random.random()
            end = min(video_len - 1, start + segm_length)
        except:
            start = 0
            end = video_len - 1

        clip_len = (end - start - interval * (self.n_trans - 1)) / self.n_trans
        start_all = []
        end_all = []

        for i in range(self.n_trans):
            clip_start = start + i * (clip_len + interval)
            clip_end = clip_start + clip_len
            start_all.append(clip_start)
            end_all.append(clip_end)

        text_all = []
        for i in order:
            start = start_all[i]
            end = end_all[i]
            text = ""
            origin_text = ""
            for index, item in enumerate(all_text):
                if float(item['time']) > start and float(item['time']) < end:
                    text += vtt['denoised'][index] + " "
                    origin_text += item['word'] + " "

            if len(text) < 10:
                Exception(IndexError)

            text_all.append(text)

        label = np.arange(self.num_clips)

        return text_all, label, start_all, end_all, video_len

    def get_text(self, sample, order):
        caption_csv = self.get_caption_path(sample)
        text_all, label, start_all, end_all, duration = self.get_caption_multi(caption_csv, order)
        return {"text": text_all}, {"label": label}, start_all, end_all, duration

    def get_caption_path(self, sample):
        # YTTemporal/videos/subset_87/data/xx.mp4 -> YTTemporal/videos/subset_87/annotations/xx.csv
        return os.path.join(self.data_dir, 'videos', sample.split('/')[0], 'annotations',
                            sample.split('/')[-1][:-4] + '.json')

    def _get_video_path(self, sample):
        rel_video_fp = sample
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_raw_video_multi(self, sample, begin_all, end_all, duration, order):
        abs_fp, rel_fp = self._get_video_path(sample)

        if self.reader == 'cv2':
            imgs = video_clip_reader_cat(abs_fp, begin_all, end_all, duration,
                                         self.num_frames, self.num_clips, order)
        elif self.reader == 'decord':
            imgs = video_clip_reader_cat_decord(abs_fp, begin_all, end_all, duration,
                                                self.num_frames, self.num_clips, order)
        else:
            raise Exception('Unknown reader')

        if imgs.size(0) != self.num_frames * self.num_clips:
            raise Exception("video length not enough!", rel_fp)
        if imgs is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return imgs

    def get_video_multi(self, sample, start_all, end_all, duration, order):
        try:
            # C x T x H x W
            imgs_all = self.get_raw_video_multi(sample, start_all, end_all, duration, order).permute(1, 0, 2, 3)
        except:
            print('video loading error', sample)
            imgs_all = th.zeros((3, self.num_frames * self.num_clips, self.size, self.size), dtype=th.uint8)

        # T x C x H x W
        imgs_tensor = self.video_transform(imgs_all).permute(1, 0, 2, 3)
        return imgs_all.permute(1, 0, 2, 3), imgs_tensor

    def get_suite(self, index):
        result = None
        max_try = 5
        try_time = 0

        order = list(range(self.n_trans))

        # ViT-Base
        patches_per_frame = 196
        n_temp = 8
        n_keep = int(patches_per_frame * (1 - self.mask_ratio))

        keep_ind = np.empty((n_temp, n_keep), dtype=np.int64)
        for i in range(n_temp):
            ind = np.arange(patches_per_frame)
            np.random.shuffle(ind)
            keep_ind[i] = ind[:n_keep]

        while result is None:
            try_time += 1
            sample = self.metadata.iloc[index]

            try:
                ret = dict()
                text_all, label, start_all, end_all, duration = self.get_text(sample, order)
                ret.update(label)
                ret.update(text_all)

                imgs_all, imgs_tensor = self.get_video_multi(sample, start_all, end_all, duration, order)

                ret.update({
                    "video": imgs_tensor,
                    "gt_order": 0,
                    "img_index": index,
                    "cap_index": index,
                    "raw_index": index,
                    "keep_ind": keep_ind,
                })
                ret.update({"replica": True if ret["cap_index"] > 0 else False})
                result = True
            except:
                print('load sample %s error, retrying...' % sample)
                index = random.randint(0, len(self.metadata) - 1)

            if try_time > max_try:
                print(f"exceed max time Error while read file idx {sample} in {self.names[0]}")
                sys.exit(-1)

        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
