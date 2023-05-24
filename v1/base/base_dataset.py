import random
import cv2
import os
import numpy as np
import torch
import random
from PIL import Image, ImageFile
from abc import abstractmethod
import tarfile
from io import BytesIO
from torchvision import transforms
from torch.utils.data import Dataset, get_worker_info
from video_transforms.videoaug import VideoTransform
import sys
import decord


class TextVideoDataset(Dataset):
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
                 reader='cv2'
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        # self.transforms = tsfms
        self.transforms = VideoTransform(mode=self.split)
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride

        # New
        self.reader = reader
        if self.reader == 'cv2':
            print('using OpenCV as video reader for %s' % dataset_name)
        else:
            print('using decord as video reader for %s' % dataset_name)
            decord.bridge.set_bridge('torch')

        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)
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
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        # this makes 3D Conv => 2D Conv
        # final = final.repeat_interleave(2, dim=0)

        # ViT-Base
        mask_ratio = 0
        patches_per_frame = 196
        n_temp = 8
        n_keep = int(patches_per_frame * (1 - mask_ratio))

        keep_ind = np.empty((n_temp, n_keep), dtype=np.int64)
        for i in range(n_temp):
            ind = np.arange(patches_per_frame)
            np.random.shuffle(ind)
            keep_ind[i] = ind[:n_keep]

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'keep_ind': keep_ind, 'meta': meta_arr}
        return data


class TextImageDataset(TextVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)
        # assert os.path.exists(video_fp)
        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            print('loading error', video_fp)
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # define image transform in _load_meta()!!!
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        # this makes 3D Conv => 2D Conv
        img = img.unsqueeze(0).repeat_interleave(2, dim=0)

        # ViT-Base
        mask_ratio = 0
        patches_per_frame = 196
        n_temp = 8
        n_keep = int(patches_per_frame * (1 - mask_ratio))

        keep_ind = np.empty((n_temp, n_keep), dtype=np.int64)
        for i in range(n_temp):
            ind = np.arange(patches_per_frame)
            np.random.shuffle(ind)
            keep_ind[i] = ind[:n_keep]

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'keep_ind': keep_ind, 'meta': meta_arr}
        return data


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).byte()
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass

    frames = torch.stack(frames).float()
    cap.release()
    return frames, success_idxs


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)

    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)

    video_reader.seek(0)
    frames = video_reader.get_batch(frame_idxs).byte()
    # T x C x H x W
    frames = frames.permute(0, 3, 1, 2)

    return frames, None


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    'cv2': read_frames_cv2,
    'decord': read_frames_decord
}
