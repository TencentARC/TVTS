import random
import sys

import torch
import io
import os
import cv2
import numpy as np
from PIL import Image
from video_transforms import keys_to_transforms
import decord
from decord import cpu, VideoReader
import imageio
# add for ytt asr clean
import ftfy
import regex as re
import demoji
import editdistance
import tslearn.metrics
import string
from video_transforms.videoaug import VideoTransform
import pims


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_keys: list,
            image_size: int,
            names: list,
            text_column_name: str = "",
            remove_duplicate=True,
            max_text_len=40,
            draw_false_image=0,
            draw_false_text=0,
            image_only=False,
            num_frames=1,
            draw_options_text=0,
            backend='v100'
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        if len(names) != 0:
            dataset_name = names[0].split('_')[0]
            split_name = dataset_name
        if torch.distributed.get_rank() == 0:
            print('*' * 100)
            print("video datasets: {}".format(names))
        self.draw_options_text = draw_options_text
        self.num_frames = num_frames
        if torch.distributed.get_rank() == 0:
            print("# frames for base dataset is: {}".format(self.num_frames))
        if split_name in ['cc3m', 'webvid', 'yttemporal']:
            if torch.distributed.get_rank() == 0:
                print("no arrow available for {}, load from disk".format(names[0]))
        else:
            print("not support video dataset")
        self.video_transform = VideoTransform(
            mode=self.split, crop_size=image_size, backend=backend)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', str(sample['video_id']) + '.mp4'), str(
            sample['video_id']) + '.mp4'

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        imgs, idxs, vlen = read_frames_decord(
            abs_fp, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return imgs

    def get_video(self, sample):
        imgs = self.get_raw_video(sample).permute(1, 0, 2, 3)  # to cthw
        imgs_tensor = [self.video_transform(
            imgs).permute(1, 0, 2, 3)]  # to tchw
        return imgs_tensor

    def get_false_video(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        imgs = self.get_raw_video(sample).permute(1, 0, 2, 3)  # to cthw
        assert imgs.size(1) == self.num_frames
        imgs_tensor = [self.video_transform(
            imgs).permute(1, 0, 2, 3)]  # to tchw
        return {f"false_image_{rep}": imgs_tensor}

    def _get_caption(self, sample):
        caption = sample['captions'][0]
        return caption

    def get_text(self, raw_index, sample):
        text = self._get_caption(sample)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        return {
            "text": (text, encoding),
            "img_index": raw_index,
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        text = self._get_caption(sample)
        encoding = self.tokenizer(
            text,
            # padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            sample = self.metadata.iloc[index]
            try:
                video_tensor = self.get_video(sample)
                ret = {
                    "image": video_tensor,
                    "img_index": index,
                    "cap_index": index,
                    "raw_index": index,
                }
                if not self.image_only:
                    txt = self.get_text(index, sample)
                    ret.update(
                        {"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_video(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(
                    f"Error while read file idx {sample.name} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.metadata) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {
            k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                    len(size) == 4
            ), f"Collate error, an image should be in shape of (T, 3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[2] for i in img_sizes])
            max_width = max([i[3] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, self.num_frames,
                            3, max_height, max_width)
                for _ in range(view_size)
            ]
            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        continue
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, :, : orig.shape[-2],
                        : orig.shape[-1]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]]
                     for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]]
                         for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size *
                                              (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size *
                                           (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(
                        _attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(
                    input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(
        start=0, stop=vlen, num=acc_samples + 1).astype(int)
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
    frame_idxs = sample_frames(
        num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
    frames = torch.stack(frames)
    cap.release()
    return frames, success_idxs, vlen


def read_frames_decord(video_path, num_frames, mode='train', fix_start=None):
    if mode in ['train', 'val']:
        sample = 'rand'
    else:
        sample = 'uniform'
    video_reader = decord.VideoReader(
        video_path, width=512, height=512, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    frame_idxs = sample_frames(
        num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs).byte()
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def video_clip_reader_cat(video_path, begin_time_all, end_time_all, duration, num_frames, num_clips, order):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    average_fps = vlen / duration

    clip_len = (end_time_all[-1] - begin_time_all[0]) * average_fps
    frame_idxs = sample_frames(num_frames * num_clips, int(clip_len), sample='rand')
    frames = []
    success_idxs = []

    rel_index = int(begin_time_all[0] * average_fps)
    rel_index = max(rel_index, 0)

    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rel_index + index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass

    frames = torch.stack(frames)
    cap.release()

    if frames.size(0) != num_frames * num_clips:
        Exception(RuntimeError)
    return frames


def video_clip_reader_cat_decord(video_path, begin_time_all, end_time_all, duration, num_frames, num_clips, order):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)

    average_fps = vlen / duration

    clip_len = (end_time_all[-1] - begin_time_all[0]) * average_fps

    frame_idxs = sample_frames(num_frames * num_clips, int(clip_len), sample='rand')

    rel_index = int(begin_time_all[0] * average_fps)
    rel_index = max(rel_index, 0)

    frame_idxs = [idx + rel_index for idx in frame_idxs]

    video_reader.seek(0)
    frames = video_reader.get_batch(frame_idxs)
    # T x C x H x W
    frames = frames.permute(0, 3, 1, 2)

    if frames.size(0) != num_frames * num_clips:
        Exception(RuntimeError)

    return frames


def fast_decode(video_path, num_frames, mode='train', fix_start=None, fps=30):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    max_len = vlen / 30
    num_sec = num_frames / float(fps)
    size = 224
    crop_only = True
    random_flip = True
    start_seek = random.randint(0, int(max(max_len, max_len - num_sec)))
    cmd = (
        ffmpeg
        .input(video_path, ss=start_seek, t=num_sec + 0.1)
        .filter('fps', fps=fps)
    )
    if mode == 'train':
        aw, ah = random.uniform(0, 1), random.uniform(0, 1)
    else:
        aw, ah = 0.5, 0.5
    if crop_only:
        cmd = (
            cmd.crop('(iw - {})*{}'.format(size, aw),
                     '(ih - {})*{}'.format(size, ah),
                     str(size), str(size))
        )
    else:
        cmd = (
            cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                     '(ih - min(iw,ih))*{}'.format(ah),
                     'min(iw,ih)',
                     'min(iw,ih)')
            .filter('scale', size, size)
        )
    if random_flip and random.uniform(0, 1) > 0.5:
        cmd = cmd.hflip()
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, size, size, 3])
    video = torch.from_numpy(video)
    video = video.permute(3, 0, 1, 2)
    return video, _, _


def align_using_dtw(input_asr, grover_output, radius_perc=0.1, radius_abs=32):
    """
    :param input_asr: List of words
    :param grover_output: List of words also, could be different size
    :param radius_perc: Percent of input ASR
    :param radius_abs: Absolute ntokens
    :return:
    """
    max_radius = int(max(len(input_asr) * radius_perc, radius_abs))
    # sometimes grover just keeps going
    if len(grover_output) > len(input_asr):
        grover_output = grover_output[:len(input_asr) + max_radius]

    # DONT give the alignment freedom if it's at the end of a sequence to just "give up" by padding with zeros
    # Default value is high
    auto2other = np.zeros((len(input_asr), len(
        grover_output)), dtype=np.float32) + 9999.0

    def _preprocess_text(x):
        return x.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    input_asr_pre = [_preprocess_text(x) for x in input_asr]
    input_gro_pre = [_preprocess_text(x) for x in grover_output]
    for a_idx, a in enumerate(input_asr_pre):
        start = max(a_idx - max_radius, 0)
        end = min(a_idx + max_radius, len(input_gro_pre))
        for o_idx in range(start, end):
            o = input_gro_pre[o_idx]
            auto2other[a_idx, o_idx] = editdistance.eval(a, o)

    idxs, score = tslearn.metrics.dtw_path_from_metric(
        auto2other, metric='precomputed')
    denoised_out = [[] for x in input_asr]
    has_seen = -1
    for idx1, idx2 in idxs:
        if (idx1 >= len(input_asr)) or (idx2 >= len(grover_output)):
            break
        if idx2 > has_seen:
            # Basically don't add if it's a duplicate -- a grover output that matches to 2 things
            # This often leads to slightly weird results because we really should match the next thing, but we instead matched the first thing
            # e.g.
            # input_asr_pre = ['much', 'of', 'a', 'pancake', 'waffle', 'person', 'so', 'i', 'love', 'a']
            # input_gro_pre = ['much', 'of', 'a', 'pancakewaffle', 'person', 'so', 'i', 'love', 'a', 'good']
            # but we align pancakewaffle-> pancake and person -> waffle AND person -> person
            denoised_out[idx1].append(grover_output[idx2])
        has_seen = idx2
    return [' '.join(x) for x in denoised_out]


def clean_subtitles(subtitle_dicts):
    """
    :param subtitle_dicts: {'word': X, 'time': Y}
    :return:
    """
    # Remove &gt;&gt; maybe using ftfy or something
    new_dicts = []
    for x in subtitle_dicts:
        if x['word'].startswith('&') or x['word'].endswith(';'):
            continue
        fixed_word = ftfy.ftfy(x['word'])
        if len(fixed_word) == 0:
            continue
        x['word'] = fixed_word
        new_dicts.append(x)
    return new_dicts


def clean_description(text):
    # Strip emojis first
    all_emojis = demoji.findall(text)
    for k, v in all_emojis.items():
        text = text.replace(k, f'[{v}]'.replace(' ', ''))
    text = text.strip()

    # Remove URLs
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
    text = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        "%", text)

    text = re.sub(' +', ' ', text)  # Probably should have done
    text = re.sub('\s*\n+', '\n', text)
    text = text.strip()
    return text
