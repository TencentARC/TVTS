from base import BaseDataLoader, BaseDataLoaderExplicitSplit, DistBaseDataLoaderExplicitSplit, \
    MultiDistBaseDataLoaderExplicitSplit, BaseMultiDataLoader
from video_transforms.image_transforms import init_transform_dict
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.DiDeMo_dataset import DiDeMo
from data_loader.LSMDC_dataset_our import LSMDC
from data_loader.SSV2_mc_dataset import SSV2_mc
from data_loader.WebVid_dataset import WebVid
from data_loader.HMDB51_dataset import HMDB51
from data_loader.UCF101_dataset import UCF101
from data_loader.K400_dataset import Kinetics400
from data_loader.YTTemporal_dataset import YTTemporal


def dataset_loader(dataset_name,
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
                   patches_per_frame=196,
                   mask_ratio=0.):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader,
        patches_per_frame=patches_per_frame,
        mask_ratio=mask_ratio,
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "YTTemporal":
        dataset = YTTemporal(**kwargs)
    elif dataset_name == "WebVid":
        dataset = WebVid(**kwargs)
    elif dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "DiDeMo":
        dataset = DiDeMo(**kwargs)
    elif dataset_name == "LSMDC":
        dataset = LSMDC(**kwargs)
    elif dataset_name == "HMDB51":
        dataset = HMDB51(**kwargs)
    elif dataset_name == "UCF101":
        dataset = UCF101(**kwargs)
    elif dataset_name == "Kinetics400":
        dataset = Kinetics400(**kwargs)
    elif dataset_name == "SSV2_mc":
        dataset = SSV2_mc(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 patches_per_frame=196,
                 mask_ratio=0.,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader, patches_per_frame, mask_ratio)

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class DistTextVideoDataLoader(DistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 patches_per_frame=196,
                 mask_ratio=0.,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader, patches_per_frame, mask_ratio)

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class MultiDistTextVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 patches_per_frame=196,
                 mask_ratio=0.,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader, patches_per_frame, mask_ratio)

        super().__init__(args, dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)
