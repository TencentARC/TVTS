import os
from torchvision import transforms
from transforms import *
from ssv2 import SSVideoClsDataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = 'data/ssv2/sthv2_train_list_videos.txt'
        elif test_mode is True:
            mode = 'test'
            anno_path = 'data/ssv2/sthv2_val_list_videos.txt'
        else:
            mode = 'validation'
            anno_path = 'data/ssv2/sthv2_val_list_videos.txt'

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
