import video_transforms.video_transform as video_transform
from torchvision import transforms


# input: (C, T, H, W) output: (C, T, H, W)
def VideoTransform(mode='train', crop_size=224, backend='v100'):
    if mode == 'train':
        data_transforms = transforms.Compose([
            video_transform.TensorToNumpy(),
            video_transform.Resize(int(crop_size * 1.2)),
            video_transform.RandomCrop(crop_size),
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            video_transform.TensorToNumpy(),
            video_transform.Resize(int(crop_size * 1.2)),  # 256
            video_transform.CenterCrop(crop_size),  # 224
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return data_transforms
