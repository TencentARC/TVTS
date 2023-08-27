from base.base_dataset import TextVideoDataset
import pandas as pd
import os


class LSMDC(TextVideoDataset):
    def _load_metadata(self):
        metadata_dir = 'meta_data'
        split_files = {
            'train': 'lsmdc/LSMDC16_annos_training_real.csv',
            'val': 'lsmdc/LSMDC16_challenge_1000_publictect.csv',  # there is no test
            'test': 'lsmdc/LSMDC16_challenge_1000_publictect.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        video_fp = sample[0]
        sub_path = video_fp.split('.')[0]
        remove = sub_path.split('_')[-1]
        sub_path = sub_path.replace('_' + remove, '/')
        rel_video_fp = sub_path + video_fp + '.avi'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        assert os.path.exists(full_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[-1]
