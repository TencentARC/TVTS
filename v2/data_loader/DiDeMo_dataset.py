from base.base_dataset import TextVideoDataset
import pandas as pd
import os


class DiDeMo(TextVideoDataset):
    def _load_metadata(self):
        metadata_dir = 'meta_data'
        split_files = {
            'train': 'didemo/DiDeMo_train.tsv',
            'val': 'didemo/DiDeMo_test.tsv',  # there is no test
            'test': 'didemo/DiDeMo_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        rel_video_fp = sample[1]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[0]
