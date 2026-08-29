from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose

from src.data_getter.base_getter import BaseDatasetGetter

import logging
log = logging.getLogger(__name__)

class SpeechCommandsDataset(Dataset):
    def __init__(self, fo_dataset, transform=None, label_to_idx=None):
        self.ds = fo_dataset
        self.transform = transform

        self.ids = self.ds.values("id")
        self.filepaths = self.ds.values("filepath")
        self.label_strs = self.ds.values("label")

        self.unique_labels = sorted(list(set(self.label_strs)))
        if label_to_idx is None:
            self.label_to_idx = {label.strip().lower(): i for i, label in enumerate(self.unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.targets = [self.label_to_idx[label.strip().lower()] for label in self.label_strs]
        self.num_classes = len(self.unique_labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.targets[idx]
        data = np.load(fp).astype(np.float32)
        data = torch.from_numpy(data)
        data = data.unsqueeze(0)

        if self.transform:
            data = self.transform(data)

        return data, label

class GoogleSpeechKWS(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500, label_per_device=4, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        import fiftyone as fo

        try:
            train_fo = fo.Dataset.from_dir(
                dataset_dir='./data/kws/train',
                dataset_type=fo.types.FiftyOneDataset,
            )
            test_fo = fo.Dataset.from_dir(
                dataset_dir='./data/kws/test',
                dataset_type=fo.types.FiftyOneDataset,
            )
        except FileNotFoundError:
            train_fo = fo.Dataset.from_dir(
                dataset_dir='../../data/kws/train',
                dataset_type=fo.types.FiftyOneDataset
            )
            test_fo = fo.Dataset.from_dir(
                dataset_dir='../../data/kws/test',
                dataset_type=fo.types.FiftyOneDataset,
            )

        transform = Compose([Normalize(-0.4, 37)])

        self._total_train_dataset = SpeechCommandsDataset(train_fo, transform=transform)
        self._total_test_dataset = SpeechCommandsDataset(test_fo, label_to_idx=self._total_train_dataset.label_to_idx, transform=transform)

        self._total_classes = self._total_test_dataset.num_classes
        log.info(f'There are {self._total_classes} classes in current dataset')


if __name__ == '__main__':
    kws = GoogleSpeechKWS(num_devices=10, label_per_device=5, data_per_device=600)
    kws._download_dataset()
    print(len(kws._total_train_dataset))
    print(len(kws._total_test_dataset))
    sample_data, _ = kws._total_train_dataset.__getitem__(0)
    print(sample_data)
    print(sample_data.shape)

    # kws._split_subdataset_train()
    # kws._split_subdataset_test()
    # #kws._convert_to_c_code_after_split('./dataset/kws/', 'kws')
    #
    # kws._allocate_labels_to_devices(method='ordered', loop_step=1)
    #
    # kws._allocate_train_data_to_devices(method='dirichlet', alpha=1)
    # kws._allocate_test_valid_data(test_data_size=600, valid_data_size=0)
    # kws._convert_to_c_code_after_allocate('./dataset/kws/', 'kws')
    #
    # kws._generate_dataloader()
    # kws._training_dataset_visualization(save_path='./dataset/kws/', fig_name='dirichlet')

