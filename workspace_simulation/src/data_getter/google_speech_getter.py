from __future__ import annotations

from torch.utils.data import Dataset
import torch
import numpy as np
import logging
log = logging.getLogger(__name__)

from src.data_getter.base_getter import BaseDatasetGetter

ALL_CLASSES = [
    "_silence_",
    "Yes", "No", "Up", "Down", "Left", "Right", "On",
    "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine", "Bed", "Bird",
    "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow",  # v0.1
    "Backward", "Forward", "Follow", "Learn", "Visual"  # v0.2 addition
]

class SpeechCommandsDataset(Dataset):
    def __init__(self, fo_dataset, transform=None):
        self.ds = fo_dataset
        self.transform = transform

        self.ids = self.ds.values("id")
        self.filepaths = self.ds.values("filepath")
        self.label_strs = self.ds.values("label")

        self.label_to_idx = {label.strip().lower(): i for i, label in enumerate(ALL_CLASSES)}

        self.targets = [self.label_to_idx[label.strip().lower()] for label in self.label_strs]
        self.num_classes = len(set(self.targets))


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

class GoogleSpeechCommand(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500, label_per_device=4, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        import fiftyone as fo

        try:
            train_fo = fo.Dataset.from_dir(
                dataset_dir='./data/kws-all/train',
                dataset_type=fo.types.FiftyOneDataset,
            )
            test_fo = fo.Dataset.from_dir(
                dataset_dir='./data/kws-all/test',
                dataset_type=fo.types.FiftyOneDataset,
            )
        except FileNotFoundError:
            train_fo = fo.Dataset.from_dir(
                dataset_dir='../../data/kws-all/train',
                dataset_type=fo.types.FiftyOneDataset,
            )
            test_fo = fo.Dataset.from_dir(
                dataset_dir='../../data/kws-all/test',
                dataset_type=fo.types.FiftyOneDataset,
            )

        self._total_train_dataset = SpeechCommandsDataset(train_fo)
        self._total_test_dataset = SpeechCommandsDataset(test_fo)

        self._total_classes = self._total_test_dataset.num_classes
        log.info(f'There are {self._total_classes} classes in current dataset')



if __name__ == '__main__':
    speech = GoogleSpeechCommand(num_devices=10, label_per_device=10, data_per_device=600)
    speech._download_dataset()
    print(len(speech._total_train_dataset))
    print(len(speech._total_test_dataset))

    speech._split_subdataset_train()
    speech._split_subdataset_test()
    #speech._convert_to_c_code_after_split('./dataset/speech/', 'speech')

    speech._allocate_labels_to_devices(method='loop', loop_step=5)

    speech._allocate_train_data_to_devices(method='dirichlet', alpha=1)
    speech._allocate_test_valid_data(test_data_size=600, valid_data_size=0)
    speech._convert_to_c_code_after_allocate('./dataset/speech/', 'speech')

    speech._generate_dataloader()
    speech._training_dataset_visualization(save_path='./dataset/speech/', fig_name='dirichlet')


