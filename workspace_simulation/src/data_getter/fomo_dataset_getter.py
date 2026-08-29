from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import logging
import os
from functools import partial

from src.data_getter.base_getter import BaseDatasetGetter

log = logging.getLogger(__name__)


class FomoDataset(Dataset):
    def __init__(self, fo_dataset, transform=None, task="vehicle-binary"):
        self.fo_dataset = fo_dataset
        self.transform = transform
        self.num_classes = 1
        self.label_strs = self.fo_dataset.values("labels_strs")
        self.img_paths = self.fo_dataset.values("filepath")
        self.label_paths = self.fo_dataset.values("fomo_label_path")
        self.folder_path = os.path.split(os.path.split(self.img_paths[0])[0])[0]

        self.task = task
        if task not in ['human', 'vehicle', 'vehicle-binary']:
            raise NotImplementedError("Only supporting tasks 'human', 'vehicle' and 'vehicle-binary'")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        label_path = os.path.join(self.folder_path, label_path)
        # print(img_path, label_path)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
        label = np.load(label_path)

        if self.transform:
            img = self.transform(img)

        if self.task == 'vehicle-binary':
            label[label>0] = 1

        label = torch.from_numpy(label).long()

        return img, label


class FomoDetection(BaseDatasetGetter):
    def __init__(self, num_devices=20, data_per_device=1000, label_per_device=1, prefix_name='device_', task="vehicle-binary"):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)
        self.__task = task

    def _download_dataset(self):
        import fiftyone as fo

        if self.__task == 'vehicle-binary':
            path = "vehicle"
        else:
            path = self.__task
        try:
            train_fo = fo.Dataset.from_dir(
                dataset_dir=f"./data/coco-2017-fomo-{path}/train",
                dataset_type=fo.types.FiftyOneDataset,
            )
            val_fo = fo.Dataset.from_dir(
                dataset_dir=f"./data/coco-2017-fomo-{path}/validation",
                dataset_type=fo.types.FiftyOneDataset,
            )
        except (FileNotFoundError, ValueError):
            train_fo = fo.Dataset.from_dir(
                dataset_dir=f"../../data/coco-2017-fomo-{path}/train",
                dataset_type=fo.types.FiftyOneDataset
            )
            val_fo = fo.Dataset.from_dir(
                dataset_dir=f"../../data/coco-2017-fomo-{path}/validation",
                dataset_type=fo.types.FiftyOneDataset
            )

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self._total_train_dataset = FomoDataset(
            train_fo, transform=transform, task=self.__task
        )
        self._total_test_dataset = FomoDataset(
            val_fo, transform=transform, task=self.__task
        )

        self._total_classes = self._total_train_dataset.num_classes
        log.info(f'There are {self._total_classes} classes in current dataset')

FomoDetectionPerson = partial(FomoDetection, task="human")
FomoDetectionVehicle = partial(FomoDetection, task="vehicle")
FomoDetectionVehicleBinary = partial(FomoDetection, task="vehicle-binary")


if __name__ == '__main__':
    detection = FomoDetectionPerson(num_devices=20, data_per_device=1000)
    detection._download_dataset()
    detection._split_subdataset_train()
    detection._split_subdataset_test()
    detection._allocate_labels_to_devices()
    detection._allocate_train_data_to_devices()
    detection._allocate_test_valid_data()
    detection._generate_dataloader()

    detection = FomoDetectionVehicle(num_devices=20, data_per_device=1000)
    detection._download_dataset()
    detection._split_subdataset_train()
    detection._split_subdataset_test()
    detection._allocate_labels_to_devices()
    detection._allocate_train_data_to_devices()
    detection._allocate_test_valid_data()
    detection._generate_dataloader()
