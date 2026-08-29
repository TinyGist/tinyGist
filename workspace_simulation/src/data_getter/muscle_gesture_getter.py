from src.data_getter.base_getter import BaseDatasetGetter
import torchvision
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
log = logging.getLogger(__name__)


def zscore_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + 1e-6)

class EMG4Dataset(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        file_pattern="*.csv",
        test_split=0.2,
        seed=42,
        reshape_to_8x8=False,
        dtype=torch.float32,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.file_pattern = file_pattern
        self.test_split = test_split
        self.seed = seed
        self.reshape_to_8x8 = reshape_to_8x8
        self.dtype = dtype

        # ---- Load all CSVs ----
        csv_paths = sorted(glob.glob(os.path.join(root, file_pattern)))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {root} with pattern {file_pattern}")

        dfs = [pd.read_csv(p, header=None) for p in csv_paths]
        data = pd.concat(dfs, ignore_index=True)

        # last column = label
        features = data.iloc[:, :-1].values  # (N, num_features)
        labels = data.iloc[:, -1].values     # (N,)

        if self.reshape_to_8x8 and features.shape[1] != 64:
            raise ValueError(
                f"reshape_to_8x8=True but got {features.shape[1]} features instead of 64."
            )

        self.features = features
        self.labels = labels  # labels for the *full* dataset (before split)

        # ---- Create deterministic train/test split ----
        N = len(self.features)
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(N)

        test_size = int(self.test_split * N)
        train_size = N - test_size

        if train:
            self.indices = perm[:train_size]
        else:
            self.indices = perm[train_size:]

        # ---- CIFAR-like attributes ----
        # labels for this split only
        self.targets = [int(self.labels[i]) for i in self.indices]

        # optional: class info like CIFAR100
        unique_classes = sorted(set(self.targets))
        self.classes = [str(c) for c in unique_classes]  # or human-readable names if you have them
        self.class_to_idx = {c: c for c in unique_classes}  # identity mapping (int->int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        x = self.features[real_idx]
        y = int(self.labels[real_idx])

        x = torch.tensor(x, dtype=self.dtype)
        if self.reshape_to_8x8:
            x = x.view(8, 8)  # or x.view(1, 8, 8) if you want a channel dim
            x = x.unsqueeze(dim=0)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)
        else:
            y = torch.tensor(y, dtype=torch.long)

        return x, y


class MuscleGesture(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500,
                 label_per_device=3, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):

        emg_train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(zscore_tensor),
        ])

        emg_test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(zscore_tensor),
        ])

        self._total_train_dataset = EMG4Dataset(
            root="./data/muscle-gesture",  # folder with your CSVs
            train=True,
            transform=emg_train_transform,
            reshape_to_8x8=True,  # or True if you want (8,8) tensors
        )

        self._total_test_dataset = EMG4Dataset(
            root="./data/muscle-gesture",
            train=False,
            transform=emg_test_transform,
            reshape_to_8x8=True,
        )
        self._total_classes = 4
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    muscle = MuscleGesture(num_devices=200, label_per_device=2, data_per_device=600)
    muscle.get_c_libraries_each_device_from_new(
        'loop', 1, 'dirichlet',
        100, 100, 100,
        './dataset/dataset_gesture/', 'Gesture'
    )
    '''muscle_gesture = MuscleGesture(num_devices=10, label_per_device=2, data_per_device=600)
    muscle_gesture._download_dataset()
    muscle_gesture._split_subdataset_train()
    muscle_gesture._split_subdataset_test()
    muscle_gesture._convert_to_c_code_after_split('./dataset/muscle_gesture/', 'muscle_gesture')

    muscle_gesture._allocate_labels_to_devices(method='ordered', loop_step=1)

    muscle_gesture._allocate_train_data_to_devices(method='dirichlet', alpha=1)
    muscle_gesture._allocate_test_valid_data(test_data_size=600, valid_data_size=0)
    #muscle_gesture._convert_to_c_code_after_allocate('./dataset/muscle_gesture/', 'muscle_gesture')

    muscle_gesture._generate_dataloader()
    muscle_gesture._training_dataset_visualization(save_path='./dataset/muscle_gesture/', fig_name='dirichlet')'''
    pass
