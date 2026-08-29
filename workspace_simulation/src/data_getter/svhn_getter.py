import torchvision
from src.data_getter.base_getter import BaseDatasetGetter
import torch
import logging
log = logging.getLogger(__name__)

class SVHNWithTargets(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, download=True):
        self.dataset = torchvision.datasets.SVHN(
            root=root,
            split=split,
            transform=transform,
            download=download
        )

        # convert labels
        self.targets = (self.dataset.labels % 10).tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = self.targets[idx]
        return img, label

class SVHN(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500,
                 label_per_device=3, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        self._total_train_dataset = SVHNWithTargets(
            './data', split="train", download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.4377, 0.4438, 0.4728),
                    std=(0.1980, 0.2010, 0.1970)
                )
            ])
        )
        self._total_test_dataset = SVHNWithTargets(
            './data', split="test", download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.4377, 0.4438, 0.4728),
                    std=(0.1980, 0.2010, 0.1970)
                )
            ])
        )
        self._total_classes = len(set(self._total_test_dataset.targets))
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    svhn = SVHN(num_devices=10, label_per_device=3, data_per_device=900)
    svhn._download_dataset()
    svhn._split_subdataset_train()
    svhn._split_subdataset_test()
    svhn._convert_to_c_code_after_split('./dataset/svhn/', 'SVHN')

    svhn._allocate_labels_to_devices(method='random', loop_step=3)

    svhn._allocate_train_data_to_devices(method='random')
    svhn._allocate_test_valid_data(test_data_size=500, valid_data_size=0)
    svhn._convert_to_c_code_after_allocate('./dataset/svhn/', 'SVHN')

    svhn._generate_dataloader()
    svhn._training_dataset_visualization(save_path='./dataset/svhn/', fig_name='random')
    pass
