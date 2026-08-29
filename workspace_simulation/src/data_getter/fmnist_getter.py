import torchvision
import logging
from src.data_getter.base_getter import BaseDatasetGetter

log = logging.getLogger(__name__)

class FashionMNIST(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500,
                 label_per_device=3, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        self._total_train_dataset = torchvision.datasets.FashionMNIST(
            './data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.2860,), (0.3530,))
            ])
        )
        self._total_test_dataset = torchvision.datasets.FashionMNIST(
            './data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.2860,), (0.3530,))
            ])
        )
        self._total_classes = len(self._total_test_dataset.classes)
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    fmnist = FashionMNIST(num_devices=10, label_per_device=3, data_per_device=600)
    fmnist.get_c_libraries_each_device_from_new(
        'loop', 1, 'dirichlet',
        100, 100, 100,
        './dataset/', 'FashionMNIST'
    )

    pass
