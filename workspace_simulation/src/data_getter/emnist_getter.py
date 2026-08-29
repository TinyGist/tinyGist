import torchvision
import logging
from .base_getter import BaseDatasetGetter

log = logging.getLogger(__name__)

class EMNIST(BaseDatasetGetter):
    def __init__(self, num_devices=20, split_method='balanced',
                 data_per_device=1000, label_per_device=27, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)
        self.__split_method = split_method

    def _download_dataset(self):
        self._total_train_dataset = torchvision.datasets.EMNIST(
            './data', train=True, download=True, split=self.__split_method,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1751,), (0.3332,))
            ])
        )
        self._total_test_dataset = torchvision.datasets.EMNIST(
            './data', train=False, download=True, split=self.__split_method,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1751,), (0.3332,))
            ])
        )
        self._total_classes = len(self._total_test_dataset.classes)
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    emnist = EMNIST(num_devices=10, label_per_device=5, data_per_device=300)
    emnist.download_dataset()
    emnist._split_subdataset_train()
    emnist._split_subdataset_test()
    #emnist.convert_to_c_code_after_split('./dataset/emnist/', 'EMNIST')

    emnist._allocate_labels_to_devices(method='ordered', loop_step=3)

    emnist._allocate_train_data_to_devices(method='random')
    emnist._allocate_test_valid_data(test_data_size=470, valid_data_size=0)
    emnist._convert_to_c_code_after_allocate('./dataset/emnist/', 'EMNIST')

    emnist._generate_dataloader()
    emnist._training_dataset_visualization(save_path='./dataset/emnist/', fig_name='random')

    pass
