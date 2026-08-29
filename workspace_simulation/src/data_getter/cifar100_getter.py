import torchvision
from src.data_getter.base_getter import BaseDatasetGetter
import logging
log = logging.getLogger(__name__)

class Cifar100(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500,
                 label_per_device=3, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        normalization = torchvision.transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
        self._total_train_dataset = torchvision.datasets.CIFAR100(
            './data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((160, 160)),
                torchvision.transforms.RandomCrop(128, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalization,
            ])
        )
        # Use the same CIFAR100 training examples and indices for validation, but
        # without random crop/flip so repeated validation measures the same inputs.
        self._total_valid_dataset = torchvision.datasets.CIFAR100(
            './data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
                normalization,
            ])
        )
        self._total_test_dataset = torchvision.datasets.CIFAR100(
            './data', train=False, download=True,
                transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
                normalization,
            ])
        )
        self._total_classes = len(self._total_test_dataset.classes)
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    cifar100 = Cifar100(num_devices=10, label_per_device=3, data_per_device=900)
    cifar100._download_dataset()
    cifar100._split_subdataset_train()
    cifar100._split_subdataset_test()
    cifar100._convert_to_c_code_after_split('./dataset/cifar100/', 'cifar100')

    cifar100._allocate_labels_to_devices(method='random', loop_step=3)

    cifar100._allocate_train_data_to_devices(method='random')
    cifar100._allocate_test_valid_data(test_data_size=500, valid_data_size=0)
    cifar100._convert_to_c_code_after_allocate('./dataset/cifar100/', 'cifar100')

    cifar100._generate_dataloader()
    cifar100._training_dataset_visualization(save_path='./dataset/cifar100/', fig_name='random')
    pass
