import torchvision
from src.data_getter.base_getter import BaseDatasetGetter
import logging
log = logging.getLogger(__name__)

class Cifar10(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500,
                 label_per_device=3, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        self._total_train_dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4915, 0.4823, 0.4468),
                    (0.2470, 0.2435, 0.2616)
                )
            ])
        )
        self._total_test_dataset = torchvision.datasets.CIFAR10(
            './data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4915, 0.4823, 0.4468),
                    (0.2470, 0.2435, 0.2616)
                )
            ])
        )
        self._total_classes = len(self._total_test_dataset.classes)
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    cifar10 = Cifar10(num_devices=10, label_per_device=3, data_per_device=900)
    cifar10._download_dataset()
    cifar10._split_subdataset_train()
    cifar10._split_subdataset_test()
    cifar10._convert_to_c_code_after_split('./dataset/cifar10/', 'CIFAR10')

    cifar10._allocate_labels_to_devices(method='random', loop_step=3)

    cifar10._allocate_train_data_to_devices(method='random')
    cifar10._allocate_test_valid_data(test_data_size=500, valid_data_size=0)
    cifar10._convert_to_c_code_after_allocate('./dataset/cifar10/', 'CIFAR10')

    cifar10._generate_dataloader()
    cifar10._training_dataset_visualization(save_path='./dataset/cifar10/', fig_name='random')
    pass
