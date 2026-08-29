import torchvision
import logging
from src.data_getter.base_getter import BaseDatasetGetter

log = logging.getLogger(__name__)


class MNIST(BaseDatasetGetter):
    def __init__(self, num_devices=10, data_per_device=500,
                 label_per_device=3, prefix_name='device_'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        self._total_train_dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )
        self._total_test_dataset = torchvision.datasets.MNIST(
            './data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )
        self._total_classes = len(self._total_test_dataset.classes)
        log.info(f'There are {self._total_classes} classes in current dataset')

if __name__ == '__main__':
    mnist = MNIST(num_devices=200, label_per_device=3, data_per_device=600)
    mnist.get_c_libraries_each_device_from_new(
        'loop', 3, 'dirichlet',
        100, 100, 100,
        './Data/dataset_mnist/', 'MNIST'
    )

    '''mnist = MNIST(num_devices=10, label_per_device=2, data_per_device=600)
    mnist._download_dataset()
    mnist._split_subdataset_train()
    mnist._split_subdataset_test()
    mnist._convert_to_c_code_after_split('./dataset/mnist/', 'MNIST')

    mnist._allocate_labels_to_devices(method='random', loop_step=3)

    mnist._allocate_train_data_to_devices(method='random')
    mnist._allocate_test_valid_data(test_data_size=500, valid_data_size=0)
    mnist._convert_to_c_code_after_allocate('./dataset/mnist/', 'MNIST')

    mnist._generate_dataloader()
    mnist._training_dataset_visualization(save_path='./dataset/mnist/', fig_name='random')'''
