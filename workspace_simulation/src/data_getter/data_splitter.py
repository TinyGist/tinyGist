import os
from pathlib import Path
import numpy as np
import logging
import torch
from torch.utils.data import Subset, ConcatDataset
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib

from .allocation import apportion_integer_counts, build_training_dataset
from .dataloader_factory import build_dataloaders

matplotlib.use("Agg")

log = logging.getLogger(__name__)


class DataSplit:
    def __init__(self):
        # These four parameters should be filled with value in the inheriting classes
        self._num_device = None
        self._data_per_device = None
        self._label_per_device = None

        self._prefix_name = 'device_'
        self._dataset_train_store_directory = 'train/'
        self._dataset_test_store_directory = 'test/'
        self._dataset_valid_store_directory = 'valid/'
        self._dataset_for_c_store_directory = 'C_library/'

        # These three parameter should be filled after executing the _download_dataset method
        self._total_train_dataset = None
        # Optional view of the training split with deterministic validation transforms.
        # When unset, validation uses _total_train_dataset as before.
        self._total_valid_dataset = None
        self._total_test_dataset = None
        self._total_classes = None
        # following would be dictionary, they are variables used to store useful/required data
        # after getting, allocating data, the dataloader can be easily access through the variables.
        self._device_id_to_labels = None
        self._label_to_subset_train = None
        self._label_to_subset_valid = None
        self._label_to_subset_test = None
        self._device_id_to_allocated_training_dataset = None
        self._dataset_for_test_valid = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._dataloader_valid = None

    def _download_dataset(self):
        '''
        The is method must be realized, the following methods need,
        self._total_classes
        self._total_train_dataset
        self._total_test_dataset
        to work properly
        :return:
        None
        '''
        raise NotImplementedError('Please inherit and re-implement this method for the target dataset')

    def _validate_downloaded_dataset(self):
        if self._total_train_dataset is None or self._total_test_dataset is None:
            raise ValueError(
                'Dataset download must initialize both training and test datasets'
            )
        if not isinstance(self._total_classes, int) or self._total_classes <= 0:
            raise ValueError(
                f'Dataset class count must be a positive integer, got '
                f'{self._total_classes!r}'
            )
        if self._label_per_device > self._total_classes:
            raise ValueError(
                f'number of labels per device {self._label_per_device} exceeds '
                f'the total classes {self._total_classes}'
            )
        if self._num_device * self._label_per_device < self._total_classes:
            raise ValueError(
                f'device count times labels per device '
                f'({self._num_device * self._label_per_device}) is smaller than '
                f'the number of classes {self._total_classes}'
            )
        if (
                self._total_valid_dataset is not None
                and len(self._total_valid_dataset) != len(self._total_train_dataset)
        ):
            raise ValueError(
                'A separate validation transform view must preserve training '
                'dataset length and indices'
            )

    @staticmethod
    def _dataset_targets(dataset):
        """Return labels without loading samples when the dataset exposes targets."""
        if isinstance(dataset, Subset):
            parent_targets = DataSplit._dataset_targets(dataset.dataset)
            indices = torch.as_tensor(dataset.indices, dtype=torch.long)
            if indices.device.type != 'cpu':
                indices = indices.cpu()
            return parent_targets.index_select(0, indices)
        if isinstance(dataset, ConcatDataset):
            target_parts = [DataSplit._dataset_targets(part) for part in dataset.datasets]
            if not target_parts:
                return torch.empty(0, dtype=torch.long)
            return torch.cat(target_parts)

        targets = getattr(dataset, 'targets', None)
        if targets is None:
            # Non-standard datasets may not publish targets. This fallback preserves
            # compatibility, while torchvision classification datasets use the fast path.
            targets = [label for _, label in dataset]
        target_tensor = torch.as_tensor(targets)
        if target_tensor.ndim != 1:
            raise ValueError(
                f'Classification targets must be one-dimensional, got shape '
                f'{tuple(target_tensor.shape)}'
            )
        return target_tensor.to(dtype=torch.long)

    def _allocate_labels_to_devices(self, method='ordered', loop_step=2):
        if self._total_classes < self._label_per_device:
            raise ValueError(
                f'number of labels per device is {self._label_per_device}, however, '
                f'number of total classes is {self._total_classes}. Thus, too many '
                f'labels per device.'
            )
        self._device_id_to_labels = dict()
        # L = l + s(n-1)
        if method == 'ordered':
            base_labels = list(range(self._label_per_device))
            step = (
                0.0
                if self._num_device == 1
                else float(self._total_classes - self._label_per_device)
                / float(self._num_device - 1)
            )
            for device_id in range(self._num_device):
                labels = (
                    (np.asarray(base_labels) + np.ceil(step * device_id))
                    % self._total_classes
                )
                self._device_id_to_labels[self._prefix_name + str(device_id)] = (
                    labels.astype(int).tolist()
                )
            log.info('Labels have been allocated using [ordered] method')
        elif method == 'random':
            for device_id in range(self._num_device):
                labels = np.random.permutation(self._total_classes)[:self._label_per_device]
                self._device_id_to_labels[self._prefix_name+str(device_id)] = labels.tolist()
            log.info('Labels have been allocated using [random] method')
        elif method == 'loop':
            base_labels = list(range(self._label_per_device))
            step = loop_step
            for device_id in range(self._num_device):
                labels = (
                    (np.asarray(base_labels) + np.ceil(step * device_id))
                    % self._total_classes
                )
                self._device_id_to_labels[self._prefix_name + str(device_id)] = (
                    labels.astype(int).tolist()
                )
            log.info('Labels have been allocated using [loop] method')
        else:
            raise NotImplementedError('Only implemented [ordered, random, loop]')

        log.info(f"Labels on each device are: {self._device_id_to_labels}")

    def _split_subdataset_train(self):
        valid_dataset = (
            self._total_valid_dataset
            if self._total_valid_dataset is not None
            else self._total_train_dataset
        )
        if self._total_classes > 1:
            targets = self._dataset_targets(self._total_train_dataset)
            unique_label = torch.unique(targets).tolist()
            self._label_to_subset_train = dict()
            self._label_to_subset_valid = dict()
            for label in unique_label:
                indices = torch.where(targets == label)[0].tolist()
                split_index = int(0.8 * len(indices))
                self._label_to_subset_train[int(label)] = Subset(
                    self._total_train_dataset, indices[:split_index]
                )
                self._label_to_subset_valid[int(label)] = Subset(
                    valid_dataset, indices[split_index:]
                )
            log.info('Subdatasets for training and validation have been split [classification]')
        elif self._total_classes == 1:
            self._label_to_subset_train = dict()
            self._label_to_subset_valid = dict()

            indices = torch.randperm(len(self._total_train_dataset)).tolist()
            split_index = int(0.8 * len(indices))
            self._label_to_subset_train[0] = Subset(self._total_train_dataset, indices[:split_index])
            self._label_to_subset_valid[0] = Subset(valid_dataset, indices[split_index:])
            log.info('Subdatasets for training and validation have been split [object detection]')
        else:
            raise ValueError(f"Infeasible number [{self._total_classes}] of classes")


    def _split_subdataset_test(self):
        if self._total_classes > 1:
            self._label_to_subset_test = dict()
            targets = self._dataset_targets(self._total_test_dataset)
            unique_label = torch.unique(targets).tolist()
            for label in unique_label:
                indices_bool = torch.where(targets == label)[0].tolist()
                self._label_to_subset_test[int(label)] = Subset(self._total_test_dataset, indices_bool)
            log.info('Subdataset for testing have been allocated [classification]')
        elif self._total_classes == 1:
            self._label_to_subset_test = dict()
            self._label_to_subset_test[int(0)] = self._total_test_dataset
            log.info('Subdataset for testing have been allocated [object detection]')
        else:
            raise ValueError(f"Infeasible number [{self._total_classes}] of classes")

    @staticmethod
    def _apportion_integer_counts(total, weights):
        return apportion_integer_counts(total, weights)

    def _build_training_dataset(self, labels, sample_counts):
        return build_training_dataset(
            self._label_to_subset_train,
            labels,
            sample_counts,
            logger=log,
        )

    def _allocate_train_data_to_devices(self, method='dirichlet', alpha=100):
        if not isinstance(self._label_to_subset_train, dict):
            raise RuntimeError(
                'Please split the training dataset before allocating clients'
            )

        self._device_id_to_allocated_training_dataset = dict()

        if method in {'dirichlet', 'uniform'}:
            for device_id in range(self._num_device):
                labels = self._device_id_to_labels[self._prefix_name + str(device_id)]
                weights = (
                    np.random.dirichlet([alpha] * len(labels))
                    if method == 'dirichlet'
                    else np.ones(len(labels), dtype=np.float64)
                )
                sample_counts = self._apportion_integer_counts(
                    self._data_per_device,
                    weights,
                )
                self._device_id_to_allocated_training_dataset[
                    self._prefix_name + str(device_id)
                ] = self._build_training_dataset(labels, sample_counts)
        elif method == 'random':
            for device_id in range(self._num_device):
                combined_dataset = []
                for label_id in self._device_id_to_labels[self._prefix_name+str(device_id)]:
                    combined_dataset.append(self._label_to_subset_train[label_id])
                combined_dataset = ConcatDataset(combined_dataset)
                combined_dataset_length = len(combined_dataset)
                indices = np.random.randint(0, combined_dataset_length, self._data_per_device)
                combined_dataset = Subset(combined_dataset, indices)

                perm = torch.randperm(len(combined_dataset))
                self._device_id_to_allocated_training_dataset[self._prefix_name+str(device_id)] = Subset(combined_dataset, perm)
        else:
            raise NotImplementedError('Only implemented [dirichlet, uniform, random]')

    def _allocate_test_valid_data(self, test_data_size=1000, valid_data_size=1000):
        if not isinstance(self._label_to_subset_test, dict):
            raise RuntimeError('Please split the test dataset before allocation')
        if not isinstance(self._label_to_subset_valid, dict):
            raise RuntimeError('Please split the validation dataset before allocation')

        self._dataset_for_test_valid = dict()

        equal_class_weights = np.ones(self._total_classes, dtype=np.float64)
        test_counts = self._apportion_integer_counts(test_data_size, equal_class_weights)
        valid_counts = self._apportion_integer_counts(valid_data_size, equal_class_weights)

        dataset_for_test = []
        dataset_for_valid = []

        for label, subset in self._label_to_subset_test.items():
            length_subset = len(subset)
            requested_length = test_counts[int(label)]
            used_length = requested_length
            if length_subset < requested_length:
                used_length = length_subset
                log.warning(f'test dataset for label {label} is too small')
            log.info(f'dataset allocated for test is with a size of {used_length}')

            used_subset = Subset(subset, torch.randperm(length_subset).tolist())
            dataset_for_test.append(Subset(used_subset, range(used_length)))

        for label, subset in self._label_to_subset_valid.items():
            length_subset = len(subset)
            requested_length = valid_counts[int(label)]
            used_length = requested_length
            if length_subset < requested_length:
                used_length = length_subset
                log.warning(f'valid dataset for label {label} is too small')
            log.info(f'dataset allocated for validation is with a size of {used_length}')

            used_subset = Subset(subset, torch.randperm(length_subset).tolist())
            dataset_for_valid.append(Subset(used_subset, range(used_length)))

        self._dataset_for_test_valid['test'] = ConcatDataset(dataset_for_test)
        self._dataset_for_test_valid['valid'] = ConcatDataset(dataset_for_valid)
        log.info('validation and test dataset have been allocated')

    def _generate_dataloader(
            self, train_batch_size=64, test_batch_size=64, valid_batch_size=64,
            num_workers=0
    ):
        (
            self._dataloader_train,
            self._dataloader_test,
            self._dataloader_valid,
        ) = build_dataloaders(
            self._device_id_to_allocated_training_dataset,
            self._dataset_for_test_valid,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            valid_batch_size=valid_batch_size,
            num_workers=num_workers,
        )

    def _training_dataset_visualization(self, save_path, fig_width=15, fig_height=9, fig_name='data_dis'):
        if not isinstance(self._device_id_to_allocated_training_dataset, dict):
            raise ValueError('Allocated training datasets are not feasible')
        os.makedirs(save_path, exist_ok=True)
        if self._total_classes == 1:# means object detection task
            return
        device_num = len(self._device_id_to_allocated_training_dataset)
        x_labels = [f'd{i}' for i in range(device_num)]
        all_device_labels = dict()
        all_labels = []
        for device_id, dataset in self._device_id_to_allocated_training_dataset.items():
            device_labels = self._dataset_targets(dataset)
            all_labels.extend(device_labels.tolist())
            labels, label_counts = torch.unique(device_labels, return_counts=True)
            counts = dict(zip(labels.tolist(), label_counts.tolist()))
            all_device_labels[device_id] = counts
        classes_num = len(np.unique(all_labels))

        bottoms = np.zeros(device_num)
        data_split_matrix = np.zeros((device_num, classes_num))
        idx = 0
        for label_to_number in all_device_labels.values():
            for label, number in label_to_number.items():
                if label >= data_split_matrix.shape[1]:
                    data_split_matrix = np.concatenate([data_split_matrix, np.zeros((device_num, label-data_split_matrix.shape[1]+1))], axis=1)
                data_split_matrix[idx, label] = number
            idx += 1

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        for i in range(data_split_matrix.shape[1]):
            bars = ax.bar(x_labels, data_split_matrix[:, i], label=f'c{i}', bottom=bottoms)
            bottoms = bottoms[:] + data_split_matrix[:, i]
            ax.bar_label(
                bars,
                labels=[f'c{i}\n{int(v)}' if v > 0 else '' for v in data_split_matrix[:, i]],
                label_type='center',
                fontsize=8
            )

        ax.set_xlabel('Device', fontsize=12, labelpad=6)
        ax.set_ylabel('Number of Labels', fontsize=12, labelpad=6)
        ax.set_title('Label Distribution Across Devices', fontsize=14, pad=10)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(
            title="Labels",
            bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0,
            fontsize=9, title_fontsize=10, ncol=2
        )
        fig.savefig(os.path.join(save_path, fig_name + '.svg'))
        plt.close()

    def _store_split_dataset(self, save_path: str):
        if not isinstance(self._device_id_to_allocated_training_dataset, dict):
            raise ValueError('train dataset is not feasible')
        if not isinstance(self._dataset_for_test_valid, dict):
            raise ValueError('test/validation datasets are not feasible')
        base_path = Path(save_path)
        train_save_path = base_path / self._dataset_train_store_directory
        train_save_path.mkdir(parents=True, exist_ok=True)
        for device_id, sub_dataset in self._device_id_to_allocated_training_dataset.items():
            torch.save(sub_dataset, train_save_path / f'train_{device_id}.pt')

        test_save_path = base_path / self._dataset_test_store_directory
        valid_save_path = base_path / self._dataset_valid_store_directory
        test_save_path.mkdir(parents=True, exist_ok=True)
        valid_save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._dataset_for_test_valid['test'], test_save_path / 'test.pt')
        torch.save(self._dataset_for_test_valid['valid'], valid_save_path / 'valid.pt')

    def _load_split_dataset(self, load_path):
        if (
                self._device_id_to_allocated_training_dataset is not None
                or self._dataset_for_test_valid is not None
        ):
            raise RuntimeError('Dataset has already been prepared')

        base_path = Path(load_path)
        train_load_path = base_path / self._dataset_train_store_directory
        try:
            training_datasets = {
                f'{self._prefix_name}{device_id}': torch.load(
                    train_load_path / f'train_{self._prefix_name}{device_id}.pt',
                    weights_only=False,
                )
                for device_id in range(self._num_device)
            }
            evaluation_datasets = {
                'test': torch.load(
                    base_path / self._dataset_test_store_directory / 'test.pt',
                    weights_only=False,
                ),
                'valid': torch.load(
                    base_path / self._dataset_valid_store_directory / 'valid.pt',
                    weights_only=False,
                ),
            }
        except Exception as exc:
            raise RuntimeError(
                f'Could not load the complete split dataset from {base_path}'
            ) from exc

        self._device_id_to_allocated_training_dataset = training_datasets
        self._dataset_for_test_valid = evaluation_datasets

    def _convert_to_c_code_after_allocate(self, save_path: str, dataset_name: str):
        assert isinstance(self._device_id_to_allocated_training_dataset, dict), 'Does not have proper training dataset, redownload or load from local'
        assert isinstance(self._dataset_for_test_valid, dict), 'Does not have test/valid dataset, redownload or load from local'
        assert self._total_classes is not None, 'Do not know how many classes there are'

        library_store_path = save_path + self._dataset_for_c_store_directory if save_path[-1] == '/' \
            else save_path + '/' + self._dataset_for_c_store_directory
        os.makedirs(library_store_path, exist_ok=True)
        for device_id, dataset in self._device_id_to_allocated_training_dataset.items():
            data_store_path = library_store_path + '/' + device_id
            os.makedirs(data_store_path, exist_ok=True)
            data_list = []
            label_list = []
            for data, label in dataset:
                if data.shape[0] == 3:
                    data = data.permute(1, 2, 0).contiguous()
                data_list.append(data)
                label_list.append(label)
            with open(f'{data_store_path}/{dataset_name}_train_data.h', 'w') as f:
                f.write(f"#ifndef {dataset_name}_TRAINING_DATA\n#define {dataset_name}_TRAINING_DATA\n")
                f.write('const float ' + dataset_name + '_training_data[] = {\n')
                for data in data_list:
                    temp_data = data.reshape(-1)
                    for point in temp_data:
                        f.write(f'{float(point):.5}f, ')
                    f.write('\n')
                f.write('\n};\n')
                f.write('#endif')
                log.info(f'C library of training dataset for {device_id} of {dataset_name} has generated')
            with open(f'{data_store_path}/{dataset_name}_train_data_label.h', 'w') as f:
                f.write(f"#ifndef {dataset_name}_TRAINING_DATA_LABEL\n#define {dataset_name}_TRAINING_DATA_LABEL\n")
                f.write('const float ' + dataset_name + '_training_data_label[] = {\n')
                for label in label_list:
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    label_one_hot = F.one_hot(torch.tensor(label).round().to(torch.long), self._total_classes).tolist()
                    for point in label_one_hot:
                        f.write(f'{float(point)}f, ')
                    f.write('\n')
                f.write('\n};\n')
                f.write('#endif')
                log.info(f'C library of training label for {device_id} of {dataset_name} has generated')

        test_dataset = self._dataset_for_test_valid['test']
        valid_dataset = self._dataset_for_test_valid['valid']
        data_list = []
        label_list = []
        for data, label in test_dataset:
            if data.shape[0] == 3:
                data = data.permute(1, 2, 0).contiguous()
            data_list.append(data)
            label_list.append(label)
        with open(f'{library_store_path}/{dataset_name}_test_data.h', 'w') as f:
            f.write(f"#ifndef {dataset_name}_TEST_DATA\n#define {dataset_name}_TEST_DATA\n")
            f.write('const float ' + dataset_name + '_test_data[] = {\n')
            for data in data_list:
                temp_data = data.reshape(-1)
                for point in temp_data:
                    f.write(f'{float(point):.5}f, ')
                f.write('\n')
            f.write('\n};\n')
            f.write('#endif')
            log.info(f'C library of test dataset of {dataset_name} has generated')
        with open(f'{library_store_path}/{dataset_name}_test_data_label.h', 'w') as f:
            f.write(f"#ifndef {dataset_name}_TEST_DATA_LABEL\n#define {dataset_name}_TEST_DATA_LABEL\n")
            f.write('const float ' + dataset_name + '_test_data_label[] = {\n')
            for label in label_list:
                if isinstance(label, torch.Tensor):
                    label = label.item()
                label_one_hot = F.one_hot(torch.tensor(label).round().to(torch.long), self._total_classes).tolist()
                for point in label_one_hot:
                    f.write(f'{float(point)}f, ')
                f.write('\n')
            f.write('\n};\n')
            f.write('#endif')
            log.info(f'C library of test label of {dataset_name} has generated')

        data_list = []
        label_list = []
        for data, label in valid_dataset:
            if data.shape[0] == 3:
                data = data.permute(1, 2, 0).contiguous()
            data_list.append(data)
            label_list.append(label)
        with open(f'{library_store_path}/{dataset_name}_val_data.h', 'w') as f:
            f.write(f"#ifndef {dataset_name}_VAL_DATA\n#define {dataset_name}_VAL_DATA\n")
            f.write('const float ' + dataset_name + '_val_data[] = {\n')
            for data in data_list:
                temp_data = data.reshape(-1)
                for point in temp_data:
                    f.write(f'{float(point):.5}f, ')
                f.write('\n')
            f.write('\n};\n')
            f.write('#endif')
            log.info(f'C library of validation dataset of {dataset_name} has generated')
        with open(f'{library_store_path}/{dataset_name}_val_data_label.h', 'w') as f:
            f.write(f"#ifndef {dataset_name}_VAL_DATA_LABEL\n#define {dataset_name}_VAL_DATA_LABEL\n")
            f.write('const float ' + dataset_name + '_val_data_label[] = {\n')
            for label in label_list:
                if isinstance(label, torch.Tensor):
                    label = label.item()
                label_one_hot = F.one_hot(torch.tensor(label).round().to(torch.long), self._total_classes).tolist()
                for point in label_one_hot:
                    f.write(f'{float(point)}f, ')
                f.write('\n')
            f.write('\n};\n')
            f.write('#endif')
            log.info(f'C library of validation label of {dataset_name} has generated')

    def _convert_to_c_code_after_split(self, save_path: str, dataset_name: str):
        assert isinstance(self._label_to_subset_train, dict), 'Not have properly split training dataset, split it manually'
        assert isinstance(self._label_to_subset_test, dict), 'Not have properly split test dataset, split it manually'
        assert self._total_classes is not None, 'Do not know how many classes there are'

        library_store_path = save_path + self._dataset_for_c_store_directory if save_path[-1] == '/' \
            else save_path + '/' + self._dataset_for_c_store_directory
        os.makedirs(library_store_path, exist_ok=True)

        for label, _ in self._label_to_subset_train.items():
            train_dataset = self._label_to_subset_train[label]
            test_dataset = self._label_to_subset_test[label]
            data_train_list, label_train_list = [], []
            data_test_list, label_test_list = [], []
            for data, lbl in train_dataset:
                if data.shape[0] == 3:
                    data = data.permute(1, 2, 0).contiguous()
                data_train_list.append(data)
                label_train_list.append(lbl)
            for data, lbl in test_dataset:
                if data.shape[0] == 3:
                    data = data.permute(1, 2, 0).contiguous()
                data_test_list.append(data)
                label_test_list.append(lbl)
            dir_path = f'{library_store_path}/label_{label}/'
            os.makedirs(dir_path, exist_ok=True)
            with open(f'{dir_path}{dataset_name}_training_data_{label}.h', 'w') as f:
                f.write(f"#ifndef {dataset_name}_TRAINING_DATA_{label}\n#define {dataset_name}_TRAINING_DATA_{label}\n")
                f.write('const float ' + dataset_name + '_training_data_' + str(label) + '[] = {\n')
                for data in data_train_list:
                    temp_data = data.reshape(-1)
                    for point in temp_data:
                        f.write(f'{float(point):.5}f, ')
                    f.write('\n')
                f.write('\n};\n')
                f.write('#endif')
                log.info(f'C library of training dataset for label {label} of {dataset_name} has generated')
            with open(f'{dir_path}{dataset_name}_test_data_{label}.h', 'w') as f:
                f.write(f"#ifndef {dataset_name}_TEST_DATA_{label}\n#define {dataset_name}_TEST_DATA_{label}\n")
                f.write('const float ' + dataset_name + '_test_data_' + str(label) + '[] = {\n')
                for data in data_test_list:
                    temp_data = data.reshape(-1)
                    for point in temp_data:
                        f.write(f'{float(point):.5}f, ')
                    f.write('\n')
                f.write('\n};\n')
                f.write('#endif')
                log.info(f'C library of test dataset for label {label} of {dataset_name} has generated')
            with open(f'{dir_path}{dataset_name}_training_label_{label}.h', 'w') as f:
                f.write(f"#ifndef {dataset_name}_TRAINING_LABEL_{label}\n#define {dataset_name}_TRAINING_LABEL_{label}\n")
                f.write('const float ' + dataset_name + '_training_label_' + str(label) + '[] = {\n')
                for lbl in label_train_list:
                    if isinstance(lbl, torch.Tensor):
                        lbl = lbl.item()
                    label_one_hot = F.one_hot(torch.tensor(lbl).to(torch.long), self._total_classes).tolist()
                    for point in label_one_hot:
                        f.write(f'{float(point)}f, ')
                    f.write('\n')
                f.write('\n};\n')
                f.write('#endif')
                log.info(f'C library of training label for label {label} of {dataset_name} has generated')
            with open(f'{dir_path}{dataset_name}_test_label_{label}.h', 'w') as f:
                f.write(f"#ifndef {dataset_name}_TEST_LABEL_{label}\n#define {dataset_name}_TEST_LABEL_{label}\n")
                f.write('const float ' + dataset_name + '_test_label_' + str(label) + '[] = {\n')
                for lbl in label_test_list:
                    if isinstance(lbl, torch.Tensor):
                        lbl = lbl.item()
                    label_one_hot = F.one_hot(torch.tensor(lbl).to(torch.long), self._total_classes).tolist()
                    for point in label_one_hot:
                        f.write(f'{float(point)}f, ')
                    f.write('\n')
                f.write('\n};\n')
                f.write('#endif')
                log.info(f'C library of test label for label {label} of {dataset_name} has generated')

    def get_data_from_new(
            self,
            label_allo_method:str, label_allo_loop_step:int,
            data_allo_method:str, data_allo_dirichlet_alpha:int,
            test_data_size:int, valid_data_size:int,
            data_store_folder:str,
            train_batch_size:int, test_batch_size:int, valid_batch_size:int,
            num_workers:int = 0
    ):
        self._download_dataset()
        self._validate_downloaded_dataset()
        self._split_subdataset_train()
        self._split_subdataset_test()
        self._allocate_labels_to_devices(label_allo_method, label_allo_loop_step)
        self._allocate_train_data_to_devices(data_allo_method, data_allo_dirichlet_alpha)
        self._allocate_test_valid_data(test_data_size, valid_data_size)
        self._generate_dataloader(
            train_batch_size, test_batch_size, valid_batch_size, num_workers
        )
        self._training_dataset_visualization(save_path=data_store_folder)
        #self._store_split_dataset(save_path=data_store_folder)
        #self._convert_to_c_code_after_allocate(data_store_folder, 'MNIST')
        
    def get_data_from_store(
            self,
            data_store_folder:str,
            train_batch_size: int, test_batch_size: int, valid_batch_size: int,
            num_workers: int = 0
    ):
        self._load_split_dataset(data_store_folder)
        self._generate_dataloader(
            train_batch_size, test_batch_size, valid_batch_size, num_workers
        )
        
    def get_c_libraries_each_label(
            self,
            data_store_folder:str, dataset_name:str
    ):
        self._download_dataset()
        self._split_subdataset_test()
        self._split_subdataset_train()
        self._convert_to_c_code_after_split(data_store_folder, dataset_name)
        
    def get_c_libraries_each_device_from_store(
            self,
            data_load_folder:str,
            data_store_folder:str, dataset_name:str
    ):
        self._load_split_dataset(data_load_folder)
        self._convert_to_c_code_after_allocate(data_store_folder, dataset_name)
        
    def get_c_libraries_each_device_from_new(
            self,
            label_allo_method: str, label_allo_loop_step: int,
            data_allo_method: str, data_allo_dirichlet_alpha: int,
            test_data_size: int, valid_data_size: int,
            data_store_folder: str, dataset_name: str
    ):
        self._download_dataset()
        self._split_subdataset_train()
        self._split_subdataset_test()
        self._allocate_labels_to_devices(label_allo_method, label_allo_loop_step)
        self._allocate_train_data_to_devices(data_allo_method, data_allo_dirichlet_alpha)
        self._allocate_test_valid_data(test_data_size, valid_data_size)
        self._convert_to_c_code_after_allocate(data_store_folder, dataset_name)

    def get_training_dataloader_dict(self):
        return self._dataloader_train

    def get_test_dataloader(self):
        return self._dataloader_test

    def get_valid_dataloader(self):
        return self._dataloader_valid

    def get_total_classes(self):
        if not isinstance(self._total_classes, int) or self._total_classes <= 0:
            raise RuntimeError("Dataset class count is unavailable before loading the dataset")
        return self._total_classes
