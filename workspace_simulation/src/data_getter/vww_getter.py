from __future__ import annotations

import logging
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.data_getter.base_getter import BaseDatasetGetter

log = logging.getLogger(__name__)

class VWWSampleDataset(Dataset):
    def __init__(self, fo_dataset, transform=None, label_map=None):
        self.ds = fo_dataset  # Dataset or DatasetView
        self.transform = transform

        self.ids = self.ds.values("id")

        self.filepaths = self.ds.values("filepath")
        self.label_strs = self.ds.values("ground_truth.label")

        if label_map is None:
            label_map = {"background": 0, "person": 1}
        self.label_map = label_map

        self.targets = [self.label_map.get(x, -1) for x in self.label_strs]

        if any(t == -1 for t in self.targets):
            bad = sorted({s for s, t in zip(self.label_strs, self.targets) if t == -1})
            raise ValueError(f"Found unknown labels: {bad}. Update label_map.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.targets[idx]

        with Image.open(fp) as im:
            img = im.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class VisualWakeWords(BaseDatasetGetter):
    def __init__(
        self,
        num_devices: int = 10,
        data_per_device: int = 1000,
        label_per_device: int = 1,
        prefix_name: str = "device_",
    ):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)

    def _download_dataset(self):
        import fiftyone as fo

        try:
            train_fo = fo.Dataset.from_dir(
                dataset_dir='./data/coco-2017-vww-json/train',
                dataset_type=fo.types.FiftyOneImageClassificationDataset,
            )
            val_fo = fo.Dataset.from_dir(
                dataset_dir='./data/coco-2017-vww-json/validation',
                dataset_type=fo.types.FiftyOneImageClassificationDataset,
            )
        except ValueError:
            train_fo = fo.Dataset.from_dir(
                dataset_dir='../../data/coco-2017-vww-json/train',
                dataset_type=fo.types.FiftyOneImageClassificationDataset,
            )
            val_fo = fo.Dataset.from_dir(
                dataset_dir='../../data/coco-2017-vww-json/validation',
                dataset_type=fo.types.FiftyOneImageClassificationDataset,
            )

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self._total_train_dataset = VWWSampleDataset(train_fo, transform=transform)
        self._total_test_dataset = VWWSampleDataset(val_fo, transform=transform)
        self._total_classes = 2

        # Debug: count person/no-person samples
        train_persons = sum(self._total_train_dataset.targets)
        train_no_persons = len(self._total_train_dataset.targets) - train_persons

        test_persons = sum(self._total_test_dataset.targets)
        test_no_persons = len(self._total_test_dataset.targets) - test_persons

        log.info(
            f"Train set: {train_persons} person, {train_no_persons} no-person "
            f"(total {len(self._total_train_dataset)})"
        )
        log.info(
            f"Val set:   {test_persons} person, {test_no_persons} no-person "
            f"(total {len(self._total_test_dataset)})"
        )

        log.info(
            f"VWW dataset prepared: train={len(self._total_train_dataset)}, "
            f"val={len(self._total_test_dataset)}"
        )

if __name__ == "__main__":
    vww = VisualWakeWords(num_devices=10, label_per_device=2, data_per_device=800)
    vww._download_dataset()
    print(len(vww._total_train_dataset))
    print(len(vww._total_test_dataset))

    vww._split_subdataset_train()
    vww._split_subdataset_test()
    # vww._convert_to_c_code_after_split('./dataset/vww/', 'vww')

    vww._allocate_labels_to_devices(method='loop', loop_step=1)

    vww._allocate_train_data_to_devices(method='dirichlet', alpha=1)
    vww._allocate_test_valid_data(test_data_size=600, valid_data_size=0)
    vww._convert_to_c_code_after_allocate('./dataset/vww/', 'vww')

    vww._generate_dataloader()
    vww._training_dataset_visualization(save_path='./dataset/vww/', fig_name='dirichlet')
