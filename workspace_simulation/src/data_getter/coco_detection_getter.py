from __future__ import annotations

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from functools import partial
import logging

from src.data_getter.base_getter import BaseDatasetGetter

log = logging.getLogger(__name__)

class COCOGetter(Dataset):
    def __init__(self, fo_dataset, task="human", transform=None, S=5, B=2):
        self.ds = fo_dataset
        self.task = task
        self.transform = transform
        self.S = S
        self.B = B

        self.filepaths = self.ds.values("filepath")
        self.detections = self.ds.values("detections")
        self.metadata = self.ds.values("metadata")
        self.num_classes = 1

        if self.task == "human":
            self.labelstr_to_idx = {
                "person": 0,
            }
        elif self.task == "vehicle":
            self.labelstr_to_idx = {
                "bicycle": 0,
                "bus": 1,
                "car": 2,
                "motorcycle": 3,
                "truck": 4,
            }
            self.num_classes = 5
        elif self.task == "vehicle-binary":
            self.labelstr_to_idx = {
                "bicycle": 0,
                "bus": 0,
                "car": 0,
                "motorcycle": 0,
                "truck": 0,
            }
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        det = self.detections[idx]
        detection_info = det["detections"] if det is not None else []
        width, height = self.metadata[idx]["width"], self.metadata[idx]["height"]
        labelstrs = []
        bboxes = []
        with Image.open(filepath) as img:
            img = img.convert("RGB")
        for info in detection_info:
            labelstr, bbox = info["label"], info["bounding_box"]
            labelstrs.append(labelstr)
            bboxes.append(bbox)
        if len(bboxes) > self.B:
            # Keep a fixed target shape while retaining the intended random
            # target selection for crowded images.
            selected = np.random.choice(len(bboxes), size=self.B, replace=False)
            bboxes = [bboxes[i] for i in selected]
            labelstrs = [labelstrs[i] for i in selected]
        if len(bboxes) < self.B:
            for i in range(0, self.B - len(bboxes)):
                bboxes.append([0,0,0,0])
                labelstrs.append("background")

        final_label = self.generate_label_from_existed_box(width=width, height=height, bboxes=bboxes, labelstrs=labelstrs)

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(final_label).float()

    def generate_label_from_existed_box(self, width, height, bboxes, labelstrs):
        flat_boxes_list = []
        if self.task == "human":
            num_classes = 1
        elif self.task == "vehicle":
            num_classes = 5
        elif self.task == "vehicle-binary":
            num_classes = 1
        else:
            raise NotImplementedError
        cls_label = np.zeros((self.S, self.S, num_classes), dtype=np.float32)
        for bbox, labelstr in zip(bboxes, labelstrs):
            box_x, box_y, box_w, box_h = bbox
            box_center = (box_x + box_w / 2, box_y + box_h / 2)
            gx = min(int(np.floor(box_center[0] * self.S)), self.S - 1)
            gy = min(int(np.floor(box_center[1] * self.S)), self.S - 1)
            if labelstr == "background":
                label = -1
            else:
                label = self.labelstr_to_idx[labelstr]
            if width*box_w*height*box_h < 12*12:
                confidence = 0.0
                label = -1
            else:
                confidence = 1.0

            if label != -1:
                cls_label[gy, gx, label] = 1.0

            flat_boxes_list.append(np.array([box_x, box_y, box_w, box_h, confidence]))
        flat_label_list = [cls_label.reshape(-1)] + flat_boxes_list
        flat_label = np.concatenate(flat_label_list)

        return flat_label

class COCODetection(BaseDatasetGetter):
    def __init__(self, num_devices=20, data_per_device=1000, label_per_device=1, prefix_name='device_', task='human'):
        super().__init__(num_devices, data_per_device, label_per_device, prefix_name)
        self.__task = task

    def _download_dataset(self):
        import fiftyone as fo

        if self.__task == "vehicle-binary":
            path = "vehicle"
        else:
            path = self.__task
        try:
            train_fo = fo.Dataset.from_dir(
                dataset_dir=f"./data/coco-2017-yolo-{path}/train",
                dataset_type=fo.types.COCODetectionDataset,
            )
            val_fo = fo.Dataset.from_dir(
                dataset_dir=f"./data/coco-2017-yolo-{path}/validation",
                dataset_type=fo.types.COCODetectionDataset,
            )
        except (FileNotFoundError, ValueError):
            train_fo = fo.Dataset.from_dir(
                dataset_dir=f"../../data/coco-2017-yolo-{path}/train",
                dataset_type=fo.types.COCODetectionDataset,
            )
            val_fo = fo.Dataset.from_dir(
                dataset_dir=f"../../data/coco-2017-yolo-{path}/validation",
                dataset_type=fo.types.COCODetectionDataset,
            )

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self._total_train_dataset = COCOGetter(train_fo, task=self.__task, transform=transform)
        self._total_test_dataset = COCOGetter(val_fo, task=self.__task, transform=transform)

        self._semantic_classes = self._total_train_dataset.num_classes
        # DataSplit partitions detection images as one dataset category. The
        # detector's semantic class count remains on COCOGetter/model outputs.
        self._total_classes = 1
        log.info(f'There are {self._semantic_classes} semantic classes in current dataset')

COCODetectionPerson = partial(COCODetection, task="human")
COCODetectionVehicle = partial(COCODetection, task="vehicle")
COCODetectionVehicleBinary = partial(COCODetection, task="vehicle-binary")

if __name__ == '__main__':
    detection = COCODetectionPerson(num_devices=20, data_per_device=1000)
    detection._download_dataset()
    detection._split_subdataset_train()
    detection._split_subdataset_test()
    detection._allocate_labels_to_devices()
    detection._allocate_train_data_to_devices()
    detection._allocate_test_valid_data()
    detection._generate_dataloader()

    detection = COCODetectionVehicle(num_devices=20, data_per_device=1000)
    detection._download_dataset()
    detection._split_subdataset_train()
    detection._split_subdataset_test()
    detection._allocate_labels_to_devices()
    detection._allocate_train_data_to_devices()
    detection._allocate_test_valid_data()
    detection._generate_dataloader()
    pass
