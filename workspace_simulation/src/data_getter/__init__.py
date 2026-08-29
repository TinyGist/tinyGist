from collections.abc import Mapping
from importlib import import_module

from .base_getter import BaseDatasetGetter
from .data_splitter import DataSplit


DATASET_IMPORTS = {
    "mnist": ("src.data_getter.mnist_getter", "MNIST"),
    "emnist": ("src.data_getter.emnist_getter", "EMNIST"),
    "cifar10": ("src.data_getter.cifar10_getter", "Cifar10"),
    "fmnist": ("src.data_getter.fmnist_getter", "FashionMNIST"),
    "vww": ("src.data_getter.vww_getter", "VisualWakeWords"),
    "cifar100": ("src.data_getter.cifar100_getter", "Cifar100"),
    "muscle_gesture": ("src.data_getter.muscle_gesture_getter", "MuscleGesture"),
    "google_speech": ("src.data_getter.google_speech_getter", "GoogleSpeechCommand"),
    "google_speech_kws": ("src.data_getter.kws_getter", "GoogleSpeechKWS"),
    "fomo_person": ("src.data_getter.fomo_dataset_getter", "FomoDetectionPerson"),
    "fomo_vehicle": ("src.data_getter.fomo_dataset_getter", "FomoDetectionVehicle"),
    "fomo_vehicle_binary": ("src.data_getter.fomo_dataset_getter", "FomoDetectionVehicleBinary"),
    "yolo_person": ("src.data_getter.coco_detection_getter", "COCODetectionPerson"),
    "yolo_vehicle": ("src.data_getter.coco_detection_getter", "COCODetectionVehicle"),
    "yolo_vehicle_binary": ("src.data_getter.coco_detection_getter", "COCODetectionVehicleBinary"),
    "svhn": ("src.data_getter.svhn_getter", "SVHN"),
}

CLASS_IMPORTS = {
    class_name: (module_name, class_name)
    for module_name, class_name in DATASET_IMPORTS.values()
}


class LazyDatasetRegistry(Mapping):
    """Resolve only the selected dataset and its optional dependencies."""

    def __init__(self, imports):
        self._imports = dict(imports)
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._imports:
            raise KeyError(name)
        if name not in self._cache:
            module_name, class_name = self._imports[name]
            self._cache[name] = getattr(import_module(module_name), class_name)
        return self._cache[name]

    def __iter__(self):
        return iter(self._imports)

    def __len__(self):
        return len(self._imports)

    def keys(self):
        return self._imports.keys()


DATASETS = LazyDatasetRegistry(DATASET_IMPORTS)


def __getattr__(name):
    target = CLASS_IMPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, class_name = target
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value


__all__ = [
    "DATASETS",
    "DATASET_IMPORTS",
    "LazyDatasetRegistry",
    "DataSplit",
    "BaseDatasetGetter",
    *CLASS_IMPORTS,
]
