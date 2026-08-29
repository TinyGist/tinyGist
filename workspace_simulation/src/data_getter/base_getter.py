from .data_splitter import DataSplit


class BaseDatasetGetter(DataSplit):
    def __init__(self, num_devices=10, data_per_device=500, label_per_device=3, prefix_name='device_'):
        super().__init__()
        self._num_device = num_devices
        self._data_per_device = data_per_device
        self._label_per_device = label_per_device
        self._prefix_name = prefix_name
