"""Poisson-sampled DataLoaders derived from existing training loaders."""

import torch
from torch.utils.data import DataLoader, Sampler


class PoissonBatchSampler(Sampler):
    def __init__(
            self,
            dataset_size,
            expected_batch_size,
            *,
            steps_per_epoch,
            generator=None,
    ):
        self.dataset_size = int(dataset_size)
        self.expected_batch_size = min(
            int(expected_batch_size),
            self.dataset_size,
        )
        if self.dataset_size <= 0:
            raise ValueError("DP-SGD requires a non-empty local dataset")
        if self.expected_batch_size <= 0:
            raise ValueError("DP-SGD expected batch size must be positive")
        self.sample_rate = self.expected_batch_size / self.dataset_size
        self.steps_per_epoch = int(steps_per_epoch)
        if self.steps_per_epoch <= 0:
            raise ValueError("DP-SGD steps per epoch must be positive")
        self.generator = generator

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            selected = torch.rand(
                self.dataset_size,
                generator=self.generator,
            ) < self.sample_rate
            yield selected.nonzero(as_tuple=False).flatten().tolist()

    def __len__(self):
        return self.steps_per_epoch


class _EmptyAwareCollate:
    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, samples):
        if not samples:
            return None
        return self.collate_fn(samples)


def build_poisson_dataloader(base_loader, *, steps_per_epoch, generator=None):
    if not isinstance(base_loader, DataLoader):
        raise TypeError("DP-SGD requires a torch DataLoader")
    if base_loader.batch_size is None:
        raise ValueError("DP-SGD requires an existing DataLoader batch_size")
    sampler = PoissonBatchSampler(
        len(base_loader.dataset),
        base_loader.batch_size,
        steps_per_epoch=steps_per_epoch,
        generator=generator,
    )
    kwargs = {
        "dataset": base_loader.dataset,
        "batch_sampler": sampler,
        "num_workers": base_loader.num_workers,
        "collate_fn": _EmptyAwareCollate(base_loader.collate_fn),
        "pin_memory": base_loader.pin_memory,
        "timeout": base_loader.timeout,
        "worker_init_fn": base_loader.worker_init_fn,
        "persistent_workers": base_loader.persistent_workers,
    }
    if base_loader.num_workers > 0:
        kwargs["prefetch_factor"] = base_loader.prefetch_factor
        if base_loader.multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = base_loader.multiprocessing_context
    pin_memory_device = getattr(base_loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return DataLoader(**kwargs), sampler
