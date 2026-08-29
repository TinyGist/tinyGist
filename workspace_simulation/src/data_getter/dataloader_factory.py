"""Validated construction of training, validation, and test DataLoaders."""

import torch
from torch.utils.data import DataLoader


def build_dataloaders(
        training_datasets,
        evaluation_datasets,
        train_batch_size=64,
        test_batch_size=64,
        valid_batch_size=64,
        num_workers=0,
):
    if not isinstance(training_datasets, dict):
        raise RuntimeError("Please allocate training subsets to clients first")
    if not isinstance(evaluation_datasets, dict):
        raise RuntimeError("Please allocate test and validation datasets first")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, got {num_workers!r}")

    empty_devices = [
        device_id
        for device_id, subset in training_datasets.items()
        if len(subset) == 0
    ]
    if empty_devices:
        raise ValueError(f"Training allocation produced empty datasets for {empty_devices}")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    training_loaders = {
        device_id: DataLoader(
            subset,
            train_batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        for device_id, subset in training_datasets.items()
    }
    test_loader = DataLoader(
        evaluation_datasets["test"],
        test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        evaluation_datasets["valid"],
        valid_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return training_loaders, test_loader, validation_loader
