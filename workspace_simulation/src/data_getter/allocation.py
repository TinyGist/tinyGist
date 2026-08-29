"""Pure helpers for constructing per-device dataset allocations."""

import logging

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset


log = logging.getLogger(__name__)


def apportion_integer_counts(total, weights):
    """Convert non-negative proportions to integer counts summing to ``total``."""
    if isinstance(total, bool) or not isinstance(total, (int, np.integer)) or total < 0:
        raise ValueError(f"total must be a non-negative integer, got {total!r}")
    weights = np.asarray(weights, dtype=np.float64)
    if (
            weights.ndim != 1
            or weights.size == 0
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
    ):
        raise ValueError("weights must be a non-empty finite non-negative vector")

    exact_counts = weights / weights.sum() * int(total)
    integer_counts = np.floor(exact_counts).astype(np.int64)
    remainder = int(total) - int(integer_counts.sum())
    if remainder:
        fractional_parts = exact_counts - integer_counts
        order = np.argsort(-fractional_parts, kind="stable")
        integer_counts[order[:remainder]] += 1
    return integer_counts.tolist()


def build_training_dataset(label_to_subset_train, labels, sample_counts, logger=log):
    """Build one shuffled client dataset without changing the RNG call sequence."""
    combined_parts = []
    for label_id, sample_count in zip(labels, sample_counts):
        subset = label_to_subset_train[label_id]
        if not isinstance(subset, Subset):
            raise TypeError(f"Training label {label_id} is not a Subset")
        subset_length = len(subset)
        if subset_length == 0:
            raise ValueError(
                f"Training label {label_id} has no samples after validation split"
            )
        replicate_times, extra_length = divmod(sample_count, subset_length)
        if replicate_times:
            logger.warning(f"dataset of label {label_id} is too small, so replicate it")
            combined_parts.extend([subset] * replicate_times)
        if extra_length:
            random_indices = torch.randperm(subset_length)[:extra_length]
            combined_parts.append(Subset(subset, random_indices))

    if not combined_parts:
        raise ValueError("Training allocation produced an empty dataset")
    combined_dataset = ConcatDataset(combined_parts)
    return Subset(combined_dataset, torch.randperm(len(combined_dataset)))
