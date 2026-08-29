from dataclasses import dataclass

import numpy as np
import torch
from bitmap import BitMap

from .segment_ops import combine_indexed_score_values


@dataclass
class Combo:
    idx: str
    score: float
    seg_param: torch.Tensor
    bitmap: BitMap
    block_idx: list[int] | tuple[int, ...] | None = None
    fisher_seg: torch.Tensor | None = None
    bn_param: torch.Tensor | None = None


def segment_selection_probabilities(mean_values, exponential_normalization, exponential_base):
    mean_values = np.asarray(mean_values, dtype=np.float64)
    if mean_values.size == 0:
        return mean_values.copy(), mean_values.copy()
    total = mean_values.sum()
    if total <= 0:
        base_probabilities = np.full_like(mean_values, 1.0 / len(mean_values))
    else:
        base_probabilities = mean_values / total
    if not exponential_normalization:
        return base_probabilities, base_probabilities

    log_weights = np.log(exponential_base) * base_probabilities
    weights = np.exp(log_weights - log_weights.max())
    return base_probabilities, weights / weights.sum()


def split_block_indices_by_size(block_sizes, segment_count, ordered_block_indices):
    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    ordered_block_indices = [int(block_idx) for block_idx in ordered_block_indices]
    if segment_count <= 0:
        raise ValueError(f"The number of segments from one model is {segment_count}")
    if not ordered_block_indices:
        raise ValueError("Block partition must contain at least one block")
    if len(ordered_block_indices) != len(block_sizes):
        raise ValueError("Block partition must cover the prepared block layout")
    if len(set(ordered_block_indices)) != len(ordered_block_indices):
        raise ValueError("A block appears multiple times in one block partition")
    if not all(0 <= block_idx < len(block_sizes) for block_idx in ordered_block_indices):
        raise ValueError("Block partition contains an invalid block index")

    effective_segment_count = min(segment_count, len(ordered_block_indices))
    target_payload_size = float(block_sizes.sum()) / effective_segment_count
    segments = []
    current_blocks = []
    current_payload_size = 0

    for position, block_idx in enumerate(ordered_block_indices):
        current_blocks.append(block_idx)
        current_payload_size += int(block_sizes[block_idx])
        future_slots = effective_segment_count - len(segments) - 1
        remaining_blocks = len(ordered_block_indices) - position - 1
        can_split = len(segments) < effective_segment_count - 1
        reached_target = current_payload_size >= target_payload_size
        must_reserve_blocks = remaining_blocks <= future_slots
        if can_split and (reached_target or must_reserve_blocks):
            segments.append(current_blocks)
            current_blocks = []
            current_payload_size = 0

    if current_blocks:
        segments.append(current_blocks)
    if [block for segment in segments for block in segment] != ordered_block_indices:
        raise RuntimeError("Block partition changed block order or coverage")
    return segments


def split_scored_block_groups_by_size(
        block_sizes,
        segment_count,
        ordered_block_groups,
        ordered_group_scores=None,
        segment_score_combine="mean",
):
    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    if segment_count <= 0:
        raise ValueError(f"The number of segments from one model is {segment_count}")
    groups = list(ordered_block_groups)
    group_count = len(groups)
    if group_count == 0:
        raise ValueError("Grouped block partition must contain at least one group")

    group_lengths = np.fromiter(
        (len(group) for group in groups),
        dtype=np.int64,
        count=group_count,
    )
    group_scores = None
    if ordered_group_scores is not None:
        if len(ordered_group_scores) != group_count:
            raise ValueError("Grouped block score count must match grouped block count")
        group_scores = np.asarray(ordered_group_scores, dtype=np.float64)
    flattened_order = np.fromiter(
        (int(block_idx) for group in groups for block_idx in group),
        dtype=np.int64,
        count=int(group_lengths.sum()),
    )
    if len(flattened_order) != len(block_sizes):
        raise ValueError("Grouped block partition must cover the prepared block layout")
    if not np.all((0 <= flattened_order) & (flattened_order < len(block_sizes))):
        raise ValueError("Grouped block partition contains an invalid block index")
    if not np.all(np.bincount(flattened_order, minlength=len(block_sizes)) == 1):
        raise ValueError("A block appears multiple times in one grouped block partition")

    effective_segment_count = min(segment_count, group_count)
    target_payload_size = float(block_sizes.sum()) / effective_segment_count
    group_payload_sizes = combine_indexed_score_values(block_sizes, groups, "sum")
    if group_count <= effective_segment_count:
        group_ranges = [(position, position + 1) for position in range(group_count)]
    else:
        cumulative_payload = np.cumsum(group_payload_sizes)
        group_ranges = []
        start = 0
        for segment_position in range(effective_segment_count - 1):
            payload_before = cumulative_payload[start - 1] if start else 0.0
            payload_end = int(np.searchsorted(
                cumulative_payload,
                payload_before + target_payload_size,
                side="left",
            )) + 1
            future_slots = effective_segment_count - segment_position - 1
            latest_end = group_count - future_slots
            end = min(max(start + 1, payload_end), latest_end)
            group_ranges.append((start, end))
            start = end
        group_ranges.append((start, group_count))

    group_offsets = np.concatenate(([0], np.cumsum(group_lengths)))
    segments = [
        flattened_order[group_offsets[start]:group_offsets[end]].tolist()
        for start, end in group_ranges
    ]
    segment_scores = None
    if group_scores is not None:
        segment_scores = combine_indexed_score_values(
            group_scores,
            [list(range(start, end)) for start, end in group_ranges],
            segment_score_combine,
        ).tolist()
    return segments, segment_scores


def ordered_group_positions(block_groups, group_scores):
    if len(block_groups) != len(group_scores):
        raise ValueError("Grouped block score count must match grouped block count")
    group_lengths = np.fromiter(
        (len(group) for group in block_groups),
        dtype=np.int64,
        count=len(block_groups),
    )
    if not np.all(group_lengths > 0):
        raise ValueError("Block groups must not be empty")
    flat_indices = np.fromiter(
        (int(block_idx) for group in block_groups for block_idx in group),
        dtype=np.int64,
        count=int(group_lengths.sum()),
    )
    starts = np.concatenate(([0], np.cumsum(group_lengths[:-1])))
    minimum_block_indices = np.minimum.reduceat(flat_indices, starts)
    return np.lexsort(
        (minimum_block_indices, np.asarray(group_scores, dtype=np.float64)),
    ).tolist()


def split_parameter_indices(index_array, segment_count):
    """Split parameter indices without creating empty communication segments."""
    if segment_count <= 0:
        raise ValueError(f"The number of segments from one model is {segment_count}")
    indices = np.asarray(index_array)
    if indices.ndim != 1:
        raise ValueError("Parameter indices must be one-dimensional")
    if indices.size == 0:
        return []
    effective_segment_count = min(segment_count, indices.size)
    return [
        segment.tolist()
        for segment in np.array_split(indices, effective_segment_count)
        if segment.size
    ]
