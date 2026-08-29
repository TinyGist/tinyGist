import numpy as np
import torch
from bitmap import BitMap


def create_bitmap(bitmap_length, bitmap_ones_idx) -> BitMap:
    bitmap = BitMap(bitmap_length)
    for idx in bitmap_ones_idx:
        bitmap.set(idx)
    return bitmap


def create_segments_based_bitmap(model_parameters: torch.Tensor, bitmap_list: list[BitMap]) -> list[torch.Tensor]:
    return [model_parameters[bitmap.nonzero()] for bitmap in bitmap_list]


def flatten_parameter_block(parameter_block) -> torch.Tensor:
    weight, bias = parameter_block
    flat_weight = weight.reshape(-1)
    if bias is None:
        return flat_weight
    return torch.cat((flat_weight, bias.reshape(-1)))


def parameter_block_size(parameter_block) -> int:
    weight, bias = parameter_block
    size = weight.numel()
    if bias is not None:
        size += bias.numel()
    return size


def compute_parameter_block_score_tensor(parameter_block, block_score_method: str) -> torch.Tensor:
    weight, bias = parameter_block
    block_size = parameter_block_size(parameter_block)
    assert block_size > 0, "Parameter block must not be empty"

    if block_score_method == "mean_abs":
        score_sum = weight.abs().sum()
        if bias is not None:
            score_sum = score_sum + bias.abs().sum()
        score = score_sum / block_size
    elif block_score_method == "rms":
        squared_sum = torch.square(weight).sum()
        if bias is not None:
            squared_sum = squared_sum + torch.square(bias).sum()
        score = torch.sqrt(squared_sum / block_size)
    elif block_score_method == "l2":
        squared_sum = torch.square(weight).sum()
        if bias is not None:
            squared_sum = squared_sum + torch.square(bias).sum()
        score = torch.sqrt(squared_sum)
    elif block_score_method == "lipschitz":
        out_dim = bias.numel() if bias is not None else 1
        mat = weight.reshape(out_dim, -1)
        if out_dim == 1:
            score = mat.norm(p=2)
        else:
            score = torch.linalg.matrix_norm(mat, p=2)
    else:
        raise ValueError(f"Invalid block_score_method [{block_score_method}]")
    return score.to(dtype=torch.float32)


def compute_parameter_block_score(parameter_block, block_score_method: str) -> float:
    return float(compute_parameter_block_score_tensor(parameter_block, block_score_method))


def reduce_parameter_score_block_tensor(parameter_score_block, block_score_method: str) -> torch.Tensor:
    """Reduce a block of per-parameter scores to one block score.

    This is intentionally separate from compute_parameter_block_score(...):
    Lipschitz is a model-block property and must be computed from weights, not
    from a block of already-recorded parameter scores such as Fisher.
    """
    if block_score_method == "lipschitz":
        raise ValueError("lipschitz block score must be computed from model parameter blocks")
    return compute_parameter_block_score_tensor(parameter_score_block, block_score_method)


def reduce_parameter_score_block(parameter_score_block, block_score_method: str) -> float:
    return float(reduce_parameter_score_block_tensor(parameter_score_block, block_score_method))


def combine_score_values(values, combine_method: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if combine_method == "mean":
        return float(np.mean(values))
    if combine_method == "sum":
        return float(np.sum(values))
    if combine_method == "max":
        return float(np.max(values))
    squared_sum = np.sum(np.square(values))
    if combine_method == "l2":
        return float(np.sqrt(squared_sum))
    if combine_method == "rms":
        return float(np.sqrt(squared_sum / values.size))
    raise ValueError(f"Unsupported score combine method [{combine_method}]")


def combine_indexed_score_values(values, index_groups, combine_method: str) -> np.ndarray:
    """Combine many score groups without one Python/NumPy call per group."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    group_lengths = np.fromiter(
        (len(index_group) for index_group in index_groups),
        dtype=np.int64,
        count=len(index_groups),
    )
    combined = np.zeros(len(group_lengths), dtype=np.float64)
    nonempty_positions = np.flatnonzero(group_lengths)
    if nonempty_positions.size == 0:
        return combined

    nonempty_groups = [index_groups[position] for position in nonempty_positions]
    nonempty_lengths = group_lengths[nonempty_positions]
    flat_indices = np.fromiter(
        (int(index) for index_group in nonempty_groups for index in index_group),
        dtype=np.int64,
        count=int(nonempty_lengths.sum()),
    )
    if np.any(flat_indices < 0) or np.any(flat_indices >= len(values)):
        raise IndexError("Score group contains an invalid index")

    starts = np.concatenate(([0], np.cumsum(nonempty_lengths[:-1])))
    selected = values[flat_indices]
    if combine_method == "max":
        reduced = np.maximum.reduceat(selected, starts)
    else:
        squared = combine_method in {"l2", "rms"}
        reduced = np.add.reduceat(np.square(selected) if squared else selected, starts)
        if combine_method in {"mean", "rms"}:
            reduced = reduced / nonempty_lengths
        elif combine_method not in {"sum", "l2"}:
            raise ValueError(f"Unsupported score combine method [{combine_method}]")
        if squared:
            reduced = np.sqrt(reduced)
    combined[nonempty_positions] = reduced
    return combined


def create_segment_based_block_bitmap(parameter_blocks: list, block_bitmap: BitMap) -> torch.Tensor:
    """Flatten selected blocks in block-bitmap order for transmission.

    The bitmap indexes blocks, not original flattened model parameters. Receivers
    can parse the flat payload because every model builds the same block list.
    """
    block_idx_list = block_bitmap.nonzero()
    if len(block_idx_list) == 0:
        return torch.empty(0)
    return torch.cat([
        flatten_parameter_block(parameter_blocks[block_idx])
        for block_idx in block_idx_list
    ])


def create_segments_based_block_bitmap(parameter_blocks: list, bitmap_list: list[BitMap]) -> list[torch.Tensor]:
    return [
        create_segment_based_block_bitmap(parameter_blocks, bitmap)
        for bitmap in bitmap_list
    ]


def _unflatten_parameter_block(flat_block: torch.Tensor, reference_block):
    reference_weight, reference_bias = reference_block
    weight_num = reference_weight.numel()
    weight = flat_block[:weight_num].view_as(reference_weight)
    bias = None
    if reference_bias is not None:
        bias = flat_block[weight_num:weight_num + reference_bias.numel()].view_as(reference_bias)
    return weight, bias


def _aggregation_weight_tensor(weight, reference, expected_numel, value_name):
    weight_dtype = (
        torch.float64
        if reference.dtype == torch.float64
        else torch.float32
    )
    weight = torch.as_tensor(
        weight,
        device=reference.device,
        dtype=weight_dtype,
    ).reshape(-1)
    if weight.numel() == 1 and expected_numel != 1:
        weight = weight.expand(expected_numel)
    if weight.numel() != expected_numel:
        raise ValueError(f"{value_name} length does not match its parameter values")
    if not torch.all(torch.isfinite(weight)):
        raise ValueError(f"{value_name} contains non-finite values")
    if torch.any(weight < 0):
        raise ValueError(f"{value_name} contains negative values")
    return weight


def _safe_weighted_average(weighted_values, weights, fallback_values):
    valid = weights > 0
    result = fallback_values.to(
        device=weighted_values.device,
        dtype=weighted_values.dtype,
    ).detach().clone()
    result[valid] = weighted_values[valid] / weights[valid]
    return result


def _aggregate_indexed_weighted_values(
        target_values,
        target_weight,
        received_values,
        received_indices,
        received_weights,
        exponential_base=None,
        fallback_values=None,
):
    if not (
            len(received_values)
            == len(received_indices)
            == len(received_weights)
    ):
        raise ValueError("Received values, indexes, and weights must have equal lengths")
    target_values = target_values.detach().reshape(-1)
    target_dtype = target_values.dtype
    fallback_values = (
        target_values
        if fallback_values is None
        else fallback_values.detach().reshape(-1)
    )
    if fallback_values.numel() != target_values.numel():
        raise ValueError("Fallback and target parameter lengths differ")

    target_weight = _aggregation_weight_tensor(
        target_weight,
        target_values,
        target_values.numel(),
        "Local aggregation weight",
    )
    prepared = []
    raw_weight_sum = target_weight.clone()
    for values, indices, weights in zip(
            received_values,
            received_indices,
            received_weights,
    ):
        indices = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=target_values.device,
        ).reshape(-1)
        values = values.to(
            device=target_values.device,
            dtype=target_values.dtype,
        ).detach().reshape(-1)
        if values.numel() != indices.numel():
            raise ValueError("Received payload length does not match parameter indexes")
        if torch.any(indices < 0) or torch.any(indices >= target_values.numel()):
            raise IndexError("Received parameter index is out of range")
        weights = _aggregation_weight_tensor(
            weights,
            target_values,
            indices.numel(),
            "Received aggregation weight",
        )
        prepared.append((values, indices, weights))
        raw_weight_sum.index_add_(0, indices, weights)

    if exponential_base is None:
        effective_target_weight = target_weight
        effective_received_weights = [weights for _, _, weights in prepared]
    else:
        if isinstance(exponential_base, bool):
            raise ValueError("Aggregation weight exponential base must exceed 1")
        exponential_base = float(exponential_base)
        if not np.isfinite(exponential_base) or exponential_base <= 1:
            raise ValueError("Aggregation weight exponential base must exceed 1")
        max_weight = target_weight.clone()
        for _, indices, weights in prepared:
            max_weight.scatter_reduce_(
                0,
                indices,
                weights,
                reduce="amax",
                include_self=True,
            )
        denominator = torch.where(
            raw_weight_sum > 0,
            raw_weight_sum,
            torch.ones_like(raw_weight_sum),
        )
        normalized_max = max_weight / denominator
        log_base = float(np.log(exponential_base))
        normalized_target = target_weight / denominator
        effective_target_weight = torch.exp(
            log_base * (normalized_target - normalized_max)
        )
        effective_received_weights = []
        for _, indices, weights in prepared:
            normalized = weights / denominator[indices]
            effective_received_weights.append(torch.exp(
                log_base * (normalized - normalized_max[indices])
            ))

    weighted_values = target_values * effective_target_weight
    effective_weight_sum = effective_target_weight.clone()
    for (values, indices, _), weights in zip(
            prepared,
            effective_received_weights,
    ):
        weighted_values.index_add_(0, indices, values * weights)
        effective_weight_sum.index_add_(0, indices, weights)
    if not torch.all(torch.isfinite(effective_weight_sum)):
        raise ValueError("Processed aggregation weights contain non-finite values")
    result = _safe_weighted_average(
        weighted_values,
        effective_weight_sum,
        fallback_values,
    ).to(dtype=target_dtype)
    if not torch.all(torch.isfinite(result)):
        raise ValueError("Aggregated packed parameter vector contains non-finite values")
    return result


def aggregate_dense_weighted_values(
        values,
        weights,
        exponential_base=None,
):
    if len(values) != len(weights) or not values:
        raise ValueError("Dense aggregation requires equally sized non-empty inputs")
    target = values[0].detach().reshape(-1)
    prepared_values = [
        value.to(device=target.device, dtype=target.dtype).detach().reshape(-1)
        for value in values
    ]
    if any(value.numel() != target.numel() for value in prepared_values):
        raise ValueError("Dense aggregation parameter lengths differ")
    indices = torch.arange(target.numel(), device=target.device)
    fallback = torch.stack(prepared_values).mean(dim=0)
    return _aggregate_indexed_weighted_values(
        target,
        weights[0],
        prepared_values[1:],
        [indices] * (len(values) - 1),
        weights[1:],
        exponential_base=exponential_base,
        fallback_values=fallback,
    )


def aggregate_segment_scores_based(
        target_params,
        received_seg_list,
        received_bitmap_list,
        scores_list,
        target_score,
        exponential_base=None,
):
    if len(received_seg_list) != len(received_bitmap_list):
        raise ValueError("Received segment and bitmap counts differ")
    return _aggregate_indexed_weighted_values(
        target_params,
        target_score,
        received_seg_list,
        [bitmap.nonzero() for bitmap in received_bitmap_list],
        scores_list,
        exponential_base=exponential_base,
    )



def aggregate_packed_segments_scores_based(
        target_params,
        received_seg_list,
        received_parameter_idx_list,
        scores_list,
        target_score,
        exponential_base=None,
):
    return _aggregate_indexed_weighted_values(
        target_params,
        target_score,
        received_seg_list,
        received_parameter_idx_list,
        scores_list,
        exponential_base=exponential_base,
    )


def aggregate_packed_segments_parameter_fisher_based(
        target_params,
        target_fisher,
        received_seg_list,
        received_parameter_idx_list,
        received_fisher_list,
        exponential_base=None,
):
    if target_params.numel() != target_fisher.numel():
        raise ValueError("Local Fisher weight length does not match parameters")
    return _aggregate_indexed_weighted_values(
        target_params,
        target_fisher,
        received_seg_list,
        received_parameter_idx_list,
        received_fisher_list,
        exponential_base=exponential_base,
    )


def aggregate_packed_segments_block_fisher_based(
        target_params,
        target_block_fisher,
        block_sizes,
        received_seg_list,
        received_parameter_idx_list,
        received_block_idx_list,
        received_fisher_list,
        exponential_base=None,
):
    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    if target_block_fisher.numel() != len(block_sizes):
        raise ValueError("Local block Fisher length does not match block count")
    size_tensor = torch.as_tensor(
        block_sizes,
        dtype=torch.long,
        device=target_params.device,
    )
    expanded_target_fisher = torch.repeat_interleave(
        target_block_fisher.to(
            device=target_params.device,
            dtype=target_params.dtype,
        ),
        size_tensor,
    )
    expanded_received_fisher = []
    for block_idx, fisher in zip(received_block_idx_list, received_fisher_list):
        selected_sizes = torch.as_tensor(
            block_sizes[np.asarray(block_idx, dtype=np.int64)],
            dtype=torch.long,
            device=target_params.device,
        )
        expanded_received_fisher.append(torch.repeat_interleave(
            fisher.to(device=target_params.device, dtype=target_params.dtype),
            selected_sizes,
        ))
    return _aggregate_indexed_weighted_values(
        target_params,
        expanded_target_fisher,
        received_seg_list,
        received_parameter_idx_list,
        expanded_received_fisher,
        exponential_base=exponential_base,
    )


def aggregate_segment_fisher_based(
        target_params,
        target_fisher,
        received_seg_list,
        received_bitmap_list,
        received_fisher_list,
        exponential_base=None,
):
    if target_params.numel() != target_fisher.numel():
        raise ValueError("Local Fisher weight length does not match parameters")
    return _aggregate_indexed_weighted_values(
        target_params,
        target_fisher,
        received_seg_list,
        [bitmap.nonzero() for bitmap in received_bitmap_list],
        received_fisher_list,
        exponential_base=exponential_base,
    )

def aggregate_block_segments_scores_based(
        target_blocks: list,
        received_seg_list: list[torch.Tensor],
        received_block_bitmap_list: list[BitMap],
        scores_list: list,
        target_score,
) -> list:
    """Aggregate flattened block payloads using block bitmaps and local shapes."""
    assert len(received_seg_list) == len(received_block_bitmap_list), \
        "Invalid pairs, the number of received segments is not equal to the number of received bitmaps"
    assert len(scores_list) == len(received_block_bitmap_list), \
        "Invalid pairs, the number of scores is not equal to the number of received bitmaps"

    aggregated_blocks = [
        (
            weight.detach().clone() * target_score,
            None if bias is None else bias.detach().clone() * target_score,
        )
        for weight, bias in target_blocks
    ]
    block_score_counts = [float(target_score) for _ in target_blocks]

    for seg, bitmap, score in zip(received_seg_list, received_block_bitmap_list, scores_list):
        cursor = 0
        for block_idx in bitmap.nonzero():
            reference_block = target_blocks[block_idx]
            block_size = parameter_block_size(reference_block)
            assert cursor + block_size <= seg.numel(), \
                "Invalid block segment, received payload is shorter than block bitmap requires"
            flat_block = seg[cursor:cursor + block_size]
            received_weight, received_bias = _unflatten_parameter_block(flat_block, reference_block)
            aggregated_weight, aggregated_bias = aggregated_blocks[block_idx]
            received_weight = received_weight.to(device=aggregated_weight.device, dtype=aggregated_weight.dtype)
            aggregated_weight = aggregated_weight + received_weight * score
            if aggregated_bias is not None:
                received_bias = received_bias.to(device=aggregated_bias.device, dtype=aggregated_bias.dtype)
                aggregated_bias = aggregated_bias + received_bias * score
            aggregated_blocks[block_idx] = (aggregated_weight, aggregated_bias)
            block_score_counts[block_idx] += float(score)
            cursor += block_size

        assert cursor == seg.numel(), \
            "Invalid block segment, received payload does not match block bitmap"

    averaged_blocks = []
    for block_idx, (aggregated_weight, aggregated_bias) in enumerate(aggregated_blocks):
        denominator = block_score_counts[block_idx]
        if denominator <= 0:
            averaged_blocks.append(_clone_block(target_blocks[block_idx]))
            continue
        averaged_blocks.append((
            aggregated_weight / denominator,
            None if aggregated_bias is None else aggregated_bias / denominator,
        ))
    return averaged_blocks


def _clone_block(block):
    weight, bias = block
    return weight.detach().clone(), None if bias is None else bias.detach().clone()


def _scale_block(block, score):
    weight, bias = block
    weight_score = torch.as_tensor(score, device=weight.device, dtype=weight.dtype)
    bias_score = None if bias is None else weight_score.to(device=bias.device, dtype=bias.dtype)
    return (
        weight.detach().clone() * weight_score,
        None if bias is None else bias.detach().clone() * bias_score,
    )


def _safe_divide_block(weighted_block, weight_block, fallback_block):
    weighted_weight, weighted_bias = weighted_block
    weight_weight, weight_bias = weight_block
    fallback_weight, fallback_bias = fallback_block

    averaged_weight = _safe_weighted_average(weighted_weight, weight_weight, fallback_weight)
    averaged_bias = None
    if weighted_bias is not None:
        averaged_bias = _safe_weighted_average(weighted_bias, weight_bias, fallback_bias)
    return averaged_weight, averaged_bias


def aggregate_block_segments_parameter_fisher_based(
        target_blocks: list,
        target_fisher_blocks: list,
        received_seg_list: list[torch.Tensor],
        received_block_bitmap_list: list[BitMap],
        received_fisher_list: list[torch.Tensor],
) -> list:
    assert len(target_blocks) == len(target_fisher_blocks), \
        "Invalid local Fisher blocks, parameter and Fisher block counts differ"
    assert len(received_seg_list) == len(received_block_bitmap_list), \
        "Invalid pairs, the number of received segments is not equal to the number of received bitmaps"
    assert len(received_fisher_list) == len(received_block_bitmap_list), \
        "Invalid pairs, the number of Fisher segments is not equal to the number of received bitmaps"

    aggregated_blocks = []
    fisher_sum_blocks = []
    for target_block, fisher_block in zip(target_blocks, target_fisher_blocks):
        target_weight, target_bias = target_block
        fisher_weight, fisher_bias = fisher_block
        fisher_weight = fisher_weight.to(device=target_weight.device, dtype=target_weight.dtype)
        weighted_weight = target_weight.detach().clone() * fisher_weight
        weighted_bias = None
        if target_bias is not None:
            assert fisher_bias is not None, "Invalid local Fisher block, missing bias Fisher weights"
            fisher_bias = fisher_bias.to(device=target_bias.device, dtype=target_bias.dtype)
            weighted_bias = target_bias.detach().clone() * fisher_bias
        aggregated_blocks.append((weighted_weight, weighted_bias))
        fisher_sum_blocks.append((
            fisher_weight.detach().clone(),
            None if fisher_bias is None else fisher_bias.detach().clone(),
        ))

    for seg, bitmap, fisher_seg in zip(received_seg_list, received_block_bitmap_list, received_fisher_list):
        cursor = 0
        fisher_cursor = 0
        for block_idx in bitmap.nonzero():
            reference_block = target_blocks[block_idx]
            block_size = parameter_block_size(reference_block)
            assert cursor + block_size <= seg.numel(), \
                "Invalid block segment, received payload is shorter than block bitmap requires"
            assert fisher_cursor + block_size <= fisher_seg.numel(), \
                "Invalid Fisher segment, received Fisher payload is shorter than block bitmap requires"
            flat_block = seg[cursor:cursor + block_size]
            flat_fisher = fisher_seg[fisher_cursor:fisher_cursor + block_size]
            received_weight, received_bias = _unflatten_parameter_block(flat_block, reference_block)
            received_fisher_weight, received_fisher_bias = _unflatten_parameter_block(flat_fisher, reference_block)

            aggregated_weight, aggregated_bias = aggregated_blocks[block_idx]
            fisher_weight_sum, fisher_bias_sum = fisher_sum_blocks[block_idx]
            received_weight = received_weight.to(device=aggregated_weight.device, dtype=aggregated_weight.dtype)
            received_fisher_weight = received_fisher_weight.to(
                device=aggregated_weight.device,
                dtype=aggregated_weight.dtype,
            )
            aggregated_weight = aggregated_weight + received_weight * received_fisher_weight
            fisher_weight_sum = fisher_weight_sum + received_fisher_weight

            if aggregated_bias is not None:
                received_bias = received_bias.to(device=aggregated_bias.device, dtype=aggregated_bias.dtype)
                received_fisher_bias = received_fisher_bias.to(
                    device=aggregated_bias.device,
                    dtype=aggregated_bias.dtype,
                )
                aggregated_bias = aggregated_bias + received_bias * received_fisher_bias
                fisher_bias_sum = fisher_bias_sum + received_fisher_bias

            aggregated_blocks[block_idx] = (aggregated_weight, aggregated_bias)
            fisher_sum_blocks[block_idx] = (fisher_weight_sum, fisher_bias_sum)
            cursor += block_size
            fisher_cursor += block_size

        assert cursor == seg.numel(), \
            "Invalid block segment, received payload does not match block bitmap"
        assert fisher_cursor == fisher_seg.numel(), \
            "Invalid Fisher segment, received payload does not match block bitmap"

    return [
        _safe_divide_block(aggregated_block, fisher_sum_block, target_block)
        for aggregated_block, fisher_sum_block, target_block in zip(
            aggregated_blocks,
            fisher_sum_blocks,
            target_blocks,
        )
    ]


def aggregate_block_segments_block_fisher_based(
        target_blocks: list,
        target_block_fisher: torch.Tensor,
        received_seg_list: list[torch.Tensor],
        received_block_bitmap_list: list[BitMap],
        received_fisher_list: list[torch.Tensor],
) -> list:
    assert target_block_fisher.numel() == len(target_blocks), \
        "Invalid local block Fisher, Fisher length does not match block count"
    assert len(received_seg_list) == len(received_block_bitmap_list), \
        "Invalid pairs, the number of received segments is not equal to the number of received bitmaps"
    assert len(received_fisher_list) == len(received_block_bitmap_list), \
        "Invalid pairs, the number of Fisher segments is not equal to the number of received bitmaps"

    target_block_fisher = target_block_fisher.detach()
    aggregated_blocks = [
        _scale_block(block, target_block_fisher[block_idx])
        for block_idx, block in enumerate(target_blocks)
    ]
    block_fisher_sum = [
        target_block_fisher[block_idx].to(
            device=target_blocks[block_idx][0].device,
            dtype=target_blocks[block_idx][0].dtype,
        ).clone()
        for block_idx in range(len(target_blocks))
    ]

    for seg, bitmap, fisher_seg in zip(received_seg_list, received_block_bitmap_list, received_fisher_list):
        block_idx_list = bitmap.nonzero()
        assert fisher_seg.numel() == len(block_idx_list), \
            "Invalid block Fisher segment, Fisher length does not match block bitmap"
        cursor = 0
        for fisher_position, block_idx in enumerate(block_idx_list):
            reference_block = target_blocks[block_idx]
            block_size = parameter_block_size(reference_block)
            assert cursor + block_size <= seg.numel(), \
                "Invalid block segment, received payload is shorter than block bitmap requires"
            flat_block = seg[cursor:cursor + block_size]
            received_weight, received_bias = _unflatten_parameter_block(flat_block, reference_block)
            aggregated_weight, aggregated_bias = aggregated_blocks[block_idx]
            block_fisher = fisher_seg[fisher_position].to(
                device=aggregated_weight.device,
                dtype=aggregated_weight.dtype,
            )
            received_weight = received_weight.to(device=aggregated_weight.device, dtype=aggregated_weight.dtype)
            aggregated_weight = aggregated_weight + received_weight * block_fisher
            if aggregated_bias is not None:
                received_bias = received_bias.to(device=aggregated_bias.device, dtype=aggregated_bias.dtype)
                aggregated_bias = aggregated_bias + received_bias * block_fisher
            aggregated_blocks[block_idx] = (aggregated_weight, aggregated_bias)
            block_fisher_sum[block_idx] += block_fisher
            cursor += block_size

        assert cursor == seg.numel(), \
            "Invalid block segment, received payload does not match block bitmap"

    averaged_blocks = []
    for block_idx, aggregated_block in enumerate(aggregated_blocks):
        denominator = block_fisher_sum[block_idx]
        aggregated_weight, aggregated_bias = aggregated_block
        fallback_weight, fallback_bias = target_blocks[block_idx]
        valid = denominator > 0
        safe_denominator = torch.where(
            valid,
            denominator,
            torch.ones_like(denominator),
        )
        averaged_blocks.append((
            torch.where(valid, aggregated_weight / safe_denominator, fallback_weight),
            None if aggregated_bias is None else torch.where(
                valid,
                aggregated_bias / safe_denominator.to(
                    device=aggregated_bias.device,
                    dtype=aggregated_bias.dtype,
                ),
                fallback_bias,
            ),
        ))
    return averaged_blocks
