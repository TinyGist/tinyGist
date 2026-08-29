from collections import OrderedDict

import numpy as np
import torch

from src.models.definitions import (
    block_refinement_enabled,
    canonical_block_refinement_config,
    fixed_chunk_slices,
)
from src.models.parameter_vector import FederatedModelMixin


def build_block_layer_keys(
        model,
        parameter_scope="all",
        conv_mode="kernel",
        channel_length=0,
        include_other_blocks=False,
        bn_mode="affine",
        block_refinement=None,
) -> list[tuple]:
    """Return one layer key per primitive block in ParameterBlockSet.all_blocks order."""
    block_refinement = canonical_block_refinement_config(block_refinement)
    bias_mode = block_refinement["bias"]
    if conv_mode == "layer":
        layer_groups = FederatedModelMixin._layer_tensor_groups(
            model.parameter_modules(parameter_scope),
            bn_mode=bn_mode,
        )
        return [
            ("layer", layer_index)
            for layer_index in range(len(layer_groups))
        ]

    module_name_by_id = {
        id(module): name or "<root>"
        for name, module in model.named_modules()
    }
    conv_keys = []
    linear_keys = []
    bn_keys = []
    other_keys = []
    seen = set()

    def module_name(module):
        return module_name_by_id.get(id(module), module.__class__.__name__)

    def append_linear_blocks(module):
        weight_id = id(module.weight)
        if weight_id in seen:
            return
        seen.add(weight_id)
        if module.bias is not None:
            seen.add(id(module.bias))
        layer_key = ("linear", module_name(module), id(module))
        if block_refinement_enabled(block_refinement, "linear"):
            input_slices = fixed_chunk_slices(
                module.in_features,
                block_refinement["linear_chunk_size"],
            )
            for _ in range(module.out_features):
                for _ in input_slices:
                    linear_keys.append(layer_key)
                if module.bias is not None and bias_mode == "separate":
                    linear_keys.append(layer_key)
            return
        for _ in range(module.out_features):
            linear_keys.append(layer_key)
            if module.bias is not None and bias_mode == "separate":
                linear_keys.append(layer_key)

    def append_conv_blocks(module):
        weight_id = id(module.weight)
        if weight_id in seen:
            return
        seen.add(weight_id)
        if module.bias is not None:
            seen.add(id(module.bias))
        layer_key = ("conv", module_name(module), id(module))

        if conv_mode == "channel":
            input_channel_slices = FederatedModelMixin._input_channel_slices(
                module.weight.size(1),
                channel_length,
            )
            for _ in range(module.out_channels):
                for _ in input_channel_slices:
                    conv_keys.append(layer_key)
                if module.bias is not None and bias_mode == "separate":
                    conv_keys.append(layer_key)
            return

        if FederatedModelMixin._is_pointwise_conv(module):
            if block_refinement_enabled(block_refinement, "pointwise"):
                input_slices = fixed_chunk_slices(
                    module.in_channels,
                    block_refinement["pointwise_chunk_size"],
                )
                for _ in range(module.out_channels):
                    for _ in input_slices:
                        conv_keys.append(layer_key)
                    if module.bias is not None and bias_mode == "separate":
                        conv_keys.append(layer_key)
                return
            for _ in range(module.out_channels):
                conv_keys.append(layer_key)
                if module.bias is not None and bias_mode == "separate":
                    conv_keys.append(layer_key)
            return

        for _ in range(module.out_channels):
            for _ in range(module.weight.size(1)):
                conv_keys.append(layer_key)
            if module.bias is not None and bias_mode == "separate":
                conv_keys.append(layer_key)

    def append_batchnorm_block(module):
        tensors = []
        for tensor in FederatedModelMixin._iter_bn_tensors(module, bn_mode):
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            tensors.append(tensor)
        if tensors:
            bn_keys.append(("bn", module_name(module), id(module)))

    def append_stateless_normalization_block(module):
        tensors = []
        for tensor in FederatedModelMixin._iter_stateless_normalization_tensors(module):
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            tensors.append(tensor)
        if tensors:
            other_keys.append(("normalization", module_name(module), id(module)))

    def append_direct_parameter_blocks(module):
        for param in module.parameters(recurse=False):
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            other_keys.append(("other", module_name(module), id(module), param_id))

    def extract_from_module(module):
        if isinstance(module, torch.nn.Linear):
            append_linear_blocks(module)
            return
        if isinstance(module, torch.nn.Conv2d):
            append_conv_blocks(module)
            return
        if FederatedModelMixin._is_batchnorm_module(module):
            append_batchnorm_block(module)
            return
        if FederatedModelMixin._is_stateless_normalization_module(module):
            append_stateless_normalization_block(module)
            return
        for child in module.children():
            extract_from_module(child)
        if include_other_blocks:
            append_direct_parameter_blocks(module)

    for module in FederatedModelMixin._as_module_list(model.parameter_modules(parameter_scope)):
        extract_from_module(module)

    return conv_keys + linear_keys + bn_keys + other_keys


def group_sorted_blocks_within_layers(
        block_scores: np.ndarray,
        layer_keys: list[tuple],
        group_size: int,
        arrangement: str,
        layer_block_indices=None,
) -> list[list[int]]:
    block_scores = np.asarray(block_scores, dtype=np.float64).reshape(-1)
    if group_size <= 0:
        raise ValueError("block group_size must be positive")
    if len(block_scores) != len(layer_keys):
        raise ValueError("Block grouping layer-key count does not match block score count")
    if arrangement not in {"sensitivity_aligned", "sensitivity_diverse"}:
        raise ValueError(f"Unsupported group arrangement [{arrangement}]")

    if layer_block_indices is None:
        blocks_by_layer = OrderedDict()
        for block_idx, layer_key in enumerate(layer_keys):
            blocks_by_layer.setdefault(layer_key, []).append(block_idx)
        layer_block_indices = blocks_by_layer.values()

    groups = []
    for block_indices in layer_block_indices:
        block_indices = np.asarray(block_indices, dtype=np.int64)
        sorted_indices = block_indices[np.lexsort((block_indices, block_scores[block_indices]))]
        if arrangement == "sensitivity_diverse":
            diverse_indices = np.empty_like(sorted_indices)
            diverse_indices[0::2] = sorted_indices[:(len(sorted_indices) + 1) // 2]
            diverse_indices[1::2] = sorted_indices[::-1][:len(sorted_indices) // 2]
            sorted_indices = diverse_indices
        groups.extend(
            sorted_indices[start:start + group_size].tolist()
            for start in range(0, len(sorted_indices), group_size)
        )
    return groups
