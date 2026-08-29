from dataclasses import dataclass

import torch

from .definitions import (
    CONV_BLOCK_MODES,
    block_refinement_enabled,
    canonical_bn_aggregation_mode,
    canonical_block_refinement_config,
    canonical_parameter_scope,
    fixed_chunk_slices,
    input_channel_slices,
)


@dataclass
class ParameterBlockSet:
    """Named parameter blocks used by block-based aggregation.

    ``all_blocks`` defines the wire/bitmap order.
    """

    conv_blocks: list
    linear_blocks: list
    bn_blocks: list
    other_blocks: list

    @property
    def all_blocks(self) -> list:
        return list(self.conv_blocks) + list(self.linear_blocks) + list(self.bn_blocks) + list(self.other_blocks)

    @property
    def split_counts(self) -> tuple[int, int, int, int]:
        return (
            len(self.conv_blocks),
            len(self.linear_blocks),
            len(self.bn_blocks),
            len(self.other_blocks),
        )

    @classmethod
    def from_all_blocks(cls, blocks, split_counts):
        conv_count, linear_count, bn_count, other_count = split_counts
        blocks = list(blocks)
        expected_count = conv_count + linear_count + bn_count + other_count
        assert len(blocks) == expected_count, (
            f"Block count mismatch, expected {expected_count}, got {len(blocks)}"
        )
        linear_start = conv_count
        bn_start = linear_start + linear_count
        other_start = bn_start + bn_count
        return cls(
            conv_blocks=blocks[:linear_start],
            linear_blocks=blocks[linear_start:bn_start],
            bn_blocks=blocks[bn_start:other_start],
            other_blocks=blocks[other_start:],
        )

    def __len__(self):
        return len(self.all_blocks)


class FederatedModelMixin:
    """Common model interface used by federated aggregation methods.

    Models should set module-list attributes, then call
    ``self.finalize_model_setup()`` at the end of ``__init__``:

    - ``self.all_modules``: modules included in full-model aggregation.
    - ``self.conv_modules``: modules included in conv/feature aggregation.
    - ``self.fc_modules``: modules included in fc/classifier aggregation.
    - ``self.custom_modules``: optional custom aggregation modules.

    For dynamic custom selection, override ``custom_parameter_modules()`` or
    define ``select_custom_parameter_modules()`` and return a module/list.
    """

    BATCHNORM_TYPES = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    STATELESS_NORMALIZATION_TYPES = (
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
    )
    BN_RUNNING_STAT_NAMES = ("running_mean", "running_var")
    PARAMETER_COUNT_ATTRS = {
        "all": "model_total_params_num",
        "conv": "model_conv_params_num",
        "fc": "model_fc_params_num",
        "custom": "model_custom_params_num",
    }

    @staticmethod
    def canonical_parameter_scope(parameter_scope: str) -> str:
        return canonical_parameter_scope(parameter_scope)

    @staticmethod
    def canonical_bn_mode(bn_mode: str = "affine") -> str:
        return canonical_bn_aggregation_mode(bn_mode)

    @staticmethod
    def _is_batchnorm_module(module) -> bool:
        return isinstance(module, FederatedModelMixin.BATCHNORM_TYPES)

    @staticmethod
    def _is_stateless_normalization_module(module) -> bool:
        return isinstance(module, FederatedModelMixin.STATELESS_NORMALIZATION_TYPES)

    @staticmethod
    def _include_bn_affine(bn_mode: str) -> bool:
        return bn_mode in {"affine", "affine_running_stats"}

    @staticmethod
    def _include_bn_running_stats(bn_mode: str) -> bool:
        return bn_mode == "affine_running_stats"

    @staticmethod
    def _as_module_list(modules):
        if modules is None:
            return []
        if isinstance(modules, torch.nn.Module):
            return [modules]
        return list(modules)

    def _modules_from_first_available_attr(self, *attr_names):
        for attr_name in attr_names:
            if not hasattr(self, attr_name):
                continue
            modules = getattr(self, attr_name)
            if callable(modules) and not isinstance(modules, torch.nn.Module):
                modules = modules()
            return self._as_module_list(modules)
        return None

    def all_parameter_modules(self):
        modules = self._modules_from_first_available_attr(
            "all_modules",
            "parameter_modules_override",
        )
        if modules is not None:
            return modules
        if hasattr(self, "blocks"):
            return list(self.blocks)
        return [self]

    def conv_parameter_modules(self):
        modules = self._modules_from_first_available_attr(
            "conv_modules",
            "feature_modules_override",
        )
        if modules is not None:
            return modules
        if hasattr(self, "blocks") and hasattr(self, "fc_modules_override"):
            fc_ids = {id(module) for module in self.fc_modules_override}
            return [module for module in self.blocks if id(module) not in fc_ids]
        return []

    def fc_parameter_modules(self):
        modules = self._modules_from_first_available_attr(
            "fc_modules",
            "fc_modules_override",
        )
        if modules is not None:
            return modules
        if hasattr(self, "classifier"):
            return [self.classifier]
        return []

    def custom_parameter_modules(self):
        selector = getattr(self, "select_custom_parameter_modules", None)
        if callable(selector):
            return self._as_module_list(selector())

        modules = self._modules_from_first_available_attr(
            "custom_modules",
            "custom_parameter_modules_override",
        )
        if modules is not None:
            return modules
        return []

    def parameter_modules(self, parameter_scope: str = "all"):
        scope = self.canonical_parameter_scope(parameter_scope)
        return {
            "all": self.all_parameter_modules,
            "conv": self.conv_parameter_modules,
            "fc": self.fc_parameter_modules,
            "custom": self.custom_parameter_modules,
        }[scope]()

    @staticmethod
    def _iter_bn_tensors(module, bn_mode: str = "affine"):
        bn_mode = canonical_bn_aggregation_mode(bn_mode)
        if FederatedModelMixin._include_bn_affine(bn_mode):
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    yield param
        if FederatedModelMixin._include_bn_running_stats(bn_mode):
            for buffer_name in FederatedModelMixin.BN_RUNNING_STAT_NAMES:
                buffer = getattr(module, buffer_name, None)
                if isinstance(buffer, torch.Tensor):
                    yield buffer

    @staticmethod
    def _iter_stateless_normalization_tensors(module):
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad:
                yield parameter

    @staticmethod
    def _iter_selected_tensors(modules, bn_mode: str = "affine"):
        bn_mode = canonical_bn_aggregation_mode(bn_mode)
        seen = set()
        for module in FederatedModelMixin._as_module_list(modules):
            for child in module.modules():
                if FederatedModelMixin._is_batchnorm_module(child):
                    tensor_iter = FederatedModelMixin._iter_bn_tensors(child, bn_mode)
                else:
                    tensor_iter = child.parameters(recurse=False)

                for tensor in tensor_iter:
                    tensor_id = id(tensor)
                    if tensor_id in seen:
                        continue
                    seen.add(tensor_id)
                    yield tensor

    @staticmethod
    def _iter_batchnorm_tensors(modules, bn_mode: str = "affine"):
        bn_mode = canonical_bn_aggregation_mode(bn_mode)
        seen = set()
        for module in FederatedModelMixin._as_module_list(modules):
            for child in module.modules():
                if not FederatedModelMixin._is_batchnorm_module(child):
                    continue
                for tensor in FederatedModelMixin._iter_bn_tensors(child, bn_mode):
                    tensor_id = id(tensor)
                    if tensor_id in seen:
                        continue
                    seen.add(tensor_id)
                    yield tensor

    @staticmethod
    def _layer_tensor_groups(modules, bn_mode: str = "affine", trainable_only=False):
        """Group each parameter-owning layer with any following BN tensors."""
        bn_mode = canonical_bn_aggregation_mode(bn_mode)
        groups = []
        seen = set()

        def unseen_tensors(tensors):
            result = []
            for tensor in tensors:
                tensor_id = id(tensor)
                if tensor_id in seen:
                    continue
                seen.add(tensor_id)
                result.append(tensor)
            return result

        def visit(module):
            if FederatedModelMixin._is_batchnorm_module(module):
                bn_tensors = unseen_tensors(
                    FederatedModelMixin._iter_bn_tensors(module, bn_mode)
                )
                if not bn_tensors:
                    return
                if groups:
                    groups[-1].extend(bn_tensors)
                else:
                    groups.append(bn_tensors)
                return
            if FederatedModelMixin._is_stateless_normalization_module(module):
                normalization_tensors = unseen_tensors(
                    FederatedModelMixin._iter_stateless_normalization_tensors(module)
                )
                if not normalization_tensors:
                    return
                if groups:
                    groups[-1].extend(normalization_tensors)
                else:
                    groups.append(normalization_tensors)
                return

            direct_parameters = unseen_tensors(module.parameters(recurse=False))
            if direct_parameters:
                groups.append(direct_parameters)
            for child in module.children():
                visit(child)

        for module in FederatedModelMixin._as_module_list(modules):
            visit(module)

        if not trainable_only:
            return groups

        trainable_groups = []
        for group in groups:
            trainable_group = [
                tensor
                for tensor in group
                if isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad
            ]
            if not trainable_group:
                raise ValueError(
                    "Every layer unit must contain at least one trainable parameter for scoring"
                )
            trainable_groups.append(trainable_group)
        return trainable_groups

    @staticmethod
    def _flatten_tensors(tensors) -> torch.Tensor:
        params = [
            tensor.detach().clone().reshape(-1)
            for tensor in tensors
        ]
        if not params:
            return torch.empty(0)
        return torch.cat(params)

    @staticmethod
    def _flatten_modules(modules, bn_mode: str = "affine") -> torch.Tensor:
        return FederatedModelMixin._flatten_tensors(
            FederatedModelMixin._iter_selected_tensors(modules, bn_mode)
        )

    @staticmethod
    def _load_tensors(tensors, flat_params: torch.Tensor) -> int:
        beginning_index = 0
        with torch.no_grad():
            for tensor in tensors:
                param_num = tensor.numel()
                tensor.copy_(
                    flat_params[beginning_index:beginning_index + param_num]
                    .view_as(tensor)
                    .to(device=tensor.device, dtype=tensor.dtype)
                )
                beginning_index += param_num
        return beginning_index

    @staticmethod
    def _load_modules(modules, flat_params: torch.Tensor, bn_mode: str = "affine") -> int:
        return FederatedModelMixin._load_tensors(
            FederatedModelMixin._iter_selected_tensors(modules, bn_mode),
            flat_params,
        )

    def get_parameter_vector(self, parameter_scope: str = "all", bn_mode: str = "affine") -> torch.Tensor:
        return self._flatten_modules(self.parameter_modules(parameter_scope), bn_mode=bn_mode)

    def get_batchnorm_vector(self, parameter_scope: str = "all", bn_mode: str = "affine") -> torch.Tensor:
        return self._flatten_tensors(
            self._iter_batchnorm_tensors(self.parameter_modules(parameter_scope), bn_mode)
        )

    def load_batchnorm_vector(
            self,
            bn_params: torch.Tensor,
            parameter_scope: str = "all",
            bn_mode: str = "affine",
    ):
        assert isinstance(bn_params, torch.Tensor), \
            "Error in load_batchnorm_vector, parameters must be in torch.Tensor type"
        expected_num = self.get_batchnorm_vector(parameter_scope, bn_mode=bn_mode).numel()
        assert bn_params.numel() == expected_num, \
            f"Error, expected {expected_num} BN parameters for [{parameter_scope}], got {bn_params.numel()}"
        loaded_num = self._load_tensors(
            self._iter_batchnorm_tensors(self.parameter_modules(parameter_scope), bn_mode),
            bn_params,
        )
        assert loaded_num == expected_num, \
            f"Error, the number of loaded BN parameters for [{parameter_scope}] does not match"

    @staticmethod
    def _append_direct_parameter_blocks(
        module,
        blocks,
        seen,
        tensor_getter=None,
    ):
        if tensor_getter is None:
            tensor_getter = lambda param: param.detach().clone()
        for param in module.parameters(recurse=False):
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            blocks.append((tensor_getter(param), None))

    @staticmethod
    def _append_batchnorm_blocks(
        module,
        blocks,
        seen,
        tensor_getter=None,
        bn_mode="affine",
    ):
        if tensor_getter is None:
            tensor_getter = lambda tensor: tensor.detach().clone()
        tensors = []
        for tensor in FederatedModelMixin._iter_bn_tensors(module, bn_mode):
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            tensors.append(tensor)
        if not tensors:
            return
        flat_block = torch.cat([
            tensor_getter(tensor).reshape(-1)
            for tensor in tensors
        ])
        blocks.append((flat_block, None))

    @staticmethod
    def _append_stateless_normalization_block(
            module,
            blocks,
            seen,
            tensor_getter=None,
    ):
        if tensor_getter is None:
            tensor_getter = lambda tensor: tensor.detach().clone()
        tensors = []
        for tensor in FederatedModelMixin._iter_stateless_normalization_tensors(module):
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            tensors.append(tensor)
        if not tensors:
            return
        flat_block = torch.cat([
            tensor_getter(tensor).reshape(-1)
            for tensor in tensors
        ])
        blocks.append((flat_block, None))

    @staticmethod
    def _input_channel_slices(input_channel_count: int, channel_length=None):
        return input_channel_slices(input_channel_count, channel_length)

    @staticmethod
    def _is_pointwise_conv(module) -> bool:
        return (
            isinstance(module, torch.nn.Conv2d)
            and module.kernel_size == (1, 1)
            and module.groups == 1
        )

    @staticmethod
    def _extract_parameter_blocks(
        modules,
        conv_mode="kernel",
        target_layer_types=None,
        channel_length=None,
        tensor_getter=None,
        include_other_blocks=False,
        bn_mode="affine",
        block_refinement=None,
        trainable_only=False,
    ):
        """Return named blocks in canonical block order."""
        if conv_mode not in CONV_BLOCK_MODES:
            raise ValueError(f"Unknown conv block mode: {conv_mode}")
        bn_mode = canonical_bn_aggregation_mode(bn_mode)
        block_refinement = canonical_block_refinement_config(block_refinement)
        bias_mode = block_refinement["bias"]
        if tensor_getter is None:
            tensor_getter = lambda param: param.detach().clone()
        if conv_mode == "layer":
            if block_refinement["enabled"]:
                raise ValueError(
                    "Layer blocks are indivisible and do not support refinement"
                )
            if channel_length not in (None, 0):
                raise ValueError(
                    "Layer blocks do not use a channel length"
                )
            layer_blocks = [
                (
                    torch.cat([
                        tensor_getter(tensor).reshape(-1)
                        for tensor in tensors
                    ]),
                    None,
                )
                for tensors in FederatedModelMixin._layer_tensor_groups(
                    modules,
                    bn_mode=bn_mode,
                    trainable_only=trainable_only,
                )
            ]
            return ParameterBlockSet([], [], [], layer_blocks)

        if target_layer_types is None:
            target_layer_types = {"conv", "linear"}
        else:
            target_layer_types = set(target_layer_types)
        unsupported = target_layer_types - {"conv", "linear"}
        if unsupported:
            raise ValueError(f"Unsupported target layer types: {sorted(unsupported)}")

        conv_blocks = []
        linear_blocks = []
        bn_blocks = []
        other_blocks = []
        seen = set()

        def block_bias_for_output(bias, output_index):
            if bias is None or bias_mode != "each_chunk":
                return None
            return bias[output_index]

        def append_separate_bias_block(blocks, bias, output_index):
            if bias is not None and bias_mode == "separate":
                blocks.append((bias[output_index].reshape(()), None))

        def append_linear_blocks(module):
            weight_id = id(module.weight)
            if weight_id in seen:
                return
            seen.add(weight_id)
            weight = tensor_getter(module.weight)
            bias = None
            if module.bias is not None and id(module.bias) not in seen:
                seen.add(id(module.bias))
                bias = tensor_getter(module.bias)
            if block_refinement_enabled(block_refinement, "linear"):
                input_slices = fixed_chunk_slices(
                    module.in_features,
                    block_refinement["linear_chunk_size"],
                )
                for output_index in range(module.out_features):
                    block_bias = block_bias_for_output(bias, output_index)
                    for input_slice in input_slices:
                        linear_blocks.append((weight[output_index, input_slice], block_bias))
                    append_separate_bias_block(linear_blocks, bias, output_index)
                return
            for output_index in range(module.out_features):
                block_bias = block_bias_for_output(bias, output_index)
                linear_blocks.append((weight[output_index], block_bias))
                append_separate_bias_block(linear_blocks, bias, output_index)

        def append_conv_blocks(module):
            weight_id = id(module.weight)
            if weight_id in seen:
                return
            seen.add(weight_id)
            weight = tensor_getter(module.weight)
            bias = None
            if module.bias is not None and id(module.bias) not in seen:
                seen.add(id(module.bias))
                bias = tensor_getter(module.bias)

            if conv_mode == "channel":
                input_channel_slices = FederatedModelMixin._input_channel_slices(
                    weight.size(1),
                    channel_length,
                )
                for output_index in range(module.out_channels):
                    block_bias = block_bias_for_output(bias, output_index)
                    for input_channel_slice in input_channel_slices:
                        conv_blocks.append((weight[output_index, input_channel_slice], block_bias))
                    append_separate_bias_block(conv_blocks, bias, output_index)
                return

            # Pointwise kernels are transmitted as one output-channel block.
            if FederatedModelMixin._is_pointwise_conv(module):
                if block_refinement_enabled(block_refinement, "pointwise"):
                    input_slices = fixed_chunk_slices(
                        module.in_channels,
                        block_refinement["pointwise_chunk_size"],
                    )
                    for output_index in range(module.out_channels):
                        block_bias = block_bias_for_output(bias, output_index)
                        for input_slice in input_slices:
                            conv_blocks.append((weight[output_index, input_slice], block_bias))
                        append_separate_bias_block(conv_blocks, bias, output_index)
                    return
                for output_index in range(module.out_channels):
                    block_bias = block_bias_for_output(bias, output_index)
                    conv_blocks.append((weight[output_index], block_bias))
                    append_separate_bias_block(conv_blocks, bias, output_index)
                return

            for output_index in range(module.out_channels):
                block_bias = block_bias_for_output(bias, output_index)
                for input_index in range(weight.size(1)):
                    conv_blocks.append(
                        (weight[output_index, input_index], block_bias)
                    )
                append_separate_bias_block(conv_blocks, bias, output_index)

        def extract_from_module(module):
            if isinstance(module, torch.nn.Linear) and "linear" in target_layer_types:
                append_linear_blocks(module)
                return
            if isinstance(module, torch.nn.Conv2d) and "conv" in target_layer_types:
                append_conv_blocks(module)
                return
            if FederatedModelMixin._is_batchnorm_module(module):
                FederatedModelMixin._append_batchnorm_blocks(
                    module,
                    bn_blocks,
                    seen,
                    tensor_getter=tensor_getter,
                    bn_mode=bn_mode,
                )
                return
            if FederatedModelMixin._is_stateless_normalization_module(module):
                FederatedModelMixin._append_stateless_normalization_block(
                    module,
                    other_blocks,
                    seen,
                    tensor_getter=tensor_getter,
                )
                return
            for child in module.children():
                extract_from_module(child)
            if include_other_blocks:
                FederatedModelMixin._append_direct_parameter_blocks(
                    module,
                    other_blocks,
                    seen,
                    tensor_getter=tensor_getter,
                )

        for module in FederatedModelMixin._as_module_list(modules):
            extract_from_module(module)
        return ParameterBlockSet(
            conv_blocks=conv_blocks,
            linear_blocks=linear_blocks,
            bn_blocks=bn_blocks,
            other_blocks=other_blocks,
        )

    def get_parameter_blocks(
            self,
            parameter_scope: str = "all",
            conv_mode="kernel",
            channel_length=None,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            trainable_only=False,
    ):
        """Extract blocks in the canonical order used by block bitmaps."""
        scope = self.canonical_parameter_scope(parameter_scope)
        return self._extract_parameter_blocks(
            self.parameter_modules(scope),
            conv_mode=conv_mode,
            channel_length=channel_length,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            block_refinement=block_refinement,
            trainable_only=trainable_only,
        )

    @staticmethod
    def _ensure_block_accumulator(accumulators, counts, tensors_by_id, tensor, value):
        tensor_id = id(tensor)
        if tensor_id in accumulators:
            return
        accumulator_device = value.device
        accumulators[tensor_id] = torch.zeros_like(
            tensor.detach(),
            device=accumulator_device,
            dtype=torch.float32,
        )
        counts[tensor_id] = torch.zeros_like(
            tensor.detach(),
            device=accumulator_device,
            dtype=torch.float32,
        )
        tensors_by_id[tensor_id] = tensor

    @staticmethod
    def _accumulate_parameter_block_value(accumulators, counts, tensors_by_id, param, index, value):
        param_id = id(param)
        FederatedModelMixin._ensure_block_accumulator(
            accumulators,
            counts,
            tensors_by_id,
            param,
            value,
        )
        value = value.detach().to(
            device=accumulators[param_id].device,
            dtype=accumulators[param_id].dtype,
        )
        target_shape = accumulators[param_id][index].shape
        accumulators[param_id][index] = accumulators[param_id][index] + value.reshape(target_shape)
        counts[param_id][index] = counts[param_id][index] + 1

    @staticmethod
    def _load_parameter_blocks(
            modules,
            parameter_blocks,
            conv_mode="kernel",
            target_layer_types=None,
            channel_length=None,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
    ):
        if conv_mode not in CONV_BLOCK_MODES:
            raise ValueError(f"Unknown conv block mode: {conv_mode}")
        bn_mode = canonical_bn_aggregation_mode(bn_mode)
        block_refinement = canonical_block_refinement_config(block_refinement)
        bias_mode = block_refinement["bias"]

        if target_layer_types is None:
            target_layer_types = {"conv", "linear"}
        else:
            target_layer_types = set(target_layer_types)

        block_set = FederatedModelMixin._ensure_parameter_block_set(parameter_blocks)
        conv_blocks = list(block_set.conv_blocks)
        linear_blocks = list(block_set.linear_blocks)
        bn_blocks = list(block_set.bn_blocks)
        other_blocks = list(block_set.other_blocks)
        if conv_mode == "layer":
            assert not conv_blocks and not linear_blocks and not bn_blocks, (
                "Layer blocks must use the other_blocks storage partition"
            )
            layer_groups = FederatedModelMixin._layer_tensor_groups(
                modules,
                bn_mode=bn_mode,
            )
            assert len(layer_groups) == len(other_blocks), (
                "Error, the number of loaded layer blocks does not match the model layout"
            )
            with torch.no_grad():
                for tensors, (flat_block, block_bias) in zip(layer_groups, other_blocks):
                    assert block_bias is None, "Layer blocks must use one flat payload"
                    flat_block = flat_block.reshape(-1)
                    cursor = 0
                    for tensor in tensors:
                        tensor_size = tensor.numel()
                        assert cursor + tensor_size <= flat_block.numel(), (
                            "Error, layer block payload is shorter than its tensors require"
                        )
                        tensor.copy_(
                            flat_block[cursor:cursor + tensor_size]
                            .view_as(tensor)
                            .to(device=tensor.device, dtype=tensor.dtype)
                        )
                        cursor += tensor_size
                    assert cursor == flat_block.numel(), (
                        "Error, layer block payload does not match its tensor sizes"
                    )
            return


        accumulators = {}
        counts = {}
        tensors_by_id = {}
        conv_block_idx = 0
        linear_block_idx = 0
        bn_block_idx = 0
        other_block_idx = 0
        seen = set()

        def take_block(blocks, block_idx, block_name):
            assert block_idx < len(blocks), f"Missing {block_name} block at index {block_idx}"
            return blocks[block_idx], block_idx + 1

        def take_conv_block():
            nonlocal conv_block_idx
            block, conv_block_idx = take_block(conv_blocks, conv_block_idx, "conv")
            return block

        def take_linear_block():
            nonlocal linear_block_idx
            block, linear_block_idx = take_block(linear_blocks, linear_block_idx, "linear")
            return block

        def take_bn_block():
            nonlocal bn_block_idx
            block, bn_block_idx = take_block(bn_blocks, bn_block_idx, "batchnorm")
            return block

        def take_other_block():
            nonlocal other_block_idx
            block, other_block_idx = take_block(other_blocks, other_block_idx, "other")
            return block

        def accumulate_bias_value(bias, output_index, value):
            if bias is None or value is None:
                return
            FederatedModelMixin._accumulate_parameter_block_value(
                accumulators,
                counts,
                tensors_by_id,
                bias,
                output_index,
                value,
            )

        def load_linear_blocks(module):
            weight_id = id(module.weight)
            if weight_id in seen:
                return
            seen.add(weight_id)
            bias = module.bias
            if bias is not None:
                seen.add(id(bias))

            if block_refinement_enabled(block_refinement, "linear"):
                input_slices = fixed_chunk_slices(
                    module.in_features,
                    block_refinement["linear_chunk_size"],
                )
                for output_index in range(module.out_features):
                    for input_slice in input_slices:
                        block_weight, block_bias = take_linear_block()
                        FederatedModelMixin._accumulate_parameter_block_value(
                            accumulators,
                            counts,
                            tensors_by_id,
                            module.weight,
                            (output_index, input_slice),
                            block_weight,
                        )
                        if bias_mode == "each_chunk":
                            accumulate_bias_value(bias, output_index, block_bias)
                    if bias is not None and bias_mode == "separate":
                        block_weight, _ = take_linear_block()
                        accumulate_bias_value(bias, output_index, block_weight)
                return

            for output_index in range(module.out_features):
                block_weight, block_bias = take_linear_block()
                FederatedModelMixin._accumulate_parameter_block_value(
                    accumulators,
                    counts,
                    tensors_by_id,
                    module.weight,
                    (output_index, slice(None)),
                    block_weight,
                )
                if bias is not None and block_bias is not None:
                    accumulate_bias_value(bias, output_index, block_bias)
                if bias is not None and bias_mode == "separate":
                    block_weight, _ = take_linear_block()
                    accumulate_bias_value(bias, output_index, block_weight)

        def load_conv_blocks(module):
            weight_id = id(module.weight)
            if weight_id in seen:
                return
            seen.add(weight_id)
            bias = module.bias
            if bias is not None:
                seen.add(id(bias))

            if conv_mode == "channel":
                input_channel_slices = FederatedModelMixin._input_channel_slices(
                    module.weight.size(1),
                    channel_length,
                )
                for output_index in range(module.out_channels):
                    for input_channel_slice in input_channel_slices:
                        block_weight, block_bias = take_conv_block()
                        FederatedModelMixin._accumulate_parameter_block_value(
                            accumulators,
                            counts,
                            tensors_by_id,
                            module.weight,
                            (output_index, input_channel_slice, slice(None), slice(None)),
                            block_weight,
                        )
                        if bias_mode == "each_chunk":
                            accumulate_bias_value(bias, output_index, block_bias)
                    if bias is not None and bias_mode == "separate":
                        block_weight, _ = take_conv_block()
                        accumulate_bias_value(bias, output_index, block_weight)
                return

            # Mirror pointwise output-channel blocks from extraction.
            if FederatedModelMixin._is_pointwise_conv(module):
                if block_refinement_enabled(block_refinement, "pointwise"):
                    input_slices = fixed_chunk_slices(
                        module.in_channels,
                        block_refinement["pointwise_chunk_size"],
                    )
                    for output_index in range(module.out_channels):
                        for input_slice in input_slices:
                            block_weight, block_bias = take_conv_block()
                            FederatedModelMixin._accumulate_parameter_block_value(
                                accumulators,
                                counts,
                                tensors_by_id,
                                module.weight,
                                (output_index, input_slice, slice(None), slice(None)),
                                block_weight,
                            )
                            if bias_mode == "each_chunk":
                                accumulate_bias_value(bias, output_index, block_bias)
                        if bias is not None and bias_mode == "separate":
                            block_weight, _ = take_conv_block()
                            accumulate_bias_value(bias, output_index, block_weight)
                    return
                for output_index in range(module.out_channels):
                    block_weight, block_bias = take_conv_block()
                    FederatedModelMixin._accumulate_parameter_block_value(
                        accumulators,
                        counts,
                        tensors_by_id,
                        module.weight,
                        (output_index, slice(None), slice(None), slice(None)),
                        block_weight,
                    )
                    if bias is not None and block_bias is not None:
                        accumulate_bias_value(bias, output_index, block_bias)
                    if bias is not None and bias_mode == "separate":
                        block_weight, _ = take_conv_block()
                        accumulate_bias_value(bias, output_index, block_weight)
                return

            for output_index in range(module.out_channels):
                for input_index in range(module.weight.size(1)):
                    block_weight, block_bias = take_conv_block()
                    FederatedModelMixin._accumulate_parameter_block_value(
                        accumulators,
                        counts,
                        tensors_by_id,
                        module.weight,
                        (output_index, input_index, slice(None), slice(None)),
                        block_weight,
                    )
                    if bias is not None and block_bias is not None:
                        accumulate_bias_value(bias, output_index, block_bias)
                if bias is not None and bias_mode == "separate":
                    block_weight, _ = take_conv_block()
                    accumulate_bias_value(bias, output_index, block_weight)

        def load_direct_parameter_blocks(module):
            for param in module.parameters(recurse=False):
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                block_weight, _ = take_other_block()
                FederatedModelMixin._accumulate_parameter_block_value(
                    accumulators,
                    counts,
                    tensors_by_id,
                    param,
                    tuple(slice(None) for _ in param.shape),
                    block_weight,
                )

        def load_batchnorm_blocks(module):
            tensors = []
            for tensor in FederatedModelMixin._iter_bn_tensors(module, bn_mode):
                tensor_id = id(tensor)
                if tensor_id in seen:
                    continue
                seen.add(tensor_id)
                tensors.append(tensor)
            if not tensors:
                return

            block_weight, _ = take_bn_block()
            flat_block = block_weight.reshape(-1)
            cursor = 0
            for tensor in tensors:
                tensor_size = tensor.numel()
                assert cursor + tensor_size <= flat_block.numel(), \
                    "Error, BN block payload is shorter than selected BN tensors require"
                tensor_value = flat_block[cursor:cursor + tensor_size].view_as(tensor)
                FederatedModelMixin._accumulate_parameter_block_value(
                    accumulators,
                    counts,
                    tensors_by_id,
                    tensor,
                    tuple(slice(None) for _ in tensor.shape),
                    tensor_value,
                )
                cursor += tensor_size
            assert cursor == flat_block.numel(), \
                "Error, BN block payload does not match selected BN tensor sizes"

        def load_stateless_normalization_block(module):
            tensors = []
            for tensor in FederatedModelMixin._iter_stateless_normalization_tensors(module):
                tensor_id = id(tensor)
                if tensor_id in seen:
                    continue
                seen.add(tensor_id)
                tensors.append(tensor)
            if not tensors:
                return

            block_weight, _ = take_other_block()
            flat_block = block_weight.reshape(-1)
            cursor = 0
            for tensor in tensors:
                tensor_size = tensor.numel()
                assert cursor + tensor_size <= flat_block.numel(), \
                    "Error, normalization block payload is shorter than its tensors require"
                tensor_value = flat_block[cursor:cursor + tensor_size].view_as(tensor)
                FederatedModelMixin._accumulate_parameter_block_value(
                    accumulators,
                    counts,
                    tensors_by_id,
                    tensor,
                    tuple(slice(None) for _ in tensor.shape),
                    tensor_value,
                )
                cursor += tensor_size
            assert cursor == flat_block.numel(), \
                "Error, normalization block payload does not match its tensor sizes"

        def load_from_module(module):
            if isinstance(module, torch.nn.Linear) and "linear" in target_layer_types:
                load_linear_blocks(module)
                return
            if isinstance(module, torch.nn.Conv2d) and "conv" in target_layer_types:
                load_conv_blocks(module)
                return
            if FederatedModelMixin._is_batchnorm_module(module):
                load_batchnorm_blocks(module)
                return
            if FederatedModelMixin._is_stateless_normalization_module(module):
                load_stateless_normalization_block(module)
                return
            for child in module.children():
                load_from_module(child)
            if include_other_blocks:
                load_direct_parameter_blocks(module)

        for module in FederatedModelMixin._as_module_list(modules):
            load_from_module(module)

        assert conv_block_idx == len(conv_blocks), \
            "Error, the number of loaded conv blocks does not match"
        assert linear_block_idx == len(linear_blocks), \
            "Error, the number of loaded linear blocks does not match"
        assert bn_block_idx == len(bn_blocks), \
            "Error, the number of loaded BN blocks does not match"
        assert other_block_idx == len(other_blocks), \
            "Error, the number of loaded other blocks does not match"

        with torch.no_grad():
            for param_id, param in tensors_by_id.items():
                covered_values = counts[param_id] > 0
                assert torch.all(covered_values), \
                    "Error, selected parameter values were only partially covered by block loading"
                averaged_param = accumulators[param_id] / counts[param_id]
                param.copy_(averaged_param.to(device=param.device, dtype=param.dtype))

    @staticmethod
    def _ensure_parameter_block_set(parameter_blocks):
        if not isinstance(parameter_blocks, ParameterBlockSet):
            raise TypeError("parameter_blocks must be a ParameterBlockSet")
        return parameter_blocks

    def load_parameter_blocks(
            self,
            parameter_blocks,
            parameter_scope: str = "all",
            conv_mode="kernel",
            channel_length=None,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
    ):
        """Load blocks produced in the same order as get_parameter_blocks()."""
        scope = self.canonical_parameter_scope(parameter_scope)
        self._load_parameter_blocks(
            self.parameter_modules(scope),
            parameter_blocks,
            conv_mode=conv_mode,
            channel_length=channel_length,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            block_refinement=block_refinement,
        )

    def parameter_count(self, parameter_scope: str = "all", bn_mode: str = "affine") -> int:
        scope = self.canonical_parameter_scope(parameter_scope)
        bn_mode = self.canonical_bn_mode(bn_mode)
        count_attr = self.PARAMETER_COUNT_ATTRS[scope]
        if bn_mode == "affine" and hasattr(self, count_attr):
            return getattr(self, count_attr)
        return self.get_parameter_vector(scope, bn_mode=bn_mode).numel()

    def load_parameter_vector(
            self,
            model_params: torch.Tensor,
            parameter_scope: str = "all",
            bn_mode: str = "affine",
    ):
        assert isinstance(model_params, torch.Tensor), \
            "Error in load_parameter_vector, parameters must be in torch.Tensor type"
        expected_num = self.parameter_count(parameter_scope, bn_mode=bn_mode)
        assert model_params.numel() == expected_num, \
            f"Error, expected {expected_num} parameters for [{parameter_scope}], got {model_params.numel()}"
        beginning_index = self._load_modules(
            self.parameter_modules(parameter_scope),
            model_params,
            bn_mode=bn_mode,
        )
        assert beginning_index == expected_num, \
            f"Error, the number of loaded parameters for [{parameter_scope}] does not match"

    def finalize_model_setup(self, validate_model_parameters=True, validate_parameter_split=False):
        actual_total_params_num = sum(p.numel() for p in self.parameters())
        self.model_total_params_num = self.get_parameter_vector("all").numel()
        self.model_conv_params_num = self.get_parameter_vector("conv").numel()
        self.model_fc_params_num = self.get_parameter_vector("fc").numel()
        self.model_custom_params_num = self.get_parameter_vector("custom").numel()
        self.model_bn_params_num = self.get_batchnorm_vector("all", bn_mode="affine").numel()
        self.model_bn_running_stats_num = (
            self.get_batchnorm_vector("all", bn_mode="affine_running_stats").numel()
            - self.model_bn_params_num
        )

        if validate_model_parameters:
            assert self.model_total_params_num == actual_total_params_num, \
                "Error in abstracting parameters"
        if validate_parameter_split:
            assert (self.model_fc_params_num + self.model_conv_params_num) == self.model_total_params_num, \
                "Error in abstracting parameters"


ParameterVectorMixin = FederatedModelMixin
