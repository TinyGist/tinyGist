import logging

import torch

from src.fl_methods.definitions import BLOCK_SEGMENT_UNITS
from src.fl_methods.segment_ops import flatten_parameter_block, parameter_block_size
from src.models.definitions import (
    canonical_block_refinement_config,
)
from src.models.parameter_vector import ParameterBlockSet

from src.sim_tools.definitions import (
    canonical_fisher_block_reduce_method,
    canonical_fisher_cal,
    canonical_fisher_granularity,
)
from .definitions import (
    BUFFERED_PARAMETER_SCORE_METHODS,
    DIRECT_PARAMETER_SCORE_METHOD,
    FISHER_DIAGONAL_EMA_ONLINE_METHOD,
    FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
    GRADIENT_ABS_EMA_ONLINE_METHOD,
    GRADIENT_ABS_ROUND_STEP_EMA_ONLINE_METHOD,
    HESSIAN_EMA_PARAMETER_SCORE_METHODS,
    HESSIAN_PARAMETER_SCORE_METHODS,
    HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
    HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    POST_TRAINING_PARAMETER_SCORE_METHODS,
    TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD,
    TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD,
    TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    WEIGHT_ABS_EMA_ONLINE_METHOD,
    canonical_parameter_score_method,
    parameter_score_record_dependencies,
)
from .per_sample_scores import (
    compute_per_sample_gradient_multi_score_sums_loop as per_sample_multi_score_sums_loop,
    compute_per_sample_gradient_multi_score_sums_vmap as per_sample_multi_score_sums_vmap,
)


log = logging.getLogger(__name__)
_GRADIENT_ABS_EMA_METHODS = frozenset({
    GRADIENT_ABS_EMA_ONLINE_METHOD,
    GRADIENT_ABS_ROUND_STEP_EMA_ONLINE_METHOD,
})


class GradientBuffer:
    """EMA buffer for diagonal Fisher-style gradient statistics.

    The buffer stores one tensor per model parameter. Read helpers reuse each
    model's parameter module traversal, so Fisher payloads line up with model
    parameter vectors and block bitmaps.
    """

    MAX_EXACT_HESSIAN_PARAMETERS = 100_000

    def __init__(self, model_dict, beta=0.9):
        self.buffer = {}
        self.beta = float(beta)
        if not 0 <= self.beta < 1:
            raise ValueError("GradientBuffer beta must be in [0, 1)")
        self.model_dict = model_dict
        self._param_name_by_id = {}
        self._score_name_by_id = {}
        self._fisher_initialized = {}
        self._parameter_score_initialized = {}
        self._block_interaction_initialized = {}
        self.parameter_score_buffers = {}
        self.block_interaction_buffers = {}
        self._vmap_fallback_warned = False
        for device_idx, model in self.model_dict.items():
            self.buffer[device_idx] = {}
            self._param_name_by_id[device_idx] = {}
            self._score_name_by_id[device_idx] = {}
            self._fisher_initialized[device_idx] = {}
            for name, param in model.named_parameters():
                self.buffer[device_idx][name] = torch.zeros_like(param.detach())
                self._param_name_by_id[device_idx][id(param)] = name
                self._score_name_by_id[device_idx][id(param)] = name
                self._fisher_initialized[device_idx][name] = False
            for name, buffer in model.named_buffers():
                if buffer.is_floating_point():
                    self._score_name_by_id[device_idx][id(buffer)] = name

    def update(
            self,
            current_trainable_list,
            fisher_cal="square",
            parameter_score_methods=None,
            block_interaction_score_configs=None,
            record_fisher=True,
    ):
        fisher_cal = canonical_fisher_cal(fisher_cal)
        parameter_score_methods = self._canonical_parameter_score_methods(parameter_score_methods)
        block_interaction_score_configs = self._canonical_block_interaction_score_configs(
            block_interaction_score_configs,
        )
        for device_idx in current_trainable_list:
            model = self.model_dict[device_idx]
            for name, param in model.named_parameters():
                if param.grad is None:
                    if WEIGHT_ABS_EMA_ONLINE_METHOD in parameter_score_methods:
                        self._update_parameter_score_tensor(
                            device_idx,
                            WEIGHT_ABS_EMA_ONLINE_METHOD,
                            name,
                            param.detach().abs(),
                        )
                    continue
                grad = param.grad.detach()
                if record_fisher:
                    self._update_fisher_tensor(device_idx, name, grad, fisher_cal)
                self._update_parameter_score_tensors(device_idx, name, param, grad, parameter_score_methods)
            if WEIGHT_ABS_EMA_ONLINE_METHOD in parameter_score_methods:
                self._update_weight_abs_ema_buffers(device_idx)
            self._update_block_interaction_scores(device_idx, block_interaction_score_configs)

    def update_from_loss(
            self,
            device_idx,
            loss,
            fisher_cal="square",
            parameter_score_methods=None,
            block_interaction_score_configs=None,
            record_fisher=True,
    ):
        """Record gradients and exact diagonal-Hessian EMA scores for one mini-batch.

        This path is intentionally separate from update(...): it builds a
        second-order autograd graph, fills param.grad for the optimizer, and
        should only be used when exact diagonal-Hessian EMA scores are requested.
        """
        fisher_cal = canonical_fisher_cal(fisher_cal)
        parameter_score_methods = self._canonical_parameter_score_methods(parameter_score_methods)
        block_interaction_score_configs = self._canonical_block_interaction_score_configs(
            block_interaction_score_configs,
        )
        model = self.model_dict[device_idx]
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return

        parameter_score_method_set = set(parameter_score_methods)
        if HESSIAN_EMA_PARAMETER_SCORE_METHODS & parameter_score_method_set:
            self._guard_exact_hessian_size(device_idx, "online exact diagonal Hessian")
        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        for name, param, grad in zip(names, params, grads):
            if grad is None:
                param.grad = None
                if WEIGHT_ABS_EMA_ONLINE_METHOD in parameter_score_methods:
                    self._update_parameter_score_tensor(
                        device_idx,
                        WEIGHT_ABS_EMA_ONLINE_METHOD,
                        name,
                        param.detach().abs(),
                    )
                continue

            detached_grad = grad.detach()
            param.grad = detached_grad.clone()
            if record_fisher:
                self._update_fisher_tensor(device_idx, name, detached_grad, fisher_cal)
            self._update_parameter_score_tensors(
                device_idx,
                name,
                param,
                detached_grad,
                [
                    method
                    for method in parameter_score_methods
                    if method not in HESSIAN_PARAMETER_SCORE_METHODS
                    and method not in POST_TRAINING_PARAMETER_SCORE_METHODS
                ],
            )

            if HESSIAN_EMA_PARAMETER_SCORE_METHODS & parameter_score_method_set:
                diagonal_hessian = self._diagonal_hessian(grad, param)
            if HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD in parameter_score_method_set:
                hessian_score = (diagonal_hessian * param.detach().square()).abs()
                self._update_parameter_score_tensor(
                    device_idx,
                    HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
                    name,
                    hessian_score,
                )
            if HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD in parameter_score_method_set:
                self._update_parameter_score_tensor(
                    device_idx,
                    HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
                    name,
                    diagonal_hessian.abs(),
                )

        if WEIGHT_ABS_EMA_ONLINE_METHOD in parameter_score_methods:
            self._update_weight_abs_ema_buffers(device_idx)
        self._update_block_interaction_scores(device_idx, block_interaction_score_configs)

    def clear(self, model_idx_list=None):
        if model_idx_list is None:
            model_idx_list = self.buffer.keys()
        for device_idx in model_idx_list:
            for name, fisher_tensor in self.buffer[device_idx].items():
                fisher_tensor.zero_()
                self._fisher_initialized[device_idx][name] = False

    def clear_parameter_scores(self, parameter_score_methods=None, model_idx_list=None):
        if parameter_score_methods is None:
            parameter_score_methods = list(self.parameter_score_buffers.keys())
        else:
            parameter_score_methods = self._canonical_parameter_score_methods(parameter_score_methods)
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()
        for method in parameter_score_methods:
            if method not in self.parameter_score_buffers:
                continue
            for device_idx in model_idx_list:
                if device_idx not in self.parameter_score_buffers[method]:
                    continue
                for name, score_tensor in self.parameter_score_buffers[method][device_idx].items():
                    score_tensor.zero_()
                    self._parameter_score_initialized[method][device_idx][name] = False

    def clear_block_interaction_scores(self, model_idx_list=None):
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()
        for block_buffers in self.block_interaction_buffers.values():
            for device_idx in model_idx_list:
                if device_idx not in block_buffers:
                    continue
                block_buffers[device_idx]["interaction"].zero_()
        for initialized_by_device in self._block_interaction_initialized.values():
            for device_idx in model_idx_list:
                if device_idx in initialized_by_device:
                    initialized_by_device[device_idx] = False

    def get_parameter_score_weights(
            self,
            parameter_score_method,
            model_idx_list=None,
            parameter_scope="all",
            bn_mode="affine",
            target_device=None,
    ) -> dict:
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()
        if parameter_score_method == DIRECT_PARAMETER_SCORE_METHOD:
            return {
                model_idx: self.get_parameter_score_vector(
                    model_idx,
                    parameter_score_method=parameter_score_method,
                    parameter_scope=parameter_scope,
                    bn_mode=bn_mode,
                    target_device=target_device,
                )
                for model_idx in model_idx_list
            }
        self._ensure_parameter_score_buffer(parameter_score_method)
        return {
            model_idx: self.get_parameter_score_vector(
                model_idx,
                parameter_score_method=parameter_score_method,
                parameter_scope=parameter_scope,
                bn_mode=bn_mode,
                target_device=target_device,
            )
            for model_idx in model_idx_list
        }

    def get_parameter_score_vector(
            self,
            device_idx,
            parameter_score_method,
            parameter_scope="all",
            bn_mode="affine",
            target_device=None,
    ) -> torch.Tensor:
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        if parameter_score_method == DIRECT_PARAMETER_SCORE_METHOD:
            vector = (
                self.model_dict[device_idx]
                .get_parameter_vector(parameter_scope, bn_mode=bn_mode)
                .detach()
                .abs()
                .clone()
            )
            return vector.to(target_device) if target_device is not None else vector

        model = self.model_dict[device_idx]
        if not hasattr(model, "parameter_modules") or not hasattr(model, "_iter_selected_tensors"):
            raise ValueError("Parameter score vectors require models to implement parameter tensor helpers.")

        self._ensure_parameter_score_buffer(parameter_score_method)
        tensors = [
            self._parameter_score_tensor(device_idx, parameter_score_method, tensor).reshape(-1)
            for tensor in model._iter_selected_tensors(model.parameter_modules(parameter_scope), bn_mode)
        ]
        if not tensors:
            return torch.empty(0)
        vector = torch.cat(tensors)
        return vector.to(target_device) if target_device is not None else vector

    def get_parameter_score_blocks(
            self,
            device_idx,
            parameter_score_method,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> list:
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        model = self.model_dict[device_idx]
        if parameter_score_method == DIRECT_PARAMETER_SCORE_METHOD:
            if not hasattr(model, "get_parameter_blocks"):
                raise ValueError("Block parameter scores require models to implement get_parameter_blocks(...).")
            if target_device is None:
                block_set = model.get_parameter_blocks(
                    parameter_scope,
                    conv_mode=conv_mode,
                    channel_length=channel_length,
                    include_other_blocks=include_other_blocks,
                    bn_mode=bn_mode,
                    block_refinement=block_refinement,
                    trainable_only=conv_mode == "layer",
                )
            else:
                tensor_cache = {}

                def target_tensor(tensor):
                    tensor_id = id(tensor)
                    if tensor_id not in tensor_cache:
                        tensor_cache[tensor_id] = tensor.detach().to(
                            device=target_device,
                            copy=True,
                        )
                    return tensor_cache[tensor_id]

                block_set = model._extract_parameter_blocks(
                    model.parameter_modules(parameter_scope),
                    conv_mode=conv_mode,
                    channel_length=channel_length,
                    tensor_getter=target_tensor,
                    include_other_blocks=include_other_blocks,
                    bn_mode=bn_mode,
                    block_refinement=block_refinement,
                    trainable_only=conv_mode == "layer",
                )
            if not isinstance(block_set, ParameterBlockSet):
                raise TypeError("get_parameter_blocks(...) must return ParameterBlockSet")
            return block_set.all_blocks

        if not hasattr(model, "parameter_modules") or not hasattr(model, "_extract_parameter_blocks"):
            raise ValueError("Block parameter scores require models to implement parameter block helpers.")

        self._ensure_parameter_score_buffer(parameter_score_method)
        tensor_cache = {}

        def target_score_tensor(tensor):
            tensor_id = id(tensor)
            if tensor_id not in tensor_cache:
                score_tensor = self._parameter_score_tensor(
                    device_idx,
                    parameter_score_method,
                    tensor,
                )
                if target_device is not None:
                    score_tensor = score_tensor.to(target_device)
                tensor_cache[tensor_id] = score_tensor
            return tensor_cache[tensor_id]

        block_set = model._extract_parameter_blocks(
            model.parameter_modules(parameter_scope),
            conv_mode=conv_mode,
            channel_length=channel_length,
            tensor_getter=target_score_tensor,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            block_refinement=block_refinement,
            trainable_only=conv_mode == "layer",
        )
        return block_set.all_blocks

    def get_parameter_score_blocks_dict(
            self,
            parameter_score_method,
            model_idx_list=None,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> dict:
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        return {
            model_idx: self.get_parameter_score_blocks(
                model_idx,
                parameter_score_method=parameter_score_method,
                parameter_scope=parameter_scope,
                conv_mode=conv_mode,
                channel_length=channel_length,
                include_other_blocks=include_other_blocks,
                bn_mode=bn_mode,
                block_refinement=block_refinement,
                target_device=target_device,
            )
            for model_idx in model_idx_list
        }

    def get_block_interaction_scores_dict(
            self,
            model_idx_list=None,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
    ) -> dict:
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()
        config = self._canonical_block_interaction_score_config(
            parameter_scope=parameter_scope,
            conv_mode=conv_mode,
            channel_length=channel_length,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            block_refinement=block_refinement,
        )
        self._ensure_block_interaction_buffer(config)
        key = self._block_interaction_score_key(config)
        return {
            model_idx: self._block_interaction_scores_from_buffer(
                self.block_interaction_buffers[key][model_idx],
            )
            for model_idx in model_idx_list
        }

    @classmethod
    def _canonical_parameter_score_methods(cls, parameter_score_methods) -> list[str]:
        if parameter_score_methods is None:
            return []
        if isinstance(parameter_score_methods, str):
            parameter_score_methods = [parameter_score_methods]
        canonical_methods = []
        for method in parameter_score_methods:
            canonical_method = canonical_parameter_score_method(method)
            for dependency in parameter_score_record_dependencies(canonical_method):
                if dependency not in BUFFERED_PARAMETER_SCORE_METHODS:
                    raise ValueError(f"Parameter score method [{method}] is not buffered")
                if dependency not in canonical_methods:
                    canonical_methods.append(dependency)
        return canonical_methods

    def _ensure_parameter_score_buffer(self, parameter_score_method):
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        if parameter_score_method == DIRECT_PARAMETER_SCORE_METHOD:
            return
        dependencies = parameter_score_record_dependencies(parameter_score_method)
        if dependencies != [parameter_score_method]:
            for dependency in dependencies:
                self._ensure_parameter_score_buffer(dependency)
            return
        if parameter_score_method not in BUFFERED_PARAMETER_SCORE_METHODS:
            raise ValueError(f"Parameter score method [{parameter_score_method}] is not buffered")
        if parameter_score_method in self.parameter_score_buffers:
            return

        self.parameter_score_buffers[parameter_score_method] = {}
        self._parameter_score_initialized[parameter_score_method] = {}
        for device_idx, model in self.model_dict.items():
            self.parameter_score_buffers[parameter_score_method][device_idx] = {
                name: torch.zeros_like(tensor.detach())
                for name, tensor in self._iter_score_tensors(model)
            }
            self._parameter_score_initialized[parameter_score_method][device_idx] = {
                name: False
                for name in self.parameter_score_buffers[parameter_score_method][device_idx].keys()
            }

    @staticmethod
    def _canonical_block_interaction_score_config(
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
    ):
        if channel_length is None:
            channel_length = 0
        return {
            "parameter_scope": str(parameter_scope),
            "conv_mode": str(conv_mode),
            "channel_length": int(channel_length),
            "include_other_blocks": bool(include_other_blocks),
            "bn_mode": str(bn_mode),
            "block_refinement": canonical_block_refinement_config(block_refinement),
        }

    @classmethod
    def _canonical_block_interaction_score_configs(cls, configs):
        if configs is None:
            return []
        if isinstance(configs, dict):
            configs = [configs]
        canonical_configs = []
        seen = set()
        for config in configs:
            canonical_config = cls._canonical_block_interaction_score_config(**config)
            key = cls._block_interaction_score_key(canonical_config)
            if key in seen:
                continue
            seen.add(key)
            canonical_configs.append(canonical_config)
        return canonical_configs

    @staticmethod
    def _block_interaction_score_key(config):
        refinement = canonical_block_refinement_config(config.get("block_refinement"))
        return (
            config["parameter_scope"],
            config["conv_mode"],
            config["channel_length"],
            config["include_other_blocks"],
            config["bn_mode"],
            refinement["enabled"],
            refinement["targets"],
            refinement["linear_chunk_size"],
            refinement["pointwise_chunk_size"],
            refinement["bias"],
        )

    def _ensure_block_interaction_buffer(self, config):
        key = self._block_interaction_score_key(config)
        if key in self.block_interaction_buffers:
            return

        self.block_interaction_buffers[key] = {}
        self._block_interaction_initialized[key] = {}
        expected_block_count = None
        for device_idx, model in self.model_dict.items():
            if not hasattr(model, "get_parameter_blocks"):
                raise ValueError("Block interaction scoring requires models to implement get_parameter_blocks(...).")
            block_set = model.get_parameter_blocks(
                config["parameter_scope"],
                conv_mode=config["conv_mode"],
                channel_length=config["channel_length"],
                include_other_blocks=config["include_other_blocks"],
                bn_mode=config["bn_mode"],
                block_refinement=config["block_refinement"],
                trainable_only=config["conv_mode"] == "layer",
            )
            if not isinstance(block_set, ParameterBlockSet):
                raise TypeError("get_parameter_blocks(...) must return ParameterBlockSet")
            block_sizes = torch.as_tensor(
                [parameter_block_size(block) for block in block_set.all_blocks],
                dtype=torch.float32,
                device=block_set.all_blocks[0][0].device,
            )
            if expected_block_count is None:
                expected_block_count = block_sizes.numel()
            assert block_sizes.numel() == expected_block_count, \
                "Block interaction score count does not match block layout"
            self.block_interaction_buffers[key][device_idx] = {
                "interaction": torch.zeros_like(block_sizes),
                "block_sizes": block_sizes,
            }
            self._block_interaction_initialized[key][device_idx] = False

    def _update_block_interaction_scores(self, device_idx, configs):
        if not configs:
            return
        for config in configs:
            self._ensure_block_interaction_buffer(config)
            key = self._block_interaction_score_key(config)
            target = self.block_interaction_buffers[key][device_idx]
            gradient_blocks = self._current_gradient_blocks(device_idx, config)
            assert len(gradient_blocks) == target["block_sizes"].numel(), \
                "Gradient block count does not match block interaction buffer"

            interaction_values = torch.stack([
                self._block_abs_interaction_score(gradient_block)
                for gradient_block in gradient_blocks
            ]).to(device=target["interaction"].device, dtype=target["interaction"].dtype)
            self._block_interaction_initialized[key][device_idx] = self._update_ema_tensor(
                target["interaction"],
                interaction_values,
                self.beta,
                self._block_interaction_initialized[key][device_idx],
            )

    def _current_gradient_blocks(self, device_idx, config):
        model = self.model_dict[device_idx]
        if not hasattr(model, "parameter_modules") or not hasattr(model, "_extract_parameter_blocks"):
            raise ValueError("Block interaction scoring requires models to implement parameter block helpers.")
        block_set = model._extract_parameter_blocks(
            model.parameter_modules(config["parameter_scope"]),
            conv_mode=config["conv_mode"],
            channel_length=config["channel_length"],
            tensor_getter=lambda tensor: self._current_gradient_tensor(tensor),
            include_other_blocks=config["include_other_blocks"],
            bn_mode=config["bn_mode"],
            block_refinement=config["block_refinement"],
            trainable_only=config["conv_mode"] == "layer",
        )
        return block_set.all_blocks

    @staticmethod
    def _current_gradient_tensor(tensor):
        grad = getattr(tensor, "grad", None)
        if grad is None:
            return torch.zeros_like(tensor.detach(), dtype=torch.float32)
        return grad.detach().to(dtype=torch.float32).clone()

    @staticmethod
    def _block_abs_interaction_score(gradient_block):
        flat_gradient = flatten_parameter_block(gradient_block).to(dtype=torch.float32)
        assert flat_gradient.numel() > 0, "Gradient block must not be empty"
        block_size = flat_gradient.numel()
        if block_size <= 1:
            return flat_gradient.new_zeros(())
        abs_gradient = flat_gradient.abs()
        interaction = (abs_gradient.sum().square() - abs_gradient.square().sum()) / (
            block_size * (block_size - 1)
        )
        return torch.clamp(interaction, min=0.0)

    @staticmethod
    def _block_interaction_scores_from_buffer(block_buffer):
        return block_buffer["interaction"].detach().cpu().clone()

    @staticmethod
    def _iter_score_tensors(model):
        for name, param in model.named_parameters():
            yield name, param
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point():
                yield name, buffer

    def _update_weight_abs_ema_buffers(self, device_idx):
        model = self.model_dict[device_idx]
        for name, buffer in model.named_buffers():
            if not buffer.is_floating_point():
                continue
            self._update_parameter_score_tensor(
                device_idx,
                WEIGHT_ABS_EMA_ONLINE_METHOD,
                name,
                buffer.detach().abs(),
            )

    def _update_fisher_tensor(self, device_idx, name, grad, fisher_cal):
        fisher_value = grad.square() if fisher_cal == "square" else grad.abs()
        self._fisher_initialized[device_idx][name] = self._update_ema_tensor(
            self.buffer[device_idx][name],
            fisher_value,
            self.beta,
            self._fisher_initialized[device_idx][name],
        )

    def _update_parameter_score_tensors(self, device_idx, name, param, grad, parameter_score_methods):
        gradient_abs = None
        for parameter_score_method in parameter_score_methods:
            if (
                    parameter_score_method in HESSIAN_PARAMETER_SCORE_METHODS
                    or parameter_score_method in POST_TRAINING_PARAMETER_SCORE_METHODS
            ):
                continue
            if parameter_score_method in _GRADIENT_ABS_EMA_METHODS:
                gradient_abs = grad.abs() if gradient_abs is None else gradient_abs
                score_value = gradient_abs
            else:
                score_value = self._parameter_score_value(parameter_score_method, param, grad)
            self._update_parameter_score_tensor(device_idx, parameter_score_method, name, score_value)

    def _update_parameter_score_tensor(self, device_idx, parameter_score_method, name, score_value):
        self._ensure_parameter_score_buffer(parameter_score_method)
        target = self.parameter_score_buffers[parameter_score_method][device_idx][name]
        self._parameter_score_initialized[parameter_score_method][device_idx][name] = self._update_ema_tensor(
            target,
            score_value,
            self.beta,
            self._parameter_score_initialized[parameter_score_method][device_idx][name],
        )

    @staticmethod
    def _update_ema_tensor(target, value, beta, initialized):
        value = value.to(device=target.device, dtype=target.dtype)
        if initialized:
            target.mul_(beta).add_(value, alpha=1 - beta)
        else:
            target.copy_(value)
        return True

    @staticmethod
    def _parameter_score_value(parameter_score_method, param, grad):
        theta = param.detach()
        if parameter_score_method == WEIGHT_ABS_EMA_ONLINE_METHOD:
            return theta.abs()
        if parameter_score_method in _GRADIENT_ABS_EMA_METHODS:
            return grad.abs()
        if parameter_score_method == FISHER_DIAGONAL_EMA_ONLINE_METHOD:
            return grad.square()
        if parameter_score_method == TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD:
            return (grad * theta).abs()
        if parameter_score_method == TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD:
            return grad.square() * theta.square()
        raise ValueError(f"Invalid buffered parameter_score_method [{parameter_score_method}]")

    def compute_hessian_score_tensors_from_loss(self, device_idx, loss) -> dict[str, torch.Tensor]:
        """Return abs(diagonal(H)) for the supplied scalar loss without EMA."""
        self._guard_exact_hessian_size(device_idx, "post-training exact diagonal Hessian")
        model = self.model_dict[device_idx]
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return {}

        if loss.ndim != 0:
            loss = loss.mean()
        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        score_tensors = {}
        for param_idx, (name, param, grad) in enumerate(zip(names, params, grads)):
            if grad is None:
                score_tensors[name] = torch.zeros_like(param.detach(), dtype=torch.float32)
                continue
            diagonal_hessian = self._diagonal_hessian(
                grad,
                param,
                retain_graph_after_param=param_idx < len(params) - 1,
            )
            score_tensors[name] = diagonal_hessian.abs().detach().to(dtype=torch.float32)
        return score_tensors

    def compute_gradient_sum_tensors(
            self,
            device_idx,
            data,
            labels,
            loss_function,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Return batch-size weighted gradient sums for one local-train batch."""
        batch_sample_count = int(data.shape[0])
        if batch_sample_count == 0:
            return {}, 0

        model = self.model_dict[device_idx]
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return {}, batch_sample_count

        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        outputs = model(data)
        loss = loss_function(outputs, labels)
        if loss.ndim != 0:
            loss = loss.mean()
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            allow_unused=True,
        )

        gradient_sums = {}
        for name, param, grad in zip(names, params, grads):
            if grad is None:
                gradient_sums[name] = torch.zeros_like(param.detach(), dtype=torch.float32)
            else:
                gradient_sums[name] = (
                    grad.detach().to(dtype=torch.float32) * batch_sample_count
                )
        return gradient_sums, batch_sample_count

    def compute_gradient_signal_preservation_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
            signal_tensors: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Return batch-size weighted H_batch @ signal sums for one batch."""
        batch_sample_count = int(data.shape[0])
        if batch_sample_count == 0:
            return {}, 0

        model = self.model_dict[device_idx]
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return {}, batch_sample_count

        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        outputs = model(data)
        loss = loss_function(outputs, labels)
        if loss.ndim != 0:
            loss = loss.mean()
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        grad_signal_dot = None
        for name, grad, param in zip(names, grads, params):
            if grad is None or not grad.requires_grad:
                continue
            signal = signal_tensors.get(name)
            if signal is None:
                signal = torch.zeros_like(param.detach(), dtype=torch.float32)
            signal = signal.detach().to(device=param.device, dtype=param.dtype)
            term = (grad * signal).sum()
            grad_signal_dot = term if grad_signal_dot is None else grad_signal_dot + term

        if grad_signal_dot is None or not grad_signal_dot.requires_grad:
            hessian_signal_products = [None] * len(params)
        else:
            hessian_signal_products = torch.autograd.grad(
                grad_signal_dot,
                params,
                retain_graph=False,
                allow_unused=True,
            )

        score_sums = {}
        for name, param, hessian_signal_product in zip(names, params, hessian_signal_products):
            if hessian_signal_product is None:
                score = torch.zeros_like(param.detach(), dtype=torch.float32)
            else:
                score = hessian_signal_product.detach().to(dtype=torch.float32)
            score_sums[name] = score * batch_sample_count
        return score_sums, batch_sample_count

    def compute_empirical_fisher_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Return per-parameter sums of per-sample gradient squares.

        The caller divides the returned tensors by the returned sample count
        before writing them into the score buffer. The vectorized path is used
        first; losses with Python-side tensor scalar extraction can fall back to
        an ordinary per-sample autograd loop.
        """
        return self._compute_per_sample_gradient_score_sums(
            device_idx,
            data,
            labels,
            loss_function,
            score_transform="square",
        )

    def compute_per_sample_gradient_abs_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Return per-parameter sums of per-sample absolute gradients."""
        return self._compute_per_sample_gradient_score_sums(
            device_idx,
            data,
            labels,
            loss_function,
            score_transform="abs",
        )

    def compute_per_sample_gradient_abs_and_square_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
        """Return absolute-gradient and gradient-square sums from one pass."""
        return self._compute_per_sample_gradient_multi_score_sums(
            device_idx,
            data,
            labels,
            loss_function,
            score_transforms=("abs", "square"),
        )

    def _compute_per_sample_gradient_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
            score_transform,
    ) -> tuple[dict[str, torch.Tensor], int]:
        multi_score_sums, sample_count = (
            self._compute_per_sample_gradient_multi_score_sums(
                device_idx,
                data,
                labels,
                loss_function,
                score_transforms=(score_transform,),
            )
        )
        return multi_score_sums.get(score_transform, {}), sample_count

    def _compute_per_sample_gradient_multi_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
            score_transforms,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
        sample_count = int(data.shape[0])
        score_transforms = tuple(dict.fromkeys(score_transforms))
        if sample_count == 0:
            return {
                score_transform: {}
                for score_transform in score_transforms
            }, 0
        try:
            return per_sample_multi_score_sums_vmap(
                self.model_dict[device_idx],
                data,
                labels,
                loss_function,
                score_transforms,
            ), sample_count
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                raise
            if not self._vmap_fallback_warned:
                log.warning(
                    "Per-sample gradient vmap is unavailable; using the slower sample loop: %s",
                    exc,
                )
                self._vmap_fallback_warned = True
            return per_sample_multi_score_sums_loop(
                self.model_dict[device_idx],
                data,
                labels,
                loss_function,
                score_transforms,
            ), sample_count



    def compute_hutchinson_diagonal_score_sums(
            self,
            device_idx,
            data,
            labels,
            loss_function,
            hutchinson_z_time=5,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Return signed Hutchinson diagonal sums for one local-train batch.

        The caller averages these sums over samples and applies abs after
        averaging, matching abs(diag(H)) for the mean local loss.
        """
        batch_sample_count = int(data.shape[0])
        if batch_sample_count == 0:
            return {}, 0

        model = self.model_dict[device_idx]
        outputs = model(data)
        loss = loss_function(outputs, labels)
        if loss.ndim != 0:
            loss = loss.mean()

        diagonal_estimates = self._hutchinson_diagonal_estimate_tensors_from_loss(
            device_idx,
            loss,
            hutchinson_z_time=hutchinson_z_time,
        )
        return {
            name: diagonal_estimate.detach().to(dtype=torch.float32) * batch_sample_count
            for name, diagonal_estimate in diagonal_estimates.items()
        }, batch_sample_count

    def _hutchinson_diagonal_estimate_tensors_from_loss(
            self,
            device_idx,
            loss,
            hutchinson_z_time=5,
    ) -> dict[str, torch.Tensor]:
        """Return signed Hutchinson estimates of diag(H) for a scalar loss."""
        hutchinson_z_time = self._hutchinson_probe_count(hutchinson_z_time)
        if loss.ndim != 0:
            loss = loss.mean()
        model = self.model_dict[device_idx]
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return {}

        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        diag_sums = {
            name: torch.zeros_like(param.detach(), dtype=torch.float32, device=param.device)
            for name, param in named_params
        }

        for z_idx in range(hutchinson_z_time):
            probes = []
            grad_probe_dot = None
            for grad, param in zip(grads, params):
                if grad is None or not grad.requires_grad:
                    probes.append(None)
                    continue
                probe = self._rademacher_like(param)
                probes.append(probe)
                term = (grad * probe).sum()
                grad_probe_dot = term if grad_probe_dot is None else grad_probe_dot + term

            if grad_probe_dot is None or not grad_probe_dot.requires_grad:
                continue
            hvps = torch.autograd.grad(
                grad_probe_dot,
                params,
                retain_graph=z_idx < hutchinson_z_time - 1,
                allow_unused=True,
            )
            for name, probe, hvp in zip(names, probes, hvps):
                if probe is None or hvp is None:
                    continue
                diag_sums[name].add_((probe * hvp).detach().to(dtype=torch.float32))

        score_tensors = {}
        for name, param in named_params:
            diagonal_estimate = diag_sums[name] / hutchinson_z_time
            score_tensors[name] = diagonal_estimate.detach()
        return score_tensors

    def _guard_exact_hessian_size(self, device_idx, operation):
        parameter_count = sum(
            param.numel()
            for param in self.model_dict[device_idx].parameters()
            if param.requires_grad
        )
        if parameter_count > self.MAX_EXACT_HESSIAN_PARAMETERS:
            raise RuntimeError(
                f"{operation} requires one second-order backward per parameter; "
                f"model {device_idx!r} has {parameter_count:,} trainable parameters, "
                f"above the safety limit {self.MAX_EXACT_HESSIAN_PARAMETERS:,}. "
                "Use a Hutchinson diagonal score for this model."
            )

    @staticmethod
    def _hutchinson_probe_count(hutchinson_z_time) -> int:
        probe_count = int(hutchinson_z_time)
        if probe_count <= 0:
            raise ValueError("hutchinson_z_time must be positive")
        return probe_count

    def weight_score_tensors_by_parameter_abs(self, device_idx, score_tensors: dict[str, torch.Tensor]):
        return self._weight_score_tensors_by_parameter(device_idx, score_tensors, parameter_power=1)

    def weight_score_tensors_by_parameter_square(self, device_idx, score_tensors: dict[str, torch.Tensor]):
        return self._weight_score_tensors_by_parameter(device_idx, score_tensors, parameter_power=2)

    def _weight_score_tensors_by_parameter(self, device_idx, score_tensors: dict[str, torch.Tensor], parameter_power):
        model = self.model_dict[device_idx]
        param_by_name = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        weighted_scores = {}
        for name, score_tensor in score_tensors.items():
            param = param_by_name.get(name)
            if param is None:
                continue
            parameter_weight = param.detach().to(dtype=torch.float32).abs()
            if parameter_power == 2:
                parameter_weight = parameter_weight.square()
            weighted_scores[name] = (
                score_tensor.detach().to(device=param.device, dtype=torch.float32)
                * parameter_weight
            ).abs()
        return weighted_scores

    @staticmethod
    def _rademacher_like(tensor):
        return (
            torch.randint(
                low=0,
                high=2,
                size=tensor.shape,
                device=tensor.device,
                dtype=torch.int8,
            )
            .to(dtype=tensor.dtype)
            .mul_(2)
            .sub_(1)
        )

    def set_parameter_score_tensors(self, device_idx, parameter_score_method, score_tensors: dict[str, torch.Tensor]):
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        self._ensure_parameter_score_buffer(parameter_score_method)
        target_tensors = self.parameter_score_buffers[parameter_score_method][device_idx]
        for name, target in target_tensors.items():
            value = score_tensors.get(name)
            if value is None:
                target.zero_()
            else:
                target.copy_(value.to(device=target.device, dtype=target.dtype))
            self._parameter_score_initialized[parameter_score_method][device_idx][name] = True

    def update_parameter_score_tensors(self, device_idx, parameter_score_method, score_tensors: dict[str, torch.Tensor]):
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        self._ensure_parameter_score_buffer(parameter_score_method)
        target_tensors = self.parameter_score_buffers[parameter_score_method][device_idx]
        for name, target in target_tensors.items():
            value = score_tensors.get(name)
            if value is None:
                value = torch.zeros_like(target)
            self._parameter_score_initialized[parameter_score_method][device_idx][name] = self._update_ema_tensor(
                target,
                value,
                self.beta,
                self._parameter_score_initialized[parameter_score_method][device_idx][name],
            )

    @staticmethod
    def _diagonal_hessian(grad, param, retain_graph_after_param=True):
        if not grad.requires_grad:
            return torch.zeros_like(param.detach())

        grad_flat = grad.reshape(-1)
        diagonal = torch.zeros_like(param.detach()).reshape(-1)
        for position in range(grad_flat.numel()):
            retain_graph = retain_graph_after_param or position < grad_flat.numel() - 1
            second_grad = torch.autograd.grad(
                grad_flat[position],
                param,
                retain_graph=retain_graph,
                allow_unused=True,
            )[0]
            if second_grad is not None:
                diagonal[position] = second_grad.reshape(-1)[position].detach()
        return diagonal.view_as(param)

    def _parameter_score_tensor(self, device_idx, parameter_score_method, tensor):
        parameter_score_method = canonical_parameter_score_method(parameter_score_method)
        name = self._score_name_by_id[device_idx].get(id(tensor))
        if isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad and name is not None:
            theta = tensor.detach()
            if parameter_score_method == TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD:
                base = self.parameter_score_buffers[GRADIENT_ABS_EMA_ONLINE_METHOD][device_idx][name]
                return base.detach().clone() * theta.abs()
            if parameter_score_method == FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD:
                base = self.parameter_score_buffers[FISHER_DIAGONAL_EMA_ONLINE_METHOD][device_idx][name]
                return base.detach().clone() * theta.square()
            if parameter_score_method == HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD:
                base = self.parameter_score_buffers[HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD][device_idx][name]
                return base.detach().clone() * theta.square()
            return (
                self.parameter_score_buffers[parameter_score_method][device_idx][name]
                .detach()
                .clone()
            )
        return torch.zeros_like(tensor.detach(), dtype=torch.float32)

    def get_fisher_weights(
            self,
            model_idx_list=None,
            parameter_scope="all",
            segment_unit="parameter",
            channel_length=0,
            fisher_granularity="parameter",
            fisher_block_reduce_method="mean",
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> dict:
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()

        fisher_granularity = canonical_fisher_granularity(fisher_granularity)
        segment_unit = str(segment_unit).strip().replace("-", "_").replace(" ", "_").lower()
        block_refinement = canonical_block_refinement_config(block_refinement)

        if fisher_granularity == "block":
            if segment_unit not in BLOCK_SEGMENT_UNITS:
                return {
                    model_idx: self.get_parameter_fisher_vector(
                        model_idx,
                        parameter_scope=parameter_scope,
                        bn_mode=bn_mode,
                        target_device=target_device,
                    )
                    for model_idx in model_idx_list
                }
            block_reduce_method = canonical_fisher_block_reduce_method(
                fisher_block_reduce_method
            )
            return {
                model_idx: self.get_block_fisher_vector(
                    model_idx,
                    parameter_scope=parameter_scope,
                    conv_mode=segment_unit,
                    channel_length=channel_length,
                    block_reduce_method=block_reduce_method,
                    include_other_blocks=include_other_blocks,
                    bn_mode=bn_mode,
                    block_refinement=block_refinement,
                    target_device=target_device,
                )
                for model_idx in model_idx_list
            }

        if segment_unit in BLOCK_SEGMENT_UNITS:
            return {
                model_idx: self.get_parameter_fisher_blocks(
                    model_idx,
                    parameter_scope=parameter_scope,
                    conv_mode=segment_unit,
                    channel_length=channel_length,
                    include_other_blocks=include_other_blocks,
                    bn_mode=bn_mode,
                    block_refinement=block_refinement,
                    target_device=target_device,
                )
                for model_idx in model_idx_list
            }

        return {
            model_idx: self.get_parameter_fisher_vector(
                model_idx,
                parameter_scope=parameter_scope,
                bn_mode=bn_mode,
                target_device=target_device,
            )
            for model_idx in model_idx_list
        }

    def get_parameter_fisher_vector(
            self,
            device_idx,
            parameter_scope="all",
            bn_mode="affine",
            target_device=None,
    ) -> torch.Tensor:
        model = self.model_dict[device_idx]
        if not hasattr(model, "parameter_modules") or not hasattr(model, "_iter_selected_tensors"):
            raise ValueError("Parameter Fisher requires models to implement parameter tensor helpers.")

        tensors = [
            self._fisher_tensor(device_idx, tensor).reshape(-1)
            for tensor in model._iter_selected_tensors(model.parameter_modules(parameter_scope), bn_mode)
        ]
        if not tensors:
            return torch.empty(0)
        fisher_vector = torch.cat(tensors)
        return fisher_vector.to(target_device) if target_device is not None else fisher_vector

    def get_parameter_fisher_blocks(
            self,
            device_idx,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> list:
        model = self.model_dict[device_idx]
        if not hasattr(model, "parameter_modules") or not hasattr(model, "_extract_parameter_blocks"):
            raise ValueError("Block Fisher requires models to implement parameter block helpers.")

        tensor_cache = {}

        def target_fisher_tensor(tensor):
            tensor_id = id(tensor)
            if tensor_id not in tensor_cache:
                fisher_tensor = self._fisher_tensor(device_idx, tensor)
                if target_device is not None:
                    fisher_tensor = fisher_tensor.to(target_device)
                tensor_cache[tensor_id] = fisher_tensor
            return tensor_cache[tensor_id]

        block_set = model._extract_parameter_blocks(
            model.parameter_modules(parameter_scope),
            conv_mode=conv_mode,
            channel_length=channel_length,
            tensor_getter=target_fisher_tensor,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            block_refinement=block_refinement,
            trainable_only=conv_mode == "layer",
        )
        return block_set.all_blocks

    def get_block_fisher_vector(
            self,
            device_idx,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            block_reduce_method="mean",
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> torch.Tensor:
        block_reduce_method = canonical_fisher_block_reduce_method(block_reduce_method)
        fisher_blocks = self.get_parameter_fisher_blocks(
            device_idx,
            parameter_scope=parameter_scope,
            conv_mode=conv_mode,
            channel_length=channel_length,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            block_refinement=block_refinement,
            target_device=target_device,
        )
        if not fisher_blocks:
            return torch.empty(0, device=target_device)
        return torch.stack([
                self._reduce_fisher_block(block, block_reduce_method)
                for block in fisher_blocks
        ]).to(dtype=torch.float32)

    def _fisher_tensor(self, device_idx, tensor):
        name = self._param_name_by_id[device_idx].get(id(tensor))
        if name is not None:
            return self.buffer[device_idx][name].detach().clone()
        return torch.ones_like(tensor.detach(), dtype=torch.float32)

    @staticmethod
    def _reduce_fisher_block(parameter_block, reduce_method: str) -> torch.Tensor:
        weight, bias = parameter_block
        block_size = parameter_block_size(parameter_block)
        assert block_size > 0, "Parameter block must not be empty"
        weight = weight.to(dtype=torch.float32)
        bias = None if bias is None else bias.to(dtype=torch.float32)
        if reduce_method == "mean":
            fisher_sum = weight.sum()
            if bias is not None:
                fisher_sum = fisher_sum + bias.sum()
            return fisher_sum / block_size
        if reduce_method == "rms":
            fisher_square_sum = torch.square(weight).sum()
            if bias is not None:
                fisher_square_sum = fisher_square_sum + torch.square(bias).sum()
            return torch.sqrt(fisher_square_sum / block_size)
        if reduce_method == "l2":
            fisher_square_sum = torch.square(weight).sum()
            if bias is not None:
                fisher_square_sum = fisher_square_sum + torch.square(bias).sum()
            return torch.sqrt(fisher_square_sum)
        raise ValueError(f"Invalid fisher block_reduce_method [{reduce_method}]")
