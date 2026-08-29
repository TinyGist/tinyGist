import torch

from src.fl_methods.definitions import MODEL_BLOCK_SCORE_METHODS, canonical_block_score_method
from src.fl_methods.segment_ops import compute_parameter_block_score_tensor
from src.models.definitions import CONV_BLOCK_MODES
from src.models.parameter_vector import ParameterBlockSet


class BlockScoreEmaBuffer:
    """EMA buffer for block scores used by selection and aggregation."""

    def __init__(self, model_dict, beta=0.5):
        self.model_dict = model_dict
        self.beta = float(beta)
        if not 0 <= self.beta < 1:
            raise ValueError("BlockScoreEmaBuffer beta must be in [0, 1)")
        self.buffer = {}

    def update(
            self,
            model_idx_list=None,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            block_score_method="mean_abs",
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> dict:
        if model_idx_list is None:
            model_idx_list = self.model_dict.keys()

        block_score_method = canonical_block_score_method(block_score_method)
        if block_score_method not in MODEL_BLOCK_SCORE_METHODS:
            raise ValueError("BlockScoreEmaBuffer only supports mean_abs, rms, and lipschitz block score methods")
        if conv_mode not in CONV_BLOCK_MODES:
            raise ValueError(f"Unknown conv block mode: {conv_mode}")

        for model_idx in model_idx_list:
            current_scores = self._compute_model_block_scores(
                model_idx,
                parameter_scope=parameter_scope,
                conv_mode=conv_mode,
                channel_length=channel_length,
                block_score_method=block_score_method,
                include_other_blocks=include_other_blocks,
                bn_mode=bn_mode,
                block_refinement=block_refinement,
                target_device=target_device,
            )
            previous_scores = self.buffer.get(model_idx)
            if previous_scores is None:
                self.buffer[model_idx] = current_scores
                continue
            if previous_scores.device != current_scores.device:
                previous_scores = previous_scores.to(current_scores.device)
            if previous_scores.numel() != current_scores.numel():
                raise ValueError("BlockScoreEmaBuffer block count changed; check model/block configuration")
            self.buffer[model_idx] = previous_scores.mul(self.beta).add(
                current_scores,
                alpha=1 - self.beta,
            )

        return self.get_block_scores(model_idx_list, target_device=target_device)

    def get_block_scores(self, model_idx_list=None, target_device=None) -> dict:
        if model_idx_list is None:
            model_idx_list = self.buffer.keys()
        return {
            model_idx: self.buffer[model_idx]
            .detach()
            .to(target_device if target_device is not None else "cpu")
            .clone()
            for model_idx in model_idx_list
            if model_idx in self.buffer
        }

    def _compute_model_block_scores(
            self,
            model_idx,
            parameter_scope="all",
            conv_mode="kernel",
            channel_length=0,
            block_score_method="mean_abs",
            include_other_blocks=False,
            bn_mode="affine",
            block_refinement=None,
            target_device=None,
    ) -> torch.Tensor:
        model = self.model_dict[model_idx]
        if not hasattr(model, "get_parameter_blocks"):
            raise ValueError("BlockScoreEmaBuffer requires models to implement get_parameter_blocks(...).")

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
        blocks = block_set.all_blocks
        if not blocks:
            return torch.empty(0, dtype=torch.float32, device=target_device)

        return torch.stack([
            compute_parameter_block_score_tensor(block, block_score_method)
            for block in blocks
        ])

