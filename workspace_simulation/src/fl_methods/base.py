import logging

import numpy as np
import torch
from bitmap import BitMap

from src.models.definitions import (
    canonical_bn_aggregation_mode,
    canonical_bn_process_mode,
    canonical_block_refinement_config,
    canonical_parameter_scope,
)
from src.models.parameter_vector import ParameterBlockSet
from src.scoring.block_grouping import build_block_layer_keys, group_sorted_blocks_within_layers
from src.scoring.definitions import (
    canonical_block_group_size,
    canonical_block_grouping_method,
    canonical_parameter_score_method,
    canonical_score_combine_method,
    canonical_score_project_method,
)
from src.scoring.parameter_scores import (
    create_bitmapidx_based_parameter_scores,
    create_parameter_score_thresholds,
    parameter_score_vector,
)
from src.utils.communication_recorder import PacketPayload

from .definitions import (
    BLOCK_ALIGNED_AGGREGATION_SCORE_SOURCES,
    BLOCK_SEGMENT_UNITS,
    REFINABLE_SEGMENT_UNITS,
    FISHER_AGGREGATION_SCORE_SOURCES,
    MODEL_BLOCK_SCORE_METHODS,
    canonical_aggregation_score_source,
    canonical_block_score_method,
    canonical_channel_length,
    canonical_segment_unit,
    SEGMENT_CREATE_METHODS,
    SEGMENT_CHOSEN_METHODS
)
from .segment_ops import (
    aggregate_packed_segments_block_fisher_based,
    aggregate_packed_segments_parameter_fisher_based,
    aggregate_packed_segments_scores_based,
    aggregate_segment_fisher_based,
    aggregate_segment_scores_based,
    combine_indexed_score_values,
    create_bitmap,
    create_segments_based_block_bitmap,
    create_segments_based_bitmap,
    compute_parameter_block_score_tensor,
    flatten_parameter_block,
    parameter_block_size,
    reduce_parameter_score_block_tensor,
)
from .segment_planning import (
    Combo,
    ordered_group_positions,
    segment_selection_probabilities as _segment_selection_probabilities,
    split_block_indices_by_size,
    split_parameter_indices,
    split_scored_block_groups_by_size,
)


log = logging.getLogger(__name__)


class FLMethods:
    def __init__(
            self,
            total_models_dict: dict,
            parameter_scope="all",
            bn_mode="affine",
            bn_process_as_base_unit=True,
            bn_aggregation_source="score",
    ):
        self._total_model_dict = total_models_dict
        self._outgoing_model_dict = total_models_dict
        self._num_total_models = len(total_models_dict)
        assert self._num_total_models > 1, f"Too few models, only {self._num_total_models}"
        self._parameter_scope = canonical_parameter_scope(parameter_scope)
        self._bn_mode = canonical_bn_aggregation_mode(bn_mode)
        self._bn_process_mode = canonical_bn_process_mode(bn_process_as_base_unit)
        self._bn_process_as_base_unit = self._bn_process_mode == "base_unit"
        self._base_bn_mode = self._bn_mode if self._bn_process_as_base_unit else "none"
        self._bn_aggregation_source = (
            "uniform"
            if self._bn_process_mode != "base_unit"
            else self._canonical_bn_aggregation_source(bn_aggregation_source)
        )
        self._working_device = None
        self._communication_recorder = None
        self._aggregation_weight_exp_base = None
        self._parameters_length = self._get_parameter_length()

    def set_working_device(self, working_device):
        """Choose where temporary FL-method tensors are stored and processed."""
        working_device = torch.device(working_device)
        if working_device.type not in {"cpu", "cuda"}:
            raise ValueError("FL working device must be cpu or cuda")
        if working_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("FL working device cuda was requested, but CUDA is unavailable")
        self._working_device = working_device

    def set_aggregation_weight_exponential(self, enabled, exponential_base=None):
        if not bool(enabled):
            self._aggregation_weight_exp_base = None
            return
        if isinstance(exponential_base, bool):
            raise ValueError("Aggregation weight exponential base must exceed 1")
        try:
            exponential_base = float(exponential_base)
        except (TypeError, ValueError) as exc:
            raise ValueError("Aggregation weight exponential base must exceed 1") from exc
        if not np.isfinite(exponential_base) or exponential_base <= 1:
            raise ValueError("Aggregation weight exponential base must exceed 1")
        self._aggregation_weight_exp_base = exponential_base

    def set_outgoing_models(self, outgoing_model_dict=None):
        """Set the model view used to construct outbound communication payloads."""
        if outgoing_model_dict is None:
            outgoing_model_dict = self._total_model_dict
        if set(outgoing_model_dict) != set(self._total_model_dict):
            raise ValueError(
                "Outgoing models must have exactly the same device ids as target models"
            )
        self._outgoing_model_dict = outgoing_model_dict

    def set_communication_recorder(self, recorder):
        self._communication_recorder = recorder

    def _record_communication_packet(self, **packet):
        if self._communication_recorder is not None:
            self._communication_recorder.record_packet(**packet)

    @staticmethod
    def _canonical_bn_aggregation_source(source):
        source = str(source).strip().replace("-", "_").replace(" ", "_").lower()
        if source in {"score", "aggregation_score", "weighted"}:
            return "aggregation_score"
        if source in {"uniform", "equal", "fedavg"}:
            return "uniform"
        raise ValueError("bn.aggregation.metric must be score or uniform")

    def _get_parameters_from_models(self, target_model_dict: dict) -> dict:
        return {
            model_idx: self._get_parameter_from_model(model)
            for model_idx, model in target_model_dict.items()
        }

    def _load_parameters_to_models(self, target_model_dict: dict, parameters_group):
        for model_idx, parameters in parameters_group.items():
            self._load_parameter_to_model(target_model_dict[model_idx], parameters)

    def _get_parameter_from_model(self, model: torch.nn.Module) -> torch.Tensor:
        parameters = model.get_parameter_vector(
            self._parameter_scope,
            bn_mode=self._base_bn_mode,
        )
        if self._working_device is not None:
            parameters = parameters.to(self._working_device)
        return parameters

    def _load_parameter_to_model(self, model: torch.nn.Module, param: torch.Tensor) -> torch.nn.Module:
        model.load_parameter_vector(param, self._parameter_scope, bn_mode=self._base_bn_mode)
        return model

    def _get_bn_from_model(self, model: torch.nn.Module) -> torch.Tensor:
        parameters = model.get_batchnorm_vector(
            self._parameter_scope,
            bn_mode=self._bn_mode,
        )
        if self._working_device is not None:
            parameters = parameters.to(self._working_device)
        return parameters

    def _load_bn_to_model(self, model: torch.nn.Module, bn_params: torch.Tensor):
        model.load_batchnorm_vector(bn_params, self._parameter_scope, bn_mode=self._bn_mode)

    def _uses_uniform_bn_aggregation(self):
        return (
            self._bn_mode != "none"
            and self._bn_process_mode != "base_unit"
            and self._bn_aggregation_source == "uniform"
        )

    def _uses_segment_bn_payloads(self):
        return (
            self._uses_uniform_bn_aggregation()
            and self._bn_process_mode == "separate_per_segment"
        )

    def _uses_recipient_once_bn_payloads(self):
        return (
            self._uses_uniform_bn_aggregation()
            and self._bn_process_mode == "separate_per_recipient"
        )

    def _bn_payload_for_model(self, model_idx):
        return self._get_bn_from_model(self._outgoing_model_dict[model_idx]).detach().clone()

    def _get_parameter_length(self):
        model_param_lengths = {}
        for model_idx, model in self._total_model_dict.items():
            if hasattr(model, "_iter_selected_tensors") and hasattr(model, "parameter_modules"):
                model_param_lengths[model_idx] = sum(
                    tensor.numel()
                    for tensor in model._iter_selected_tensors(
                        model.parameter_modules(self._parameter_scope),
                        self._base_bn_mode,
                    )
                )
            else:
                model_param_lengths[model_idx] = self._get_parameter_from_model(model).numel()
        unique_lengths = set(model_param_lengths.values())
        assert len(unique_lengths) == 1, (
            f"Models have different parameter lengths for scope [{self._parameter_scope}]: "
            f"{model_param_lengths}"
        )
        self._parameters_length = unique_lengths.pop()
        assert self._parameters_length > 0, f"No parameters found for scope [{self._parameter_scope}]"
        return self._parameters_length

class FLMethodsSeg(FLMethods):
    _LOCAL_UPDATE_UNIT_L2_MIN_REFERENCE_L2 = 1e-5

    def __init__(
            self,
            total_models_dict: dict,
            seg_divided_num: int = 1,
            parameter_scope="all",
            segment_unit="parameter",
            block_grouping_method="none",
            block_group_size=1,
            channel_length=0,
            block_refinement=None,
            aggregation_score_source="uniform",
            fisher_block=False,
            include_other_blocks=False,
            bn_mode="affine",
            bn_process_as_base_unit=True,
            bn_aggregation_source="score",
            segment_importance_metric="weight_abs",
            importance_parameter_to_unit="mean_abs",
            importance_unit_to_group="mean",
            importance_group_to_segment="mean",
            importance_lipschitz_ema_enabled=False,
            group_enabled=False,
            group_criterion_metric="gradient_abs_ema_online",
            group_criterion_parameter_to_unit="mean_abs",
            group_criterion_lipschitz_ema_enabled=False,
            segment_compose_method="score_sorted_balanced_payload",
            segment_pick_exp_normalize=True,
            segment_pick_exp_base=None,
    ):
        super().__init__(
            total_models_dict,
            parameter_scope=parameter_scope,
            bn_mode=bn_mode,
            bn_process_as_base_unit=bn_process_as_base_unit,
            bn_aggregation_source=bn_aggregation_source,
        )

        self._model_parameters_before_aggregation = None
        self._model_parameters_after_aggregation = None
        self._outgoing_model_parameters = None
        self._seg_divided_num = seg_divided_num
        self._segment_unit = canonical_segment_unit(segment_unit)
        self._block_grouping_method = canonical_block_grouping_method(block_grouping_method)
        self._block_group_size = canonical_block_group_size(block_group_size)
        self._channel_length = canonical_channel_length(channel_length)
        self._block_refinement = canonical_block_refinement_config(block_refinement)
        if self._block_refinement["enabled"] and self._segment_unit not in REFINABLE_SEGMENT_UNITS:
            raise ValueError("method.partition.refinement requires unit kernel or channel")
        self._aggregation_score_source = canonical_aggregation_score_source(aggregation_score_source)
        self._preserve_unit_l2 = False
        self._unit_l2_mode = "none"
        self._unit_l2_multiplier = None
        self._local_update_unit_l2_mode = "none"
        self._local_update_unit_l2_multiplier = None
        self._local_update_unit_l2_reference = None
        self._aggregation_update_unit_l2_mode = "none"
        self._aggregation_update_unit_l2_multiplier = None
        self._fisher_block_requested = bool(fisher_block)
        self._include_other_blocks = bool(include_other_blocks)
        self._segment_importance_metric = self._canonical_metric(segment_importance_metric)
        self._importance_parameter_to_unit = canonical_score_project_method(importance_parameter_to_unit)
        self._importance_unit_to_group = canonical_score_combine_method(importance_unit_to_group)
        self._importance_group_to_segment = canonical_score_combine_method(importance_group_to_segment)
        self._importance_lipschitz_ema_enabled = bool(importance_lipschitz_ema_enabled)
        self._group_enabled = bool(group_enabled) and self._block_grouping_method != "none"
        self._group_criterion_metric = self._canonical_metric(group_criterion_metric)
        self._group_criterion_parameter_to_unit = canonical_score_project_method(
            group_criterion_parameter_to_unit
        )
        self._group_criterion_lipschitz_ema_enabled = bool(
            group_criterion_lipschitz_ema_enabled
        )
        self._reuse_group_criterion_unit_scores = (
            self._group_enabled
            and self._group_criterion_metric == self._segment_importance_metric
            and self._group_criterion_parameter_to_unit == self._importance_parameter_to_unit
            and self._group_criterion_lipschitz_ema_enabled == self._importance_lipschitz_ema_enabled
        )
        self._segment_compose_method = segment_compose_method
        self._segment_pick_exp_normalize = bool(segment_pick_exp_normalize)
        self._segment_pick_exp_base = np.e if segment_pick_exp_base is None else float(segment_pick_exp_base)
        if not np.isfinite(self._segment_pick_exp_base) or self._segment_pick_exp_base <= 1:
            raise ValueError("segment selection exponential_base must be finite and greater than 1")
        self._segment_bitmap_method = "block" if self._segment_unit in BLOCK_SEGMENT_UNITS else "parameter"
        self._model_idx_to_bitmapANDsegments = None
        self._model_idx_to_received_segments = None
        self._model_idx_to_current_aggregating_data = None
        self._current_fisher_weights_dict = None
        self._block_sizes: np.ndarray | None = None
        self._block_count: int | None = None
        self._block_split_counts: tuple[int, int, int, int] | None = None
        self._block_layer_keys: list[tuple] | None = None
        self._block_indices_by_layer: list[np.ndarray] | None = None
        self._block_layer_ids: np.ndarray | None = None
        self._block_layer_count: int | None = None
        self._block_layer_layout_device_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._block_parameter_indices: torch.Tensor | None = None
        self._block_parameter_counts: torch.Tensor | None = None
        self._packed_parameter_to_block: np.ndarray | None = None
        self._block_lipschitz_mask: torch.Tensor | None = None
        self._block_layout_device_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._block_unit_l2_mask: torch.Tensor | None = None
        self._block_unit_l2_mask_device_cache: dict[str, torch.Tensor] = {}
        self._block_score_mask: torch.Tensor | None = None
        self._block_score_mask_device_cache: dict[str, torch.Tensor] = {}

    def _aggregate_segment_scores_based(
            self,
            target_params: torch.Tensor,
            received_seg_list: list[torch.Tensor],
            received_bitmap_list: list[BitMap],
            scores_list: list,
            target_score,
    ) -> torch.Tensor:
        log.info("Local score is %s, aggregated scores are %s", target_score, scores_list)
        return aggregate_segment_scores_based(
            target_params,
            received_seg_list,
            received_bitmap_list,
            scores_list,
            target_score,
            exponential_base=self._aggregation_weight_exp_base,
        )

    def _uses_block_segments(self):
        return self._segment_unit in BLOCK_SEGMENT_UNITS

    def _uses_fisher_scores(self):
        return self._aggregation_score_source in FISHER_AGGREGATION_SCORE_SOURCES

    def _uses_block_fisher_scores(self):
        return (
            self._uses_fisher_scores()
            and self._uses_block_segments()
            and (
                self._fisher_block_requested
                or self._aggregation_score_source in BLOCK_ALIGNED_AGGREGATION_SCORE_SOURCES
            )
        )

    @staticmethod
    def _canonical_bounded_multiplier(multiplier, value_name, minimum, inclusive):
        if isinstance(multiplier, bool):
            raise ValueError(f"{value_name} must be finite and positive")
        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{value_name} must be finite and positive") from exc
        invalid_bound = multiplier < minimum if inclusive else multiplier <= minimum
        if not np.isfinite(multiplier) or invalid_bound:
            comparator = "at least" if inclusive else "greater than"
            raise ValueError(f"{value_name} must be finite and {comparator} {minimum:g}")
        return multiplier

    def _require_block_unit_l2(self, enabled, value_name):
        if enabled and not self._uses_block_segments():
            raise ValueError(
                f"{value_name} requires a kernel, channel, or layer partition"
            )


    def set_unit_l2_constraint(self, mode, multiplier=None):
        mode = str(mode).strip().lower()
        if mode not in {"none", "exact", "bounded"}:
            raise ValueError("Unit-L2 mode must be none, exact, or bounded")
        if mode == "bounded":
            multiplier = self._canonical_bounded_multiplier(
                multiplier,
                "Unit-L2 bounded multiplier",
                1,
                True,
            )
        elif multiplier is not None:
            raise ValueError("Unit-L2 multiplier is valid only in bounded mode")
        self._require_block_unit_l2(mode != "none", "Unit-L2 constraint")
        self._unit_l2_mode = mode
        self._unit_l2_multiplier = multiplier
        self._preserve_unit_l2 = mode != "none"

    def set_preserve_unit_l2(self, preserve_unit_l2):
        self.set_unit_l2_constraint("exact" if bool(preserve_unit_l2) else "none")

    def set_local_update_unit_l2_constraint(self, mode, multiplier=None):
        mode = str(mode).strip().lower()
        if mode not in {"none", "bounded"}:
            raise ValueError("Local-update unit-L2 mode must be none or bounded")
        if mode == "bounded":
            multiplier = self._canonical_bounded_multiplier(
                multiplier,
                "Local-update unit-L2 bounded multiplier",
                0,
                False,
            )
        elif multiplier is not None:
            raise ValueError(
                "Local-update unit-L2 multiplier is valid only in bounded mode"
            )
        self._require_block_unit_l2(
            mode != "none",
            "Local-update unit-L2 constraint",
        )
        self._local_update_unit_l2_mode = mode
        self._local_update_unit_l2_multiplier = multiplier
        self._local_update_unit_l2_reference = None

    def set_aggregation_update_unit_l2_constraint(self, mode, multiplier=None):
        mode = str(mode).strip().lower()
        if mode not in {"none", "bounded"}:
            raise ValueError("Aggregation update unit-L2 mode must be none or bounded")
        if mode == "bounded":
            multiplier = self._canonical_bounded_multiplier(
                multiplier,
                "Aggregation update unit-L2 bounded multiplier",
                0,
                False,
            )
        elif multiplier is not None:
            raise ValueError(
                "Aggregation update unit-L2 multiplier is valid only in bounded mode"
            )
        self._require_block_unit_l2(
            mode != "none",
            "Aggregation update unit-L2 constraint",
        )
        self._aggregation_update_unit_l2_mode = mode
        self._aggregation_update_unit_l2_multiplier = multiplier

    @property
    def include_other_blocks(self):
        return self._include_other_blocks

    @property
    def bn_mode(self):
        return self._bn_mode

    def _prepare_block_layout(self):
        if not self._uses_block_segments():
            return
        if self._block_sizes is not None and self._block_count is not None:
            return
        assert self._parameters_length > 0, "No selected parameters for packed block layout"
        reference_model = next(iter(self._total_model_dict.values()))
        if not hasattr(reference_model, "_iter_selected_tensors") or not hasattr(
                reference_model,
                "_extract_parameter_blocks",
        ):
            raise ValueError(
                "Packed block processing requires model parameter tensor and block helpers."
            )

        selected_modules = reference_model.parameter_modules(self._parameter_scope)
        selected_tensors = list(reference_model._iter_selected_tensors(
            selected_modules,
            self._base_bn_mode,
        ))
        bias_tensor_ids = {
            id(parameter)
            for module in selected_modules
            for child in module.modules()
            for parameter_name, parameter in child.named_parameters(recurse=False)
            if parameter_name == "bias"
        }
        tensor_indices = {}
        cursor = 0
        for tensor in selected_tensors:
            tensor_indices[id(tensor)] = torch.arange(
                cursor,
                cursor + tensor.numel(),
                dtype=torch.long,
            ).view_as(tensor)
            cursor += tensor.numel()
        assert cursor == self._parameters_length, \
            "Packed block mapping does not match the selected parameter vector length"

        block_set = reference_model._extract_parameter_blocks(
            reference_model.parameter_modules(self._parameter_scope),
            conv_mode=self._segment_unit,
            channel_length=self._channel_length,
            tensor_getter=lambda tensor: tensor_indices[id(tensor)],
            include_other_blocks=self._include_other_blocks,
            bn_mode=self._base_bn_mode,
            block_refinement=self._block_refinement,
        )
        if not isinstance(block_set, ParameterBlockSet):
            raise TypeError("_extract_parameter_blocks(...) must return ParameterBlockSet")
        reference_blocks = block_set.all_blocks
        self._block_split_counts = block_set.split_counts
        self._block_count = len(reference_blocks)
        self._block_sizes = np.fromiter(
            (parameter_block_size(block) for block in reference_blocks),
            dtype=np.int64,
            count=self._block_count,
        )
        assert self._block_count > 0, "No parameter blocks found for block-based segmenting"
        assert self._block_sizes.sum() > 0, "No parameters found in block layout"

        self._block_parameter_indices = torch.cat([
            flatten_parameter_block(block)
            for block in reference_blocks
        ]).to(dtype=torch.long, device="cpu")
        canonical_unit_l2_mask = torch.cat([
            torch.full(
                (tensor.numel(),),
                (
                    isinstance(tensor, torch.nn.Parameter)
                    and tensor.requires_grad
                    and id(tensor) not in bias_tensor_ids
                ),
                dtype=torch.bool,
            )
            for tensor in selected_tensors
        ])
        self._block_unit_l2_mask = canonical_unit_l2_mask[
            self._block_parameter_indices
        ]
        if self._segment_unit == "layer":
            canonical_score_mask = torch.cat([
                torch.full(
                    (tensor.numel(),),
                    isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad,
                    dtype=torch.bool,
                )
                for tensor in selected_tensors
            ])
            self._block_score_mask = canonical_score_mask[self._block_parameter_indices]
            cursor = 0
            for block_size in self._block_sizes:
                next_cursor = cursor + int(block_size)
                if not torch.any(self._block_score_mask[cursor:next_cursor]):
                    raise ValueError(
                        "Every layer unit must contain at least one trainable parameter for scoring"
                    )
                cursor = next_cursor
        else:
            self._block_score_mask = torch.ones(
                self._block_parameter_indices.numel(),
                dtype=torch.bool,
            )

        assert int(self._block_parameter_indices.min()) >= 0
        assert int(self._block_parameter_indices.max()) < self._parameters_length
        self._block_parameter_counts = torch.bincount(
            self._block_parameter_indices,
            minlength=self._parameters_length,
        )
        assert torch.any(self._block_parameter_counts > 0), \
            "Packed block mapping does not cover any selected parameter"
        self._packed_parameter_to_block = np.repeat(
            np.arange(self._block_count, dtype=np.int32),
            self._block_sizes,
        )

        lipschitz_mask = np.ones(self._block_parameter_indices.numel(), dtype=np.bool_)
        cursor = 0
        for weight, bias in reference_blocks:
            cursor += weight.numel()
            if bias is not None:
                lipschitz_mask[cursor:cursor + bias.numel()] = False
                cursor += bias.numel()
        assert cursor == len(lipschitz_mask)
        self._block_lipschitz_mask = torch.from_numpy(lipschitz_mask)
        del reference_blocks, block_set

    def _block_layout_tensors(self, device):
        self._prepare_block_layout()
        device = torch.device(device)
        key = str(device)
        cached = self._block_layout_device_cache.get(key)
        if cached is None:
            cached = (
                self._block_parameter_indices.to(device=device),
                self._block_parameter_counts.to(device=device),
                torch.as_tensor(self._block_sizes, dtype=torch.long, device=device),
                self._block_lipschitz_mask.to(device=device),
            )
            self._block_layout_device_cache[key] = cached
        return cached

    def _block_score_mask_tensor(self, device):
        self._prepare_block_layout()
        assert self._block_score_mask is not None
        device = torch.device(device)
        key = str(device)
        cached = self._block_score_mask_device_cache.get(key)
        if cached is None:
            cached = self._block_score_mask.to(device=device)
            self._block_score_mask_device_cache[key] = cached
        return cached

    def _block_unit_l2_mask_tensor(self, device):
        self._prepare_block_layout()
        assert self._block_unit_l2_mask is not None
        device = torch.device(device)
        key = str(device)
        cached = self._block_unit_l2_mask_device_cache.get(key)
        if cached is None:
            cached = self._block_unit_l2_mask.to(device=device)
            self._block_unit_l2_mask_device_cache[key] = cached
        return cached

    def _pack_parameter_vector(self, parameter_vector: torch.Tensor) -> torch.Tensor:
        assert isinstance(parameter_vector, torch.Tensor), "Parameter vector must be a tensor"
        assert parameter_vector.numel() == self._parameters_length, \
            "Parameter vector length does not match packed block layout"
        target_device = self._working_device or parameter_vector.device
        parameter_vector = parameter_vector.detach().to(target_device)
        parameter_indices, _, _, _ = self._block_layout_tensors(target_device)
        return parameter_vector[parameter_indices]

    def _unpack_parameter_vector(
            self,
            packed_vector: torch.Tensor,
            fallback_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parameter_indices, parameter_counts, _, _ = self._block_layout_tensors(
            packed_vector.device
        )
        assert packed_vector.numel() == parameter_indices.numel(), \
            "Packed parameter vector length does not match block layout"
        parameter_sum = torch.zeros(
            self._parameters_length,
            dtype=packed_vector.dtype,
            device=packed_vector.device,
        )
        parameter_sum.index_add_(0, parameter_indices, packed_vector)
        valid = parameter_counts > 0
        if fallback_vector is None:
            assert torch.all(valid), \
                "Unpacking a partial block layout requires a fallback parameter vector"
            parameter_vector = parameter_sum
        else:
            assert fallback_vector.numel() == self._parameters_length
            parameter_vector = fallback_vector.detach().to(
                device=packed_vector.device,
                dtype=packed_vector.dtype,
            ).clone()
            parameter_vector[valid] = parameter_sum[valid]
        parameter_vector[valid] = (
            parameter_vector[valid]
            / parameter_counts[valid].to(dtype=packed_vector.dtype)
        )
        return parameter_vector

    @staticmethod
    def _validate_unit_l2_finite(tensor: torch.Tensor, value_name: str):
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(
                f"Cannot preserve unit L2 because {value_name} contains non-finite values"
            )

    def _packed_unit_l2_norms(self, packed_vector: torch.Tensor) -> torch.Tensor:
        assert isinstance(packed_vector, torch.Tensor), "Packed parameter vector must be a tensor"
        _, _, block_sizes, _ = self._block_layout_tensors(packed_vector.device)
        assert (
            packed_vector.numel() == int(block_sizes.sum())
        ), "Packed parameter vector length does not match block layout"
        self._validate_unit_l2_finite(packed_vector, "the packed parameter vector")
        norm_dtype = (
            torch.float64
            if packed_vector.dtype == torch.float64
            else torch.float32
        )
        unit_l2_mask = self._block_unit_l2_mask_tensor(packed_vector.device)
        norm_values = packed_vector.detach().to(dtype=norm_dtype).masked_fill(
            ~unit_l2_mask,
            0,
        )
        norms = torch.segment_reduce(
            norm_values.square(),
            "sum",
            lengths=block_sizes,
        ).clamp_min(0).sqrt()
        if not torch.all(torch.isfinite(norms)):
            raise ValueError("Cannot preserve unit L2 because a block norm is not finite")
        return norms

    def _bound_packed_update_unit_l2(
            self,
            packed_vector,
            reference_vector,
            multiplier,
            constraint_name,
            stage_name=None,
            *,
            skip_reference_l2_below=None,
    ):
        stage_name = constraint_name if stage_name is None else stage_name
        if packed_vector.shape != reference_vector.shape:
            raise ValueError(
                f"{constraint_name} and reference parameter shapes differ"
            )
        self._validate_unit_l2_finite(
            packed_vector,
            f"the post-{stage_name} packed parameter vector",
        )
        reference_vector = reference_vector.to(
            device=packed_vector.device,
            dtype=packed_vector.dtype,
        )
        self._validate_unit_l2_finite(
            reference_vector,
            f"the pre-{stage_name} packed parameter vector",
        )
        update_vector = packed_vector.detach() - reference_vector.detach()
        self._validate_unit_l2_finite(
            update_vector,
            f"the {constraint_name} vector",
        )
        reference_norms = self._packed_unit_l2_norms(reference_vector)
        update_norms = self._packed_unit_l2_norms(update_vector)
        upper_bounds = reference_norms * multiplier
        self._validate_unit_l2_finite(
            upper_bounds,
            f"the {constraint_name} unit-L2 bounds",
        )

        eps = 1e-12
        if skip_reference_l2_below is None:
            eligible_reference = reference_norms > eps
            skipped = (~eligible_reference) & (update_norms > eps)
            if torch.any(skipped):
                log.warning(
                    "Skipped %s unit-L2 clipping for %s zero-norm reference units",
                    constraint_name,
                    int(skipped.sum().item()),
                )
        else:
            eligible_reference = reference_norms >= skip_reference_l2_below
        scale_by_block = torch.where(
            eligible_reference & (update_norms > upper_bounds),
            upper_bounds / update_norms.clamp_min(eps),
            torch.ones_like(update_norms),
        )
        self._validate_unit_l2_finite(
            scale_by_block,
            f"the {constraint_name} scale factors",
        )

        _, _, block_sizes, _ = self._block_layout_tensors(packed_vector.device)
        scale_by_parameter = torch.repeat_interleave(
            scale_by_block,
            block_sizes,
        )
        scale_by_parameter = torch.where(
            self._block_unit_l2_mask_tensor(packed_vector.device),
            scale_by_parameter,
            torch.ones_like(scale_by_parameter),
        ).to(dtype=packed_vector.dtype)
        self._validate_unit_l2_finite(
            scale_by_parameter,
            f"the {constraint_name} parameter scale factors",
        )
        bounded = reference_vector + update_vector * scale_by_parameter
        self._validate_unit_l2_finite(
            bounded,
            f"the bounded post-{stage_name} parameter vector",
        )
        return bounded

    def _fallback_packed_update_l2_violations(
            self,
            packed_vector,
            reference_vector,
            multiplier,
    ):
        if packed_vector.shape != reference_vector.shape:
            raise ValueError(
                "Combined aggregation L2 parameter and reference shapes differ"
            )
        reference_vector = reference_vector.to(
            device=packed_vector.device,
            dtype=packed_vector.dtype,
        )
        reference_norms = self._packed_unit_l2_norms(reference_vector)
        update_norms = self._packed_unit_l2_norms(
            packed_vector.detach() - reference_vector.detach()
        )
        upper_bounds = reference_norms * multiplier
        self._validate_unit_l2_finite(
            upper_bounds,
            "the combined aggregation update unit-L2 bounds",
        )
        tolerance = (
            torch.finfo(update_norms.dtype).eps
            * torch.maximum(upper_bounds, torch.ones_like(upper_bounds))
            * 8
        )
        violations = (reference_norms > 1e-12) & (
            update_norms > upper_bounds + tolerance
        )
        if not torch.any(violations):
            return packed_vector

        _, _, block_sizes, _ = self._block_layout_tensors(packed_vector.device)
        fallback_mask = torch.repeat_interleave(violations, block_sizes)
        fallback_mask &= self._block_unit_l2_mask_tensor(packed_vector.device)
        resolved = torch.where(fallback_mask, reference_vector, packed_vector)
        self._validate_unit_l2_finite(
            resolved,
            "the combined aggregation L2 result",
        )
        log.debug(
            "Reverted %d primitive units because post-aggregation unit-L2 "
            "projection exceeded the aggregation update bound",
            int(violations.sum().item()),
        )
        return resolved

    def _restore_packed_unit_l2(
            self,
            packed_vector: torch.Tensor,
            reference_norms: torch.Tensor,
            fallback_vector: torch.Tensor,
            mode: str = "exact",
            multiplier: float | None = None,
    ) -> torch.Tensor:
        mode = str(mode).strip().lower()
        if mode not in {"exact", "bounded"}:
            raise ValueError("Unit-L2 restoration mode must be exact or bounded")
        if mode == "bounded":
            multiplier = self._canonical_bounded_multiplier(
                multiplier,
                "Unit-L2 bounded multiplier",
                1,
                True,
            )
        elif multiplier is not None:
            raise ValueError("Unit-L2 multiplier is valid only in bounded mode")
        aggregated_norms = self._packed_unit_l2_norms(packed_vector)
        if aggregated_norms.shape != reference_norms.shape:
            raise ValueError("Reference and aggregated unit-L2 layouts do not match")
        if fallback_vector.shape != packed_vector.shape:
            raise ValueError("Fallback parameter vector does not match aggregated parameters")
        self._validate_unit_l2_finite(fallback_vector, "the fallback parameter vector")
        _, _, block_sizes, _ = self._block_layout_tensors(packed_vector.device)
        reference_norms = reference_norms.to(
            device=aggregated_norms.device,
            dtype=aggregated_norms.dtype,
        )
        self._validate_unit_l2_finite(reference_norms, "the reference norms")
        if torch.any(reference_norms < 0):
            raise ValueError("Cannot preserve unit L2 because reference norms are negative")
        if mode == "bounded":
            lower_bounds = reference_norms / multiplier
            upper_bounds = reference_norms * multiplier
            self._validate_unit_l2_finite(lower_bounds, "the lower unit-L2 bounds")
            self._validate_unit_l2_finite(upper_bounds, "the upper unit-L2 bounds")
            target_norms = torch.maximum(
                lower_bounds,
                torch.minimum(aggregated_norms, upper_bounds),
            )
        else:
            target_norms = reference_norms
        self._validate_unit_l2_finite(target_norms, "the target unit-L2 norms")
        eps = 1e-12
        nonzero_aggregated = aggregated_norms > eps
        scale_by_block = torch.where(
            nonzero_aggregated,
            target_norms / aggregated_norms.clamp_min(eps),
            torch.zeros_like(aggregated_norms),
        )
        self._validate_unit_l2_finite(scale_by_block, "the per-unit scale factors")
        scale_by_parameter = torch.repeat_interleave(scale_by_block, block_sizes)
        unit_l2_mask = self._block_unit_l2_mask_tensor(packed_vector.device)
        scale_by_parameter = torch.where(
            unit_l2_mask,
            scale_by_parameter,
            torch.ones_like(scale_by_parameter),
        ).to(dtype=packed_vector.dtype)
        self._validate_unit_l2_finite(
            scale_by_parameter,
            "the parameter scale factors",
        )
        restored = packed_vector * scale_by_parameter

        use_fallback_by_block = (~nonzero_aggregated) & (target_norms > eps)
        if torch.any(use_fallback_by_block):
            use_fallback_by_parameter = torch.repeat_interleave(
                use_fallback_by_block,
                block_sizes,
            ) & unit_l2_mask
            restored = torch.where(
                use_fallback_by_parameter,
                fallback_vector.to(device=restored.device, dtype=restored.dtype),
                restored,
            )
        self._validate_unit_l2_finite(restored, "the restored parameter vector")
        return restored

    def _reduce_packed_block_values(self, packed_values, reduction, *, lipschitz=False):
        _, _, block_sizes, lipschitz_mask = self._block_layout_tensors(packed_values.device)
        assert packed_values.numel() == int(block_sizes.sum()), \
            "Packed score length does not match block layout"
        score_mask = self._block_score_mask_tensor(packed_values.device)
        if lipschitz:
            score_mask = score_mask & lipschitz_mask
            reduction = "l2"
        masked_values = packed_values.masked_fill(~score_mask, 0)
        numeric_mask = score_mask.to(dtype=packed_values.dtype)
        score_counts = torch.segment_reduce(numeric_mask, "sum", lengths=block_sizes)
        if not torch.all(score_counts > 0):
            raise ValueError("Every block must contain at least one value used for scoring")
        if reduction == "mean_abs":
            score_sums = torch.segment_reduce(
                masked_values.abs(),
                "sum",
                lengths=block_sizes,
            )
            reduced = score_sums / score_counts
        elif reduction == "mean":
            score_sums = torch.segment_reduce(
                masked_values,
                "sum",
                lengths=block_sizes,
            )
            reduced = score_sums / score_counts
        elif reduction in {"l2", "rms"}:
            reduced = torch.segment_reduce(
                masked_values.square(),
                "sum",
                lengths=block_sizes,
            )
            if reduction == "rms":
                reduced = reduced / score_counts
            reduced = reduced.clamp_min(0).sqrt()
        else:
            raise ValueError(f"Unsupported packed block reduction [{reduction}]")
        return reduced.to(dtype=torch.float32)

    def reduce_parameter_vector_to_block_scores(self, parameter_vector, reduction):
        """Project one canonical parameter vector to the prepared primitive blocks."""
        return self._reduce_packed_block_values(
            self._pack_parameter_vector(parameter_vector),
            reduction,
        )

    def _get_packed_parameters_from_models(self, target_model_dict: dict) -> dict:
        self._prepare_block_layout()
        return {
            model_idx: self._pack_parameter_vector(self._get_parameter_from_model(model))
            for model_idx, model in target_model_dict.items()
        }

    def _load_packed_parameters_to_models(self, target_model_dict: dict, packed_group: dict):
        for model_idx, packed_parameters in packed_group.items():
            fallback_parameters = self._get_parameter_from_model(target_model_dict[model_idx])
            self._load_parameter_to_model(
                target_model_dict[model_idx],
                self._unpack_parameter_vector(
                    packed_parameters,
                    fallback_vector=fallback_parameters,
                ),
            )

    def snapshot_local_update_unit_l2(self, model_idx_list):
        if self._local_update_unit_l2_mode == "none":
            self._local_update_unit_l2_reference = None
            return
        if self._local_update_unit_l2_reference is not None:
            raise RuntimeError("A local-update unit-L2 snapshot is already active")
        model_idx_list = tuple(model_idx_list)
        if len(model_idx_list) != len(set(model_idx_list)):
            raise ValueError("Local-update unit-L2 model indices must be unique")
        target_models = {
            model_idx: self._total_model_dict[model_idx]
            for model_idx in model_idx_list
        }
        packed_parameters = self._get_packed_parameters_from_models(target_models)
        for packed in packed_parameters.values():
            self._validate_unit_l2_finite(
                packed,
                "the pre-training packed parameter vector",
            )
        self._local_update_unit_l2_reference = {
            model_idx: packed.detach().clone()
            for model_idx, packed in packed_parameters.items()
        }

    def apply_local_update_unit_l2(self, model_idx_list):
        if self._local_update_unit_l2_mode == "none":
            self._local_update_unit_l2_reference = None
            return
        references = self._local_update_unit_l2_reference
        self._local_update_unit_l2_reference = None
        if references is None:
            raise RuntimeError("Local-update unit-L2 clipping requires a pre-training snapshot")
        model_idx_list = tuple(model_idx_list)
        if set(model_idx_list) != set(references):
            raise ValueError(
                "Local-update unit-L2 model indices do not match the pre-training snapshot"
            )
        target_models = {
            model_idx: self._total_model_dict[model_idx]
            for model_idx in model_idx_list
        }
        current_parameters = self._get_packed_parameters_from_models(target_models)
        bounded_parameters = {
            model_idx: self._bound_packed_update_unit_l2(
                current_parameters[model_idx],
                references[model_idx],
                self._local_update_unit_l2_multiplier,
                "local update",
                stage_name="training",
                skip_reference_l2_below=(
                    self._LOCAL_UPDATE_UNIT_L2_MIN_REFERENCE_L2
                ),
            )
            for model_idx in model_idx_list
        }
        self._load_packed_parameters_to_models(target_models, bounded_parameters)

    def discard_local_update_unit_l2_snapshot(self):
        self._local_update_unit_l2_reference = None

    def _packed_parameter_indices_for_blocks(self, block_indices):
        self._prepare_block_layout()
        block_indices = np.asarray(block_indices, dtype=np.int64)
        selected_blocks = np.zeros(self._block_count, dtype=np.bool_)
        selected_blocks[block_indices] = True
        return np.flatnonzero(selected_blocks[self._packed_parameter_to_block])

    def _packed_parameter_indices_by_segment(self, segment_block_indices):
        self._prepare_block_layout()
        block_to_segment = np.full(self._block_count, -1, dtype=np.int16)
        for segment_idx, block_indices in enumerate(segment_block_indices):
            block_to_segment[np.asarray(block_indices, dtype=np.int64)] = segment_idx
        assert np.all(block_to_segment >= 0), "Block segment partition does not cover the layout"
        parameter_segments = block_to_segment[self._packed_parameter_to_block]
        return [
            np.flatnonzero(parameter_segments == segment_idx)
            for segment_idx in range(len(segment_block_indices))
        ]

    def _prepare_block_layer_keys(self):
        self._prepare_block_layout()
        if self._block_layer_keys is not None:
            return
        reference_model = next(iter(self._total_model_dict.values()))
        self._block_layer_keys = build_block_layer_keys(
            reference_model,
            parameter_scope=self._parameter_scope,
            conv_mode=self._segment_unit,
            channel_length=self._channel_length,
            include_other_blocks=self._include_other_blocks,
            bn_mode=self._base_bn_mode,
            block_refinement=self._block_refinement,
        )
        assert self._block_count is not None, "Block layout has not been prepared"
        assert len(self._block_layer_keys) == self._block_count, (
            "Block grouping layer-key count does not match the prepared block layout"
        )
        blocks_by_layer = {}
        for block_idx, layer_key in enumerate(self._block_layer_keys):
            blocks_by_layer.setdefault(layer_key, []).append(block_idx)
        self._block_indices_by_layer = [
            np.asarray(block_indices, dtype=np.int64)
            for block_indices in blocks_by_layer.values()
        ]
        self._block_layer_count = len(self._block_indices_by_layer)
        self._block_layer_ids = np.empty(self._block_count, dtype=np.int64)
        for layer_idx, block_indices in enumerate(self._block_indices_by_layer):
            self._block_layer_ids[block_indices] = layer_idx

    def normalize_block_scores_by_layer_l2(self, block_scores: torch.Tensor) -> torch.Tensor:
        """Normalize one or more complete block-score vectors within each layer."""
        self._prepare_block_layer_keys()
        assert self._block_count is not None
        assert self._block_layer_ids is not None
        assert self._block_layer_count is not None
        scores = block_scores.detach()
        squeeze_result = scores.ndim == 1
        if squeeze_result:
            scores = scores.unsqueeze(0)
        if scores.ndim != 2 or scores.shape[1] != self._block_count:
            raise ValueError(
                "Layer L2 normalization requires one aggregation weight per block"
            )
        if not torch.all(torch.isfinite(scores) & (scores >= 0)):
            raise ValueError("Layer L2 normalization requires finite non-negative scores")

        device_key = str(scores.device)
        cached_layout = self._block_layer_layout_device_cache.get(device_key)
        if cached_layout is None:
            layer_ids = torch.as_tensor(
                self._block_layer_ids,
                dtype=torch.long,
                device=scores.device,
            )
            layer_counts = torch.bincount(
                layer_ids,
                minlength=self._block_layer_count,
            )
            cached_layout = (layer_ids, layer_counts)
            self._block_layer_layout_device_cache[device_key] = cached_layout
        layer_ids, layer_counts = cached_layout

        expanded_layer_ids = layer_ids.unsqueeze(0).expand(scores.shape[0], -1)
        layer_squared_sums = torch.zeros(
            (scores.shape[0], self._block_layer_count),
            dtype=scores.dtype,
            device=scores.device,
        )
        layer_squared_sums.scatter_add_(1, expanded_layer_ids, scores.square())
        layer_norms = layer_squared_sums.clamp_min(0).sqrt()
        block_norms = layer_norms.gather(1, expanded_layer_ids)
        normalized = scores / block_norms.clamp_min(1e-12)
        uniform_unit_norm = (
            layer_counts.to(dtype=scores.dtype).rsqrt()[layer_ids]
            .unsqueeze(0)
            .expand_as(scores)
        )
        result = torch.where(block_norms > 1e-12, normalized, uniform_unit_norm)
        return result.squeeze(0) if squeeze_result else result

    def _get_parameter_blocks_from_model(self, model: torch.nn.Module) -> list:
        if not hasattr(model, "get_parameter_blocks"):
            raise ValueError(
                "Block-based segmenting requires models to implement get_parameter_blocks(...)."
            )
        if self._working_device is None:
            block_set = model.get_parameter_blocks(
                self._parameter_scope,
                conv_mode=self._segment_unit,
                channel_length=self._channel_length,
                include_other_blocks=self._include_other_blocks,
                bn_mode=self._base_bn_mode,
                block_refinement=self._block_refinement,
            )
        else:
            if not hasattr(model, "parameter_modules") or not hasattr(
                    model,
                    "_extract_parameter_blocks",
            ):
                raise ValueError(
                    "Block-based CPU/GPU placement requires model parameter block helpers."
                )
            tensor_cache = {}

            def working_tensor(tensor):
                tensor_id = id(tensor)
                if tensor_id not in tensor_cache:
                    tensor_cache[tensor_id] = tensor.detach().to(
                        device=self._working_device,
                        copy=True,
                    )
                return tensor_cache[tensor_id]

            block_set = model._extract_parameter_blocks(
                model.parameter_modules(self._parameter_scope),
                conv_mode=self._segment_unit,
                channel_length=self._channel_length,
                tensor_getter=working_tensor,
                include_other_blocks=self._include_other_blocks,
                bn_mode=self._base_bn_mode,
                block_refinement=self._block_refinement,
            )
        if not isinstance(block_set, ParameterBlockSet):
            raise TypeError("get_parameter_blocks(...) must return ParameterBlockSet")

        split_counts = block_set.split_counts
        if self._block_split_counts is None:
            self._block_split_counts = split_counts
        assert self._block_split_counts == split_counts, (
            f"Models have different block split counts for scope [{self._parameter_scope}] "
            f"and unit [{self._segment_unit}]"
        )
        return block_set.all_blocks

    def get_parameter_blocks_for_working_device(self, model: torch.nn.Module) -> list:
        """Expose the method's canonical block layout on its configured working device."""
        return self._get_parameter_blocks_from_model(model)

    def _load_block_parameters_to_models(self, target_model_dict: dict, block_group: dict):
        assert self._block_split_counts is not None, "Block layout has not been prepared"
        for model_idx, blocks in block_group.items():
            block_set = ParameterBlockSet.from_all_blocks(
                blocks,
                self._block_split_counts,
            )
            target_model_dict[model_idx].load_parameter_blocks(
                block_set,
                parameter_scope=self._parameter_scope,
                conv_mode=self._segment_unit,
                channel_length=self._channel_length,
                include_other_blocks=self._include_other_blocks,
                bn_mode=self._base_bn_mode,
                block_refinement=self._block_refinement,
            )

    def _split_block_indices_by_size(self, ordered_block_indices):
        assert self._block_sizes is not None, "Block layout has not been prepared"
        return split_block_indices_by_size(
            self._block_sizes,
            self._seg_divided_num,
            ordered_block_indices,
        )

    def _split_scored_block_groups_by_size(
            self,
            ordered_block_groups,
            ordered_group_scores=None,
            segment_score_combine="mean",
    ):
        assert self._block_sizes is not None, "Block layout has not been prepared"
        return split_scored_block_groups_by_size(
            self._block_sizes,
            self._seg_divided_num,
            ordered_block_groups,
            ordered_group_scores,
            segment_score_combine,
        )

    @staticmethod
    def _ordered_group_positions(block_groups, group_scores):
        return ordered_group_positions(block_groups, group_scores)

    def _score_groups_from_block_scores(self, block_groups, block_scores, combine_method):
        return combine_indexed_score_values(block_scores, block_groups, combine_method)

    @staticmethod
    def _canonical_metric(metric):
        try:
            return canonical_block_score_method(metric)
        except ValueError:
            return canonical_parameter_score_method(metric)

    @staticmethod
    def _metric_is_block(metric):
        try:
            canonical_block_score_method(metric)
            return True
        except ValueError:
            return False

    def _parameter_score_blocks_for_metric(self, metric, model_blocks, block_parameter_score_blocks_by_method):
        metric = canonical_parameter_score_method(metric)
        if metric == "weight_abs":
            return model_blocks, isinstance(model_blocks, torch.Tensor)
        if block_parameter_score_blocks_by_method is None or metric not in block_parameter_score_blocks_by_method:
            raise ValueError(f"metric={metric} requires parameter score blocks")
        score_values = block_parameter_score_blocks_by_method[metric]
        if isinstance(score_values, torch.Tensor):
            return self._pack_parameter_vector(score_values), True
        return score_values, False

    def _compute_stage_block_scores(
            self,
            metric,
            model_blocks,
            block_parameter_score_blocks_by_method=None,
            selection_lipschitz_block_scores=None,
            block_interaction_scores=None,
            project_method="mean_abs",
    ):
        metric = self._canonical_metric(metric)
        if self._metric_is_block(metric):
            return self._compute_block_scores(
                model_blocks,
                block_score_method=metric,
                selection_lipschitz_block_scores=selection_lipschitz_block_scores,
                block_interaction_scores=block_interaction_scores,
            )

        parameter_score_blocks, packed = self._parameter_score_blocks_for_metric(
            metric,
            model_blocks,
            block_parameter_score_blocks_by_method,
        )
        if packed:
            return (
                self._reduce_packed_block_values(parameter_score_blocks, project_method)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        assert self._block_count is not None, "Block layout has not been prepared"
        assert len(parameter_score_blocks) == self._block_count, \
            "Parameter score block count does not match block layout"
        return (
            torch.stack([
                reduce_parameter_score_block_tensor(block, project_method)
                for block in parameter_score_blocks
            ])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def _build_consistent_block_idx_list(self):
        self._prepare_block_layout()
        return self._split_block_indices_by_size(np.arange(self._block_count))

    def _build_random_block_idx_list(self):
        self._prepare_block_layout()
        return self._split_block_indices_by_size(np.random.permutation(self._block_count))

    def _build_block_segments_from_shared_idx_list(self, block_idx_list, method_name):
        self._prepare_block_layout()
        self._outgoing_model_parameters = self._get_packed_parameters_from_models(
            self._outgoing_model_dict
        )
        self._model_idx_to_bitmapANDsegments = {}
        for model_idx, model_blocks in self._outgoing_model_parameters.items():
            pair_dict = self._build_block_pair_dict(model_blocks, block_idx_list)
            self._model_idx_to_bitmapANDsegments[model_idx] = self._attach_bn_payloads_to_pair_dict(
                model_idx,
                pair_dict,
            )
        self._model_idx_to_bitmapANDsegments["method"] = f"{method_name}_{self._segment_unit}"
        return self._model_idx_to_bitmapANDsegments

    def _build_block_pair_dict(self, model_blocks, block_idx_list, block_scores=None, segment_mean_values=None):
        pair_dict = {}
        bitmap_list = []
        normalized_block_idx_list = []
        if segment_mean_values is not None:
            assert len(segment_mean_values) == len(block_idx_list), \
                "Segment mean-value count must match segment count"
        for i, block_idx in enumerate(block_idx_list):
            sorted_block_idx = sorted(int(idx) for idx in block_idx)
            assert len(set(sorted_block_idx)) == len(sorted_block_idx), \
                "A block appears multiple times in one segment"
            normalized_block_idx_list.append(sorted_block_idx)
            bitmap_list.append(create_bitmap(self._block_count, sorted_block_idx))
        # Transport payload stays flattened, while the bitmap only marks block ids.
        # The receiver reconstructs blocks from this order and its local block shapes.
        if isinstance(model_blocks, torch.Tensor):
            parameter_idx_list = self._packed_parameter_indices_by_segment(
                normalized_block_idx_list
            )
            segments = [
                model_blocks[torch.as_tensor(
                    parameter_idx,
                    dtype=torch.long,
                    device=model_blocks.device,
                )]
                for parameter_idx in parameter_idx_list
            ]
        else:
            segments = create_segments_based_block_bitmap(model_blocks, bitmap_list)
        for i, (bitmap, block_idx, segment) in enumerate(zip(
                bitmap_list,
                normalized_block_idx_list,
                segments,
        )):
            mean_value = (
                float(segment_mean_values[i])
                if segment_mean_values is not None
                else self._block_segment_mean_value(segment, block_idx, block_scores)
            )
            pair_dict[f"pair{i}"] = {
                "bitmap": bitmap,
                "bitmap_bits": self._block_count,
                "bitmap_unit": "block",
                "block_idx": block_idx,
                "parameter_seg": segment,
                "mean_value": mean_value,
            }
        return pair_dict

    def _attach_bn_payloads_to_pair_dict(self, model_idx, pair_dict):
        if not self._uses_segment_bn_payloads():
            return pair_dict
        bn_payload = self._bn_payload_for_model(model_idx)
        for pair_data in pair_dict.values():
            pair_data["bn_param"] = bn_payload
        return pair_dict

    def _compute_block_scores(
            self,
            model_blocks: list,
            block_score_method=None,
            selection_lipschitz_block_scores=None,
            block_interaction_scores=None,
    ) -> np.ndarray:
        if block_score_method is None:
            raise ValueError("block_score_method is required")
        block_score_method = canonical_block_score_method(block_score_method)
        if block_score_method == "fisher_lipschitz_cooperation":
            return self._compute_fisher_lipschitz_cooperation_scores(
                model_blocks,
                block_interaction_scores,
                selection_lipschitz_block_scores=selection_lipschitz_block_scores,
            )

        if selection_lipschitz_block_scores is not None:
            assert self._block_count is not None, "Block layout has not been prepared"
            assert isinstance(selection_lipschitz_block_scores, torch.Tensor), \
                "Selection Lipschitz block scores must be a tensor"
            assert selection_lipschitz_block_scores.numel() == self._block_count, \
                "Selection Lipschitz block score length does not match block layout"
            return selection_lipschitz_block_scores.detach().cpu().numpy().astype(np.float64)

        if block_score_method not in MODEL_BLOCK_SCORE_METHODS:
            raise ValueError(f"Invalid model block_score_method [{block_score_method}]")
        if isinstance(model_blocks, torch.Tensor):
            return (
                self._reduce_packed_block_values(
                    model_blocks,
                    block_score_method,
                    lipschitz=block_score_method == "lipschitz",
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        return (
            torch.stack([
                compute_parameter_block_score_tensor(block, block_score_method)
                for block in model_blocks
            ])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def _compute_fisher_lipschitz_cooperation_scores(
            self,
            model_blocks: list,
            block_interaction_scores,
            selection_lipschitz_block_scores=None,
    ) -> np.ndarray:
        if block_interaction_scores is None:
            raise ValueError("block_score_method=fisher_lipschitz_cooperation requires block interaction scores")
        assert self._block_count is not None, "Block layout has not been prepared"
        assert isinstance(block_interaction_scores, torch.Tensor), \
            "Block interaction scores must be a tensor"
        assert block_interaction_scores.numel() == self._block_count, \
            "Block interaction score length does not match block layout"
        block_interaction_scores = block_interaction_scores.detach().cpu().to(dtype=torch.float32)
        if selection_lipschitz_block_scores is not None:
            assert isinstance(selection_lipschitz_block_scores, torch.Tensor), \
                "Selection Lipschitz block scores must be a tensor"
            assert selection_lipschitz_block_scores.numel() == self._block_count, \
                "Selection Lipschitz block score length does not match block layout"
            lipschitz_scores = selection_lipschitz_block_scores.detach().cpu().to(dtype=torch.float32)
        else:
            if isinstance(model_blocks, torch.Tensor):
                lipschitz_scores = self._reduce_packed_block_values(
                    model_blocks,
                    "l2",
                    lipschitz=True,
                ).detach().cpu()
            else:
                lipschitz_scores = torch.stack([
                    compute_parameter_block_score_tensor(model_block, "lipschitz")
                    for model_block in model_blocks
                ]).detach().cpu()
        return (
            (lipschitz_scores * block_interaction_scores.clamp_min(0.0))
            .clamp_min(0.0)
            .numpy()
            .astype(np.float64)
        )

    @staticmethod
    def _block_segment_mean_value(parameter_seg, block_idx, block_scores=None):
        if block_scores is not None and len(block_idx) > 0:
            return float(np.mean(block_scores[block_idx]))
        if parameter_seg.numel() == 0:
            return 0.0
        return float(parameter_seg.abs().mean())

    def _pick_segments_based_probabilistic(self, model_idx, num_pick):
        assert isinstance(self._model_idx_to_bitmapANDsegments, dict), "Not get bitmap and segments yet"

        mean_value_list = []
        pair_idx_list = []
        for pair_idx, pair_dict in self._model_idx_to_bitmapANDsegments[model_idx].items():
            mean_value_list.append(pair_dict["mean_value"])
            pair_idx_list.append(pair_idx)
        mean_value_array = np.array(mean_value_list)
        pair_idx_array = np.array(pair_idx_list)
        base_probability_array, probability_array = _segment_selection_probabilities(
            mean_value_array,
            self._segment_pick_exp_normalize,
            self._segment_pick_exp_base,
        )
        score_summary = ", ".join(
            f"{pair_idx}={float(score):.6g}"
            for pair_idx, score in zip(pair_idx_array, mean_value_array)
        )
        base_probability_summary = ", ".join(
            f"{pair_idx}={float(probability):.6g}"
            for pair_idx, probability in zip(pair_idx_array, base_probability_array)
        )
        probability_summary = ", ".join(
            f"{pair_idx}={float(probability):.6g}"
            for pair_idx, probability in zip(pair_idx_array, probability_array)
        )
        log.info(
            "Segment scores before probabilistic selection for model [%s], requested picks [%s]: "
            "scores={%s}; base_probabilities={%s}; final_probabilities={%s}; "
            "exp_normalize=%s; exponential_base=%.6g",
            model_idx,
            num_pick,
            score_summary,
            base_probability_summary,
            probability_summary,
            self._segment_pick_exp_normalize,
            self._segment_pick_exp_base,
        )
        picked_res = np.random.choice(pair_idx_array, size=num_pick, replace=True, p=probability_array)
        return picked_res.tolist()

    def _create_bitmap_segments_random_same(self):
        if self._uses_block_segments():
            block_idx_list = self._build_random_block_idx_list()
            return self._build_block_segments_from_shared_idx_list(block_idx_list, method_name="random_same")
        bitmap_list = self._build_shared_random_bitmaps()
        return self._build_segments_from_shared_bitmaps(bitmap_list, method_name="random_same")

    def _create_bitmap_segments_consistent(self):
        if self._uses_block_segments():
            block_idx_list = self._build_consistent_block_idx_list()
            return self._build_block_segments_from_shared_idx_list(block_idx_list, method_name="consistent")
        bitmap_list = self._build_consistent_bitmaps()
        return self._build_segments_from_shared_bitmaps(bitmap_list, method_name="consistent")

    def _create_bitmap_segments_random_each(self):
        assert self._seg_divided_num > 0, f"The number of segments from one model is {self._seg_divided_num}"
        if self._uses_block_segments():
            self._prepare_block_layout()
            self._model_idx_to_bitmapANDsegments = dict()
            self._outgoing_model_parameters = self._get_packed_parameters_from_models(
                self._outgoing_model_dict
            )
            for model_idx, model_blocks in self._outgoing_model_parameters.items():
                block_idx_list = self._build_random_block_idx_list()
                self._model_idx_to_bitmapANDsegments[model_idx] = self._build_block_pair_dict(
                    model_blocks,
                    block_idx_list,
                )
                self._attach_bn_payloads_to_pair_dict(model_idx, self._model_idx_to_bitmapANDsegments[model_idx])
            self._model_idx_to_bitmapANDsegments["method"] = f"random_each_{self._segment_unit}"
            return self._model_idx_to_bitmapANDsegments

        self._model_idx_to_bitmapANDsegments = dict()
        self._outgoing_model_parameters = self._get_parameters_from_models(self._outgoing_model_dict)
        for model_idx, model_params in self._outgoing_model_parameters.items():
            bitmap_list = self._build_random_bitmaps()
            self._model_idx_to_bitmapANDsegments[model_idx] = self._build_pair_dict(model_params, bitmap_list)
            self._attach_bn_payloads_to_pair_dict(model_idx, self._model_idx_to_bitmapANDsegments[model_idx])
        self._model_idx_to_bitmapANDsegments["method"] = "random_each"
        return self._model_idx_to_bitmapANDsegments

    def _required_block_importance_metrics(self):
        metrics = {self._segment_importance_metric}
        if self._group_enabled:
            metrics.add(self._group_criterion_metric)
        return metrics

    def _block_importance_score_inputs(
            self,
            model_idx,
            required_metrics,
            selection_lipschitz_score_weights_dict=None,
            block_parameter_score_weights_dict=None,
            block_interaction_score_weights_dict=None,
    ):
        block_interaction_scores = None
        if "fisher_lipschitz_cooperation" in required_metrics:
            if block_interaction_score_weights_dict is None:
                raise ValueError(
                    "metric=fisher_lipschitz_cooperation requires "
                    "current_block_interaction_score_weights_dict"
                )
            block_interaction_scores = block_interaction_score_weights_dict[model_idx]

        block_parameter_score_blocks_by_method = None
        if block_parameter_score_weights_dict is not None:
            block_parameter_score_blocks_by_method = {
                score_method: model_score_blocks[model_idx]
                for score_method, model_score_blocks in block_parameter_score_weights_dict.items()
            }

        selection_lipschitz_block_scores = None
        if selection_lipschitz_score_weights_dict is not None:
            selection_lipschitz_block_scores = selection_lipschitz_score_weights_dict[model_idx]

        return (
            selection_lipschitz_block_scores,
            block_parameter_score_blocks_by_method,
            block_interaction_scores,
        )

    def _block_importance_groups(
            self,
            model_blocks,
            block_parameter_score_blocks_by_method=None,
            selection_lipschitz_block_scores=None,
            block_interaction_scores=None,
    ):
        if not self._group_enabled:
            return [[block_idx] for block_idx in range(self._block_count)], None
        if self._block_grouping_method not in {"sensitivity_aligned", "sensitivity_diverse"}:
            raise ValueError(f"Unsupported group.method [{self._block_grouping_method}]")

        self._prepare_block_layer_keys()
        criterion_scores = self._compute_stage_block_scores(
            self._group_criterion_metric,
            model_blocks,
            block_parameter_score_blocks_by_method=block_parameter_score_blocks_by_method,
            selection_lipschitz_block_scores=(
                selection_lipschitz_block_scores
                if self._group_criterion_lipschitz_ema_enabled
                else None
            ),
            block_interaction_scores=block_interaction_scores,
            project_method=self._group_criterion_parameter_to_unit,
        )
        return (
            group_sorted_blocks_within_layers(
                criterion_scores,
                self._block_layer_keys,
                self._block_group_size,
                self._block_grouping_method,
                layer_block_indices=self._block_indices_by_layer,
            ),
            criterion_scores,
        )

    def _block_importance_scores(
            self,
            model_blocks,
            block_groups,
            block_parameter_score_blocks_by_method=None,
            selection_lipschitz_block_scores=None,
            block_interaction_scores=None,
            precomputed_unit_scores=None,
    ):
        unit_scores = precomputed_unit_scores
        if unit_scores is None:
            unit_scores = self._compute_stage_block_scores(
                self._segment_importance_metric,
                model_blocks,
                block_parameter_score_blocks_by_method=block_parameter_score_blocks_by_method,
                selection_lipschitz_block_scores=(
                    selection_lipschitz_block_scores
                    if self._importance_lipschitz_ema_enabled
                    else None
                ),
                block_interaction_scores=block_interaction_scores,
                project_method=self._importance_parameter_to_unit,
            )
        group_scores = self._score_groups_from_block_scores(
            block_groups,
            unit_scores,
            self._importance_unit_to_group,
        )
        return unit_scores, group_scores

    def _block_importance_pair_dict(
            self,
            model_idx,
            model_blocks,
            selection_lipschitz_block_scores=None,
            block_parameter_score_blocks_by_method=None,
            block_interaction_scores=None,
    ):
        block_groups, criterion_scores = self._block_importance_groups(
            model_blocks,
            block_parameter_score_blocks_by_method=block_parameter_score_blocks_by_method,
            selection_lipschitz_block_scores=selection_lipschitz_block_scores,
            block_interaction_scores=block_interaction_scores,
        )
        unit_scores, group_scores = self._block_importance_scores(
            model_blocks,
            block_groups,
            block_parameter_score_blocks_by_method=block_parameter_score_blocks_by_method,
            selection_lipschitz_block_scores=selection_lipschitz_block_scores,
            block_interaction_scores=block_interaction_scores,
            precomputed_unit_scores=(
                criterion_scores
                if self._reuse_group_criterion_unit_scores
                else None
            ),
        )
        if self._segment_compose_method != "score_sorted_balanced_payload":
            raise ValueError("segment.compose_method currently supports score_sorted_balanced_payload")

        ordered_positions = self._ordered_group_positions(block_groups, group_scores)
        ordered_block_groups = [block_groups[position] for position in ordered_positions]
        ordered_group_scores = group_scores[np.asarray(ordered_positions, dtype=np.int64)]
        block_idx_list, segment_mean_values = self._split_scored_block_groups_by_size(
            ordered_block_groups,
            ordered_group_scores,
            self._importance_group_to_segment,
        )
        pair_dict = self._build_block_pair_dict(
            model_blocks,
            block_idx_list,
            block_scores=unit_scores,
            segment_mean_values=segment_mean_values,
        )
        return self._attach_bn_payloads_to_pair_dict(model_idx, pair_dict)

    def _create_bitmap_segments_importance(
            self,
            selection_lipschitz_score_weights_dict=None,
            parameter_score_weights_dict=None,
            block_parameter_score_weights_dict=None,
            block_interaction_score_weights_dict=None,
    ):
        assert self._seg_divided_num > 0, f"The number of segments from one model is {self._seg_divided_num}"
        if self._uses_block_segments():
            self._prepare_block_layout()
            self._model_idx_to_bitmapANDsegments = dict()
            self._outgoing_model_parameters = self._get_packed_parameters_from_models(
                self._outgoing_model_dict
            )
            required_metrics = self._required_block_importance_metrics()
            for model_idx, model_blocks in self._outgoing_model_parameters.items():
                (
                    selection_lipschitz_block_scores,
                    block_parameter_score_blocks_by_method,
                    block_interaction_scores,
                ) = self._block_importance_score_inputs(
                    model_idx,
                    required_metrics,
                    selection_lipschitz_score_weights_dict=selection_lipschitz_score_weights_dict,
                    block_parameter_score_weights_dict=block_parameter_score_weights_dict,
                    block_interaction_score_weights_dict=block_interaction_score_weights_dict,
                )
                self._model_idx_to_bitmapANDsegments[model_idx] = self._block_importance_pair_dict(
                    model_idx,
                    model_blocks,
                    selection_lipschitz_block_scores=selection_lipschitz_block_scores,
                    block_parameter_score_blocks_by_method=block_parameter_score_blocks_by_method,
                    block_interaction_scores=block_interaction_scores,
                )
            self._model_idx_to_bitmapANDsegments["method"] = f"importance_{self._segment_unit}"
            return self._model_idx_to_bitmapANDsegments

        self._model_idx_to_bitmapANDsegments = dict()
        self._outgoing_model_parameters = self._get_parameters_from_models(self._outgoing_model_dict)
        for model_idx, model_params in self._outgoing_model_parameters.items():
            parameter_score_weights = None
            if parameter_score_weights_dict is not None:
                if self._segment_importance_metric in parameter_score_weights_dict:
                    parameter_score_weights = parameter_score_weights_dict[self._segment_importance_metric][model_idx]
                else:
                    parameter_score_weights = parameter_score_weights_dict[model_idx]
            parameter_scores = parameter_score_vector(
                model_params,
                self._segment_importance_metric,
                parameter_score_weights=parameter_score_weights,
            )
            thresholds_list = create_parameter_score_thresholds(parameter_scores, self._seg_divided_num)
            bit_idx_list = create_bitmapidx_based_parameter_scores(parameter_scores, thresholds_list)
            bitmap_list = [create_bitmap(self._parameters_length, bit_idx) for bit_idx in bit_idx_list]
            self._model_idx_to_bitmapANDsegments[model_idx] = self._build_pair_dict(
                model_params,
                bitmap_list,
                parameter_scores=parameter_scores,
            )
            self._attach_bn_payloads_to_pair_dict(model_idx, self._model_idx_to_bitmapANDsegments[model_idx])
        self._model_idx_to_bitmapANDsegments["method"] = "importance"
        return self._model_idx_to_bitmapANDsegments

    def _build_shared_random_bitmaps(self):
        assert self._seg_divided_num > 0, f"The number of segments from one model is {self._seg_divided_num}"
        return self._build_random_bitmaps()

    def _build_random_bitmaps(self):
        bit_idx_list = self._split_indices(np.random.permutation(self._parameters_length))
        return [create_bitmap(self._parameters_length, bit_idx) for bit_idx in bit_idx_list]

    def _build_consistent_bitmaps(self):
        assert self._seg_divided_num > 0, f"The number of segments from one model is {self._seg_divided_num}"
        bit_idx_list = self._split_indices(np.arange(self._parameters_length))
        return [create_bitmap(self._parameters_length, bit_idx) for bit_idx in bit_idx_list]

    def _split_indices(self, index_array):
        return split_parameter_indices(index_array, self._seg_divided_num)

    def _build_segments_from_shared_bitmaps(self, bitmap_list, method_name):
        self._outgoing_model_parameters = self._get_parameters_from_models(self._outgoing_model_dict)
        self._model_idx_to_bitmapANDsegments = {}
        for model_idx, model_params in self._outgoing_model_parameters.items():
            pair_dict = self._build_pair_dict(model_params, bitmap_list)
            self._model_idx_to_bitmapANDsegments[model_idx] = self._attach_bn_payloads_to_pair_dict(
                model_idx,
                pair_dict,
            )
        self._model_idx_to_bitmapANDsegments["method"] = method_name
        return self._model_idx_to_bitmapANDsegments

    def _build_pair_dict(self, model_params, bitmap_list, parameter_scores=None):
        segments = create_segments_based_bitmap(model_params, bitmap_list)
        return {
            f"pair{i}": {
                "bitmap": bitmap_list[i],
                "bitmap_bits": self._parameters_length,
                "bitmap_unit": "parameter",
                "parameter_seg": segment,
                "mean_value": self._parameter_segment_mean_value(
                    segment,
                    bitmap_list[i].nonzero(),
                    parameter_scores=parameter_scores,
                ),
            }
            for i, segment in enumerate(segments)
        }

    @staticmethod
    def _parameter_segment_mean_value(parameter_seg, parameter_idx, parameter_scores=None):
        if parameter_scores is not None and len(parameter_idx) > 0:
            return float(parameter_scores[parameter_idx].mean())
        if parameter_seg.numel() == 0:
            return 0.0
        return float(parameter_seg.abs().mean())

    def _attach_fisher_segments(self, fisher_weights_dict: dict):
        assert self._uses_fisher_scores(), "Fisher segments are only needed for Fisher score aggregation"
        if fisher_weights_dict is None:
            raise ValueError("Fisher score aggregation requires current_fisher_weights_dict")
        assert isinstance(self._model_idx_to_bitmapANDsegments, dict), "Segments must be created before Fisher attach"

        for model_idx, pair_dict in self._model_idx_to_bitmapANDsegments.items():
            if model_idx == "method":
                continue
            fisher_weights = fisher_weights_dict[model_idx]
            if self._uses_block_segments():
                self._attach_block_fisher_segments(pair_dict, fisher_weights)
            else:
                self._attach_parameter_fisher_segments(pair_dict, fisher_weights)

    def _attach_parameter_fisher_segments(self, pair_dict: dict, fisher_vector: torch.Tensor):
        assert isinstance(fisher_vector, torch.Tensor), "Parameter Fisher weights must be a tensor"
        assert fisher_vector.numel() == self._parameters_length, \
            "Parameter Fisher length does not match selected model parameter length"
        for pair_data in pair_dict.values():
            bitmap = pair_data["bitmap"]
            fisher_seg = fisher_vector[bitmap.nonzero()].detach().clone()
            pair_data["fisher_seg"] = fisher_seg
            pair_data["fisher_unit"] = "parameter"

    def _attach_block_fisher_segments(self, pair_dict: dict, fisher_weights):
        assert self._block_count is not None, "Block layout has not been prepared"
        if self._uses_block_fisher_scores():
            assert isinstance(fisher_weights, torch.Tensor), "Block Fisher weights must be a tensor"
            assert fisher_weights.numel() == self._block_count, \
                "Block Fisher length does not match block layout"
            for pair_data in pair_dict.values():
                block_idx = pair_data["block_idx"]
                fisher_seg = fisher_weights[block_idx].detach().clone()
                pair_data["fisher_seg"] = fisher_seg
                pair_data["fisher_unit"] = "block"
            return

        if isinstance(fisher_weights, torch.Tensor):
            packed_fisher = self._pack_parameter_vector(fisher_weights)
            for pair_data in pair_dict.values():
                parameter_idx = self._packed_parameter_indices_for_blocks(
                    pair_data["block_idx"]
                )
                parameter_idx = torch.as_tensor(
                    parameter_idx,
                    dtype=torch.long,
                    device=packed_fisher.device,
                )
                pair_data["fisher_seg"] = packed_fisher[parameter_idx].detach().clone()
                pair_data["fisher_unit"] = "parameter"
            return

        assert isinstance(fisher_weights, list), \
            "Block-segment parameter Fisher weights must be a tensor or block list"
        assert len(fisher_weights) == self._block_count, "Fisher block count does not match block layout"
        for pair_data in pair_dict.values():
            fisher_seg = create_segments_based_block_bitmap(fisher_weights, [pair_data["bitmap"]])[0]
            pair_data["fisher_seg"] = fisher_seg.detach().clone()
            pair_data["fisher_unit"] = "parameter"

    def _simulate_aggregation(self, current_scores_dict: dict):
        assert isinstance(self._model_idx_to_current_aggregating_data, dict), \
            "Invalid received segments, run communication and disposal first"
        assert isinstance(self._model_parameters_before_aggregation, dict), "invalid model parameters, get again"

        self._model_parameters_after_aggregation = dict()
        model_bn_after_aggregation = {}
        packed_parameter_idx_cache = {}
        for model_idx, aggregating_combo_list in self._model_idx_to_current_aggregating_data.items():
            to_be_aggregated_model_list = []
            score_list = []
            bitmap_list = []
            combo_block_idx_list = []
            seg_param_list = []
            fisher_seg_list = []
            bn_param_list = []
            for combo in aggregating_combo_list:
                to_be_aggregated_model_list.append(combo.idx)
                score_list.append(combo.score)
                bitmap_list.append(combo.bitmap)
                combo_block_idx_list.append(
                    combo.block_idx
                    if combo.block_idx is not None
                    else tuple(combo.bitmap.nonzero())
                )
                seg_param_list.append(combo.seg_param)
                if combo.fisher_seg is not None:
                    fisher_seg_list.append(combo.fisher_seg)
                if combo.bn_param is not None:
                    bn_param_list.append(combo.bn_param)
            log.info("Aggregating model %s: %s", model_idx, to_be_aggregated_model_list)
            model_current_score = current_scores_dict[model_idx]
            model_current_params = self._model_parameters_before_aggregation[model_idx]
            model_pre_aggregation_unit_l2 = (
                self._packed_unit_l2_norms(model_current_params)
                if self._unit_l2_mode != "none"
                else None
            )
            packed_parameter_idx_list = None
            received_block_idx_list = None
            if self._uses_block_segments() and isinstance(model_current_params, torch.Tensor):
                packed_parameter_idx_list = []
                received_block_idx_list = []
                for bitmap, block_idx in zip(bitmap_list, combo_block_idx_list):
                    bitmap_key = id(bitmap)
                    parameter_idx = packed_parameter_idx_cache.get(bitmap_key)
                    if parameter_idx is None:
                        parameter_idx = self._packed_parameter_indices_for_blocks(block_idx)
                        packed_parameter_idx_cache[bitmap_key] = parameter_idx
                    packed_parameter_idx_list.append(parameter_idx)
                    received_block_idx_list.append(block_idx)
            if self._uses_uniform_bn_aggregation():
                local_bn = self._get_bn_from_model(self._outgoing_model_dict[model_idx]).detach().clone()
                if local_bn.numel() > 0:
                    bn_values = [local_bn] + [
                        bn_param.to(device=local_bn.device, dtype=local_bn.dtype)
                        for bn_param in bn_param_list
                    ]
                    model_bn_after_aggregation[model_idx] = torch.stack(bn_values).mean(dim=0)
            if self._uses_fisher_scores():
                assert len(fisher_seg_list) == len(seg_param_list), \
                    "Fisher aggregation requires one Fisher payload for each received segment"
                if self._uses_block_segments():
                    assert isinstance(model_current_params, torch.Tensor), \
                        "Block simulation must use packed parameter tensors"
                    if self._uses_block_fisher_scores():
                        self._model_parameters_after_aggregation[model_idx] = \
                            aggregate_packed_segments_block_fisher_based(
                                model_current_params,
                                model_current_score,
                                self._block_sizes,
                                seg_param_list,
                                packed_parameter_idx_list,
                                received_block_idx_list,
                                fisher_seg_list,
                                exponential_base=self._aggregation_weight_exp_base,
                            )
                    else:
                        self._model_parameters_after_aggregation[model_idx] = \
                            aggregate_packed_segments_parameter_fisher_based(
                                model_current_params,
                                self._pack_parameter_vector(model_current_score),
                                seg_param_list,
                                packed_parameter_idx_list,
                                fisher_seg_list,
                                exponential_base=self._aggregation_weight_exp_base,
                            )
                else:
                    self._model_parameters_after_aggregation[model_idx] = aggregate_segment_fisher_based(
                        model_current_params,
                        model_current_score,
                        seg_param_list,
                        bitmap_list,
                        fisher_seg_list,
                        exponential_base=self._aggregation_weight_exp_base,
                    )
            elif self._uses_block_segments():
                assert isinstance(model_current_params, torch.Tensor), \
                    "Block simulation must use packed parameter tensors"
                self._model_parameters_after_aggregation[model_idx] = \
                    aggregate_packed_segments_scores_based(
                        model_current_params,
                        seg_param_list,
                        packed_parameter_idx_list,
                        score_list,
                        model_current_score,
                        exponential_base=self._aggregation_weight_exp_base,
                    )
            else:
                self._model_parameters_after_aggregation[model_idx] = self._aggregate_segment_scores_based(
                    model_current_params,
                    seg_param_list,
                    bitmap_list,
                    score_list,
                    model_current_score,
                )
            if self._aggregation_update_unit_l2_mode != "none":
                self._model_parameters_after_aggregation[model_idx] = (
                    self._bound_packed_update_unit_l2(
                        self._model_parameters_after_aggregation[model_idx],
                        model_current_params,
                        self._aggregation_update_unit_l2_multiplier,
                        "aggregation update",
                    )
                )
            if model_pre_aggregation_unit_l2 is not None:
                self._model_parameters_after_aggregation[model_idx] = (
                    self._restore_packed_unit_l2(
                        self._model_parameters_after_aggregation[model_idx],
                        model_pre_aggregation_unit_l2,
                        model_current_params,
                        mode=self._unit_l2_mode,
                        multiplier=self._unit_l2_multiplier,
                    )
                )
                if self._aggregation_update_unit_l2_mode != "none":
                    self._model_parameters_after_aggregation[model_idx] = (
                        self._fallback_packed_update_l2_violations(
                            self._model_parameters_after_aggregation[model_idx],
                            model_current_params,
                            self._aggregation_update_unit_l2_multiplier,
                        )
                    )


        if self._uses_block_segments():
            self._load_packed_parameters_to_models(
                self._total_model_dict,
                self._model_parameters_after_aggregation,
            )
        else:
            self._load_parameters_to_models(self._total_model_dict, self._model_parameters_after_aggregation)
        for model_idx, bn_params in model_bn_after_aggregation.items():
            self._load_bn_to_model(self._total_model_dict[model_idx], bn_params)

    def get_models(self) -> dict:
        return self._total_model_dict


class SegmentedMethodBase(FLMethodsSeg):
    def __init__(
            self,
            total_models_dict: dict,
            seg_divided_number=5,
            connectivity_dict=None,
            seg_create_method="consistent",
            seg_chosen_method="uniform",
            parameter_scope="all",
            segment_unit="parameter",
            block_grouping_method="none",
            block_group_size=1,
            channel_length=0,
            block_refinement=None,
            aggregation_score_source="uniform",
            fisher_block=False,
            include_other_blocks=False,
            bn_mode="affine",
            bn_process_as_base_unit=True,
            bn_aggregation_source="score",
            segment_importance_metric="weight_abs",
            importance_parameter_to_unit="mean_abs",
            importance_unit_to_group="mean",
            importance_group_to_segment="mean",
            importance_lipschitz_ema_enabled=False,
            group_enabled=False,
            group_criterion_metric="gradient_abs_ema_online",
            group_criterion_parameter_to_unit="mean_abs",
            group_criterion_lipschitz_ema_enabled=False,
            segment_compose_method="score_sorted_balanced_payload",
            segment_pick_exp_normalize=True,
            segment_pick_exp_base=None,
    ):
        super().__init__(
            total_models_dict,
            seg_divided_number,
            parameter_scope=parameter_scope,
            segment_unit=segment_unit,
            block_grouping_method=block_grouping_method,
            block_group_size=block_group_size,
            channel_length=channel_length,
            block_refinement=block_refinement,
            aggregation_score_source=aggregation_score_source,
            fisher_block=fisher_block,
            include_other_blocks=include_other_blocks,
            bn_mode=bn_mode,
            bn_process_as_base_unit=bn_process_as_base_unit,
            bn_aggregation_source=bn_aggregation_source,
            segment_importance_metric=segment_importance_metric,
            importance_parameter_to_unit=importance_parameter_to_unit,
            importance_unit_to_group=importance_unit_to_group,
            importance_group_to_segment=importance_group_to_segment,
            importance_lipschitz_ema_enabled=importance_lipschitz_ema_enabled,
            group_enabled=group_enabled,
            group_criterion_metric=group_criterion_metric,
            group_criterion_parameter_to_unit=group_criterion_parameter_to_unit,
            group_criterion_lipschitz_ema_enabled=group_criterion_lipschitz_ema_enabled,
            segment_compose_method=segment_compose_method,
            segment_pick_exp_normalize=segment_pick_exp_normalize,
            segment_pick_exp_base=segment_pick_exp_base,
        )
        self._validate_connectivity_dict(connectivity_dict)
        self._validate_segment_create_method(seg_create_method)
        self._validate_segment_chosen_method(seg_chosen_method)

        self._connectivity_dict = connectivity_dict
        self._seg_create_method = seg_create_method
        self._seg_chosen_method = seg_chosen_method

    @staticmethod
    def _validate_connectivity_dict(connectivity_dict):
        assert isinstance(connectivity_dict, dict), "Please give a valid connectivity dict"

    @staticmethod
    def _validate_segment_create_method(seg_create_method):
        assert seg_create_method in SEGMENT_CREATE_METHODS, (
            f"Not supported segments-divided method [{seg_create_method}], "
            f"only support {sorted(SEGMENT_CREATE_METHODS)}"
        )

    @staticmethod
    def _validate_segment_chosen_method(seg_chosen_method):
        assert seg_chosen_method in SEGMENT_CHOSEN_METHODS, (
            f"Not implemented method {seg_chosen_method}, "
            f"only support {sorted(SEGMENT_CHOSEN_METHODS)}"
        )

    def set_connectivity_dict(self, connectivity_dict):
        self._validate_connectivity_dict(connectivity_dict)
        self._connectivity_dict = connectivity_dict

    def _create_current_segments(
            self,
            current_selection_lipschitz_score_weights_dict=None,
            current_parameter_score_weights_dict=None,
            current_block_parameter_score_weights_dict=None,
            current_block_interaction_score_weights_dict=None,
    ):
        segment_builders = {
            "random_same": self._create_bitmap_segments_random_same,
            "random_each": self._create_bitmap_segments_random_each,
            "consistent": self._create_bitmap_segments_consistent,
            "importance": lambda: self._create_bitmap_segments_importance(
                current_selection_lipschitz_score_weights_dict,
                current_parameter_score_weights_dict,
                current_block_parameter_score_weights_dict,
                current_block_interaction_score_weights_dict,
            ),
        }
        self._model_idx_to_bitmapANDsegments = segment_builders[self._seg_create_method]()
        return self._model_idx_to_bitmapANDsegments

    def _build_score_dict(self, current_scores_dict, current_fisher_weights_dict=None):
        if self._uses_fisher_scores():
            if current_fisher_weights_dict is None:
                raise ValueError("Fisher score aggregation requires current_fisher_weights_dict")
            return current_fisher_weights_dict
        if self._aggregation_score_source == "val_acc":
            return dict(current_scores_dict)
        return {model_idx: 1.0 for model_idx in current_scores_dict.keys()}

    def _segment_aggregation_weight_elements(self, segment_data) -> int:
        aggregation_weight = segment_data.get("fisher_seg")
        if aggregation_weight is not None:
            return aggregation_weight.numel()
        if self._aggregation_score_source == "val_acc":
            return 1
        return 0

    def _record_segment_packet(
            self,
            *,
            source_idx,
            destination_idx,
            segment_id,
            current_global_round,
            current_local_round_dict,
            status,
            packet_kind,
            batch_norm_elements=None,
    ):
        segment_data = self._model_idx_to_bitmapANDsegments[source_idx][segment_id]
        if batch_norm_elements is None:
            batch_norm = segment_data.get("bn_param")
            batch_norm_elements = 0 if batch_norm is None else batch_norm.numel()
        payload = PacketPayload(
            model_parameter_elements=segment_data["parameter_seg"].numel(),
            aggregation_weight_elements=self._segment_aggregation_weight_elements(segment_data),
            batch_norm_elements=batch_norm_elements,
            bitmap_bits=segment_data.get("bitmap_bits", segment_data["bitmap"].size()),
        )
        selection_mode = {
            "segment_push": "push",
            "segment_pull_response": "pull",
        }[packet_kind]
        self._record_communication_packet(
            global_round=current_global_round,
            packet_kind=packet_kind,
            selection_mode=selection_mode,
            source_device=source_idx,
            destination_device=destination_idx,
            source_local_round=current_local_round_dict.get(source_idx),
            destination_local_round=current_local_round_dict.get(destination_idx),
            status=status,
            segment_id=segment_id,
            payload=payload,
        )

    def _recipient_once_bn_elements_for_attempt(
            self,
            source_idx,
            destination_idx,
            delivered_bn_pairs,
            bn_elements_cache,
    ):
        pair = (source_idx, destination_idx)
        if not self._uses_recipient_once_bn_payloads() or pair in delivered_bn_pairs:
            return None
        if source_idx not in bn_elements_cache:
            bn_elements_cache[source_idx] = self._get_bn_from_model(
                self._outgoing_model_dict[source_idx]
            ).numel()
        return bn_elements_cache[source_idx]

    def _attempt_segment_transfer(
            self,
            *,
            source_idx,
            destination_idx,
            segment_id,
            stability,
            current_global_round,
            current_local_round_dict,
            delivered_bn_pairs,
            bn_elements_cache,
    ) -> bool:
        batch_norm_elements = None
        if self._communication_recorder is not None:
            batch_norm_elements = self._recipient_once_bn_elements_for_attempt(
                source_idx,
                destination_idx,
                delivered_bn_pairs,
                bn_elements_cache,
            )

        delivered = np.random.rand() < stability
        if self._communication_recorder is not None:
            self._record_segment_packet(
                source_idx=source_idx,
                destination_idx=destination_idx,
                segment_id=segment_id,
                current_global_round=current_global_round,
                current_local_round_dict=current_local_round_dict,
                status="delivered" if delivered else "dropped",
                packet_kind=self._segment_packet_kind(),
                batch_norm_elements=batch_norm_elements,
            )
        if delivered and batch_norm_elements is not None:
            delivered_bn_pairs.add((source_idx, destination_idx))
        return delivered

    def _segment_packet_kind(self):
        raise NotImplementedError

    def _build_received_combo(
            self,
            source_idx,
            segment_id,
            current_global_round,
            current_local_round_dict,
            current_scores_dict,
            bn_param=None,
    ):
        seg_data = self._model_idx_to_bitmapANDsegments[source_idx][segment_id]
        effective_bn = bn_param if bn_param is not None else seg_data.get("bn_param")
        return Combo(
            idx=f"{source_idx}_{segment_id}_{current_global_round}_{current_local_round_dict[source_idx]}",
            score=1.0 if self._uses_fisher_scores() else current_scores_dict[source_idx],
            seg_param=seg_data["parameter_seg"],
            bitmap=seg_data["bitmap"],
            block_idx=seg_data.get("block_idx"),
            fisher_seg=seg_data.get("fisher_seg"),
            bn_param=effective_bn,
        )

    def _log_round_summary(self):
        log.info(
            "segment creating method is [%s], segment picked method is [%s], "
            "base unit is [%s], segment importance is [%s], parameter-to-unit reduction is [%s], "
            "channel length is [%s], "
            "block refinement is [%s], bitmap unit is [%s], bn mode is [%s], bn process mode is [%s], "
            "aggregation score source is [%s], fisher block is [%s], "
            "include other blocks is [%s], group enabled is [%s], group method is [%s], group size is [%s]",
            self._seg_create_method,
            self._seg_chosen_method,
            self._segment_unit,
            self._segment_importance_metric,
            self._importance_parameter_to_unit,
            self._channel_length,
            self._block_refinement,
            self._segment_bitmap_method,
            self._base_bn_mode,
            self._bn_process_mode,
            self._aggregation_score_source,
            self._uses_block_fisher_scores(),
            self._include_other_blocks,
            self._group_enabled,
            self._block_grouping_method,
            self._block_group_size,
        )

    def _log_transfer_summary(self):
        raise NotImplementedError

    def _simulate_communication_gossip(
            self,
            current_model_idx_list: list,
            current_global_round,
            current_local_round_dict,
    ):
        raise NotImplementedError

    def _dispose_communication_result(
            self,
            current_model_idx_list,
            current_global_round,
            current_local_round_dict,
            current_scores_dict,
    ):
        raise NotImplementedError

    def simulate_method(
            self,
            current_model_idx_list,
            current_global_round,
            current_local_round_dict,
            current_scores_dict,
            current_fisher_weights_dict=None,
            current_selection_lipschitz_score_weights_dict=None,
            current_parameter_score_weights_dict=None,
            current_block_parameter_score_weights_dict=None,
            current_block_interaction_score_weights_dict=None,
            aggregation_score_source=None,
            segment_importance_active=True,
    ):
        previous_aggregation_score_source = self._aggregation_score_source
        previous_segment_create_method = self._seg_create_method
        previous_segment_chosen_method = self._seg_chosen_method
        if aggregation_score_source is not None:
            self._aggregation_score_source = canonical_aggregation_score_source(
                aggregation_score_source
            )
        if not segment_importance_active:
            if self._seg_create_method == "importance":
                self._seg_create_method = "random_each"
            if self._seg_chosen_method == "probabilistic":
                self._seg_chosen_method = "uniform"

        try:
            if (
                    self._aggregation_score_source
                    in BLOCK_ALIGNED_AGGREGATION_SCORE_SOURCES
                    and not self._uses_block_segments()
            ):
                raise ValueError(
                    "method.aggregation.weight.metric="
                    f"{self._aggregation_score_source} requires "
                    "method.partition.unit to use a block unit"
                )
            self._log_round_summary()
            self._log_transfer_summary()
            self._create_current_segments(
                current_selection_lipschitz_score_weights_dict,
                current_parameter_score_weights_dict,
                current_block_parameter_score_weights_dict,
                current_block_interaction_score_weights_dict,
            )
            assert self._outgoing_model_parameters is not None
            self._model_parameters_before_aggregation = self._outgoing_model_parameters
            if self._uses_fisher_scores():
                self._attach_fisher_segments(current_fisher_weights_dict)
            score_dict = self._build_score_dict(
                current_scores_dict,
                current_fisher_weights_dict,
            )

            self._simulate_communication_gossip(
                current_model_idx_list,
                current_global_round,
                current_local_round_dict,
            )
            self._dispose_communication_result(
                current_model_idx_list,
                current_global_round,
                current_local_round_dict,
                score_dict,
            )
            self._simulate_aggregation(score_dict)
        finally:
            self._aggregation_score_source = previous_aggregation_score_source
            self._seg_create_method = previous_segment_create_method
            self._seg_chosen_method = previous_segment_chosen_method
