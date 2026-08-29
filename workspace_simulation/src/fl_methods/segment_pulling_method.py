import logging

import numpy as np

from .base import SegmentedMethodBase


log = logging.getLogger(__name__)


class SegmentPulling(SegmentedMethodBase):
    def __init__(
            self, total_models_dict: dict, seg_divided_number=5, connectivity_dict=None,
            seg_create_method="consistent", seg_chosen_method="uniform", seg_pulling_num=2,
            parameter_scope="all", segment_unit="parameter",
            channel_length=0,
            block_refinement=None,
            block_grouping_method="none", block_group_size=1,
            aggregation_score_source="uniform", fisher_block=False,
            include_other_blocks=False, bn_mode="affine",
            bn_process_as_base_unit=True, bn_aggregation_source="score",
            segment_importance_metric="weight_abs", importance_parameter_to_unit="mean_abs",
            importance_unit_to_group="mean", importance_group_to_segment="mean",
            importance_lipschitz_ema_enabled=False,
            group_enabled=False, group_criterion_metric="gradient_abs_ema_online",
            group_criterion_parameter_to_unit="mean_abs",
            group_criterion_lipschitz_ema_enabled=False,
            segment_compose_method="score_sorted_balanced_payload",
            segment_pick_exp_normalize=True,
            segment_pick_exp_base=None,
    ):
        super().__init__(
            total_models_dict=total_models_dict,
            seg_divided_number=seg_divided_number,
            connectivity_dict=connectivity_dict,
            seg_create_method=seg_create_method,
            seg_chosen_method=seg_chosen_method,
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
        assert seg_chosen_method == "uniform", (
            f"Not implemented method {seg_chosen_method}, Only support [uniform(random)]"
        )
        self._seg_pulling_num = seg_pulling_num

    def _log_transfer_summary(self):
        log.info(
            "Model is split into %s segments, each working model in this round receives %s segments",
            self._seg_divided_num,
            self._seg_pulling_num * self._seg_divided_num,
        )

    def _segment_packet_kind(self):
        return "segment_pull_response"

    def _simulate_communication_gossip(
            self,
            current_model_idx_list: list,
            current_global_round,
            current_local_round_dict,
    ):
        assert isinstance(self._model_idx_to_bitmapANDsegments, dict), "Not get valid bitmap yet"

        self._model_idx_to_received_segments = dict()
        delivered_bn_pairs = set()
        bn_elements_cache = {}
        for model_idx, target_idx_dict in self._connectivity_dict.items():
            if model_idx not in current_model_idx_list:
                continue
            if not target_idx_dict:
                self._model_idx_to_received_segments[model_idx] = {}
                continue
            num_total_targets = self._seg_pulling_num * self._seg_divided_num
            chosen_targets_array = np.random.choice(len(target_idx_dict), num_total_targets, replace=True)
            chosen_segment_idx_array = np.repeat(np.arange(self._seg_divided_num), self._seg_pulling_num)

            target_idx_list = list(target_idx_dict.keys())
            self._model_idx_to_received_segments[model_idx] = {
                target_idx: []
                for target_idx in target_idx_list
            }

            for chosen_model_idx, chosen_segment_idx in zip(chosen_targets_array, chosen_segment_idx_array):
                chosen_model_idx = target_idx_list[chosen_model_idx]
                segment_id = f"pair{chosen_segment_idx}"
                stability = self._connectivity_dict[model_idx][chosen_model_idx]
                if not self._attempt_segment_transfer(
                        source_idx=chosen_model_idx,
                        destination_idx=model_idx,
                        segment_id=segment_id,
                        stability=stability,
                        current_global_round=current_global_round,
                        current_local_round_dict=current_local_round_dict,
                        delivered_bn_pairs=delivered_bn_pairs,
                        bn_elements_cache=bn_elements_cache,
                ):
                    continue
                self._model_idx_to_received_segments[model_idx][chosen_model_idx].append(segment_id)

            self._model_idx_to_received_segments[model_idx] = {
                target_idx: target_seg_list
                for target_idx, target_seg_list in self._model_idx_to_received_segments[model_idx].items()
                if len(target_seg_list) > 0
            }

        log.info("Allocated segments after gossip: %s", self._model_idx_to_received_segments)
        return self._model_idx_to_received_segments

    def _dispose_communication_result(
            self,
            current_model_idx_list,
            current_global_round,
            current_local_round_dict,
            current_scores_dict,
    ):
        self._model_idx_to_current_aggregating_data = dict()
        for model_idx, target_idx_dict in self._model_idx_to_received_segments.items():
            self._model_idx_to_current_aggregating_data[model_idx] = []
            for target_idx, target_seg_list in target_idx_dict.items():
                bn_payload = (
                    self._bn_payload_for_model(target_idx)
                    if self._uses_recipient_once_bn_payloads()
                    else None
                )
                for segment_position, target_seg in enumerate(target_seg_list):
                    combo = self._build_received_combo(
                        target_idx,
                        target_seg,
                        current_global_round,
                        current_local_round_dict,
                        current_scores_dict,
                        bn_param=bn_payload if segment_position == 0 else None,
                    )
                    self._model_idx_to_current_aggregating_data[model_idx].append(combo)

        return self._model_idx_to_current_aggregating_data
