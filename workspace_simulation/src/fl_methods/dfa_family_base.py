from collections import deque
import logging

import numpy as np

from .base import SegmentedMethodBase
from .recipient_selection import RecipientSelector


log = logging.getLogger(__name__)


class SegmentedDFAMethod(SegmentedMethodBase):
    def __init__(
            self, total_models_dict: dict, seg_divided_number=5, connectivity_dict=None,
            seg_create_method="consistent", seg_chosen_method="uniform", seg_sending_num=7,
            largest_seg_num=10, aggregating_seg_threshold=5,
            parameter_scope="all", segment_unit="parameter",
            block_grouping_method="none", block_group_size=1,
            channel_length=0,
            block_refinement=None,
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
            recipient_pick_method="uniform_with_replacement",
            recipient_balance_strength=1.0,
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
        self._seg_sending_num = seg_sending_num
        self._recipient_selector = RecipientSelector(
            pick_method=recipient_pick_method,
            balance_strength=recipient_balance_strength,
        )
        self._aggregating_seg_threshold = aggregating_seg_threshold
        self._model_idx_to_stored_combo_dict = {
            model_idx: deque(maxlen=largest_seg_num)
            for model_idx in total_models_dict.keys()
        }

    def _log_transfer_summary(self):
        log.info(
            "Model is split into %s segments, each working model in this round sends %s segments, "
            "recipient pick method is [%s]",
            self._seg_divided_num,
            self._seg_sending_num,
            self._recipient_selector.pick_method,
        )

    def _segment_packet_kind(self):
        return "segment_push"

    def _simulate_communication_gossip(
            self,
            current_model_idx_list: list,
            current_global_round,
            current_local_round_dict,
    ):
        assert isinstance(self._model_idx_to_bitmapANDsegments, dict), "Not get valid bitmap yet"

        self._model_idx_to_received_segments = {
            model_idx: {target_idx: [] for target_idx in target_idx_dict.keys()}
            for model_idx, target_idx_dict in self._connectivity_dict.items()
        }
        delivered_bn_pairs = set()
        bn_elements_cache = {}

        for model_idx, target_idx_dict in self._connectivity_dict.items():
            if model_idx not in current_model_idx_list:
                continue
            target_idx_list = list(target_idx_dict.keys())
            chosen_segment_list = self._pick_segments_for_sender(model_idx)
            chosen_target_list = self._recipient_selector.select(
                model_idx,
                chosen_segment_list,
                target_idx_list,
            )

            for target_model_idx, chosen_segment_id in zip(chosen_target_list, chosen_segment_list):
                stability = self._connectivity_dict[model_idx][target_model_idx]
                if not self._attempt_segment_transfer(
                        source_idx=model_idx,
                        destination_idx=target_model_idx,
                        segment_id=chosen_segment_id,
                        stability=stability,
                        current_global_round=current_global_round,
                        current_local_round_dict=current_local_round_dict,
                        delivered_bn_pairs=delivered_bn_pairs,
                        bn_elements_cache=bn_elements_cache,
                ):
                    continue
                self._model_idx_to_received_segments[target_model_idx][model_idx].append(chosen_segment_id)

        self._remove_empty_received_segments()

        log.info("Allocated segments after gossip with bitmap: %s", self._model_idx_to_received_segments)
        return self._model_idx_to_received_segments

    def _pick_segments_for_sender(self, model_idx):
        pair_idx_list = list(self._model_idx_to_bitmapANDsegments[model_idx].keys())
        if self._seg_sending_num <= 0 or not pair_idx_list:
            return []
        if self._seg_chosen_method == "probabilistic":
            return self._pick_segments_based_probabilistic(
                model_idx,
                self._seg_sending_num,
            )
        pair_idx_array = np.asarray(pair_idx_list, dtype=object)
        return np.random.choice(pair_idx_array, size=self._seg_sending_num, replace=True).tolist()

    def _remove_empty_received_segments(self):
        for model_idx, target_model_dict in self._model_idx_to_received_segments.items():
            self._model_idx_to_received_segments[model_idx] = {
                target_model_idx: target_seg_list
                for target_model_idx, target_seg_list in target_model_dict.items()
                if len(target_seg_list) > 0
            }

    def _dispose_communication_result(
            self,
            current_model_idx_list,
            current_global_round,
            current_local_round_dict,
            current_scores_dict,
    ):
        for model_idx, target_idx_dict in self._model_idx_to_received_segments.items():
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
                    self._model_idx_to_stored_combo_dict[model_idx].append(combo)

        self._model_idx_to_current_aggregating_data = dict()
        for model_idx in current_model_idx_list:
            stored_combos = self._model_idx_to_stored_combo_dict[model_idx]
            if len(stored_combos) >= self._aggregating_seg_threshold:
                self._model_idx_to_current_aggregating_data[model_idx] = list(stored_combos)
                stored_combos.clear()

        return self._model_idx_to_current_aggregating_data
