from . import METHODS


def _common_kwargs(method_config, total_models_dict):
    return {
        "total_models_dict": total_models_dict,
        "parameter_scope": method_config.parameter_scope,
        "aggregation_score_source": method_config.aggregation_score_source,
        "bn_mode": method_config.bn_mode,
        "bn_process_as_base_unit": method_config.bn_process_mode,
        "bn_aggregation_source": method_config.bn_aggregation_source,
    }


def _segmented_kwargs(method_config, total_models_dict, connectivity_dict):
    return {
        **_common_kwargs(method_config, total_models_dict),
        "connectivity_dict": connectivity_dict,
        "segment_unit": method_config.segment_unit,
        "block_grouping_method": method_config.block_grouping_method,
        "block_group_size": method_config.block_group_size,
        "channel_length": method_config.channel_length,
        "block_refinement": method_config.block_refinement,
        "fisher_block": method_config.fisher_block,
        "segment_importance_metric": method_config.segment_importance_metric,
        "importance_parameter_to_unit": method_config.importance_parameter_to_unit,
        "importance_unit_to_group": method_config.importance_unit_to_group,
        "importance_group_to_segment": method_config.importance_group_to_segment,
        "importance_lipschitz_ema_enabled": method_config.selection_lipschitz_score_ema_enabled,
        "group_enabled": method_config.group_enabled,
        "group_criterion_metric": method_config.group_criterion_metric,
        "group_criterion_parameter_to_unit": method_config.group_criterion_parameter_to_unit,
        "group_criterion_lipschitz_ema_enabled": (
            method_config.group_criterion_lipschitz_score_ema_enabled
        ),
        "segment_compose_method": method_config.segment_compose_method,
        "segment_pick_exp_normalize": method_config.segment_pick_exp_normalize,
        "segment_pick_exp_base": method_config.segment_pick_exp_base,
    }


def _push_recipient_kwargs(method_config):
    return {
        "recipient_pick_method": method_config.recipient_pick_method,
        "recipient_balance_strength": method_config.recipient_balance_strength,
    }


def _build_centralized(method_class, method_config, total_models_dict, connectivity_dict):
    return method_class(
        **_common_kwargs(method_config, total_models_dict),
    )


def _build_segment_pulling(method_class, method_config, total_models_dict, connectivity_dict):
    return method_class(
        **_segmented_kwargs(method_config, total_models_dict, connectivity_dict),
        seg_divided_number=method_config.segment_divided_number,
        seg_pulling_num=method_config.segment_communicating_number,
        seg_create_method=method_config.segment_create_method,
        seg_chosen_method=method_config.segment_chosen_method,
    )


def _build_dfa(method_class, method_config, total_models_dict, connectivity_dict):
    return method_class(
        **_segmented_kwargs(method_config, total_models_dict, connectivity_dict),
        **_push_recipient_kwargs(method_config),
        seg_sending_num=method_config.segment_communicating_number,
        largest_seg_num=method_config.largest_seg_stored_num,
        aggregating_seg_threshold=method_config.aggregating_threshold,
    )


def _build_sdfa(method_class, method_config, total_models_dict, connectivity_dict):
    return method_class(
        **_segmented_kwargs(method_config, total_models_dict, connectivity_dict),
        **_push_recipient_kwargs(method_config),
        seg_divided_number=method_config.segment_divided_number,
        seg_sending_num=method_config.segment_communicating_number,
        seg_create_method=method_config.segment_create_method,
        seg_chosen_method=method_config.segment_chosen_method,
        largest_seg_num=method_config.largest_seg_stored_num,
        aggregating_seg_threshold=method_config.aggregating_threshold,
    )


def _build_gist(method_class, method_config, total_models_dict, connectivity_dict):
    return method_class(
        **_segmented_kwargs(method_config, total_models_dict, connectivity_dict),
        **_push_recipient_kwargs(method_config),
        seg_divided_number=method_config.segment_divided_number,
        seg_sending_num=method_config.segment_communicating_number,
        largest_seg_num=method_config.largest_seg_stored_num,
        aggregating_seg_threshold=method_config.aggregating_threshold,
    )


METHOD_BUILDERS = {
    "Centralized": _build_centralized,
    "SegmentPulling": _build_segment_pulling,
    "DFA": _build_dfa,
    "SDFA": _build_sdfa,
    "Gist": _build_gist,
}


def build_fl_method(method_config, total_models_dict, connectivity_dict):
    method_name = method_config.fl_method_name
    method_class = METHODS[method_name]
    try:
        builder = METHOD_BUILDERS[method_name]
    except KeyError as exc:
        raise NotImplementedError(f"FL method {method_name} is not implemented.") from exc
    method_instance = builder(method_class, method_config, total_models_dict, connectivity_dict)
    if method_config.aggregation_weight_exp_normalize:
        method_instance.set_aggregation_weight_exponential(
            True,
            method_config.aggregation_weight_exp_base,
        )
    if method_config.local_update_unit_l2_mode != "none":
        method_instance.set_local_update_unit_l2_constraint(
            method_config.local_update_unit_l2_mode,
            method_config.local_update_unit_l2_multiplier,
        )
    if method_config.aggregation_update_unit_l2_mode != "none":
        method_instance.set_aggregation_update_unit_l2_constraint(
            method_config.aggregation_update_unit_l2_mode,
            method_config.aggregation_update_unit_l2_multiplier,
        )
    if method_config.unit_l2_mode != "none":
        method_instance.set_unit_l2_constraint(
            method_config.unit_l2_mode,
            method_config.unit_l2_multiplier,
        )
    return method_instance


def run_method(
        method_instance,
        current_trainable_list,
        current_round,
        current_round_dict,
        current_scores_dict,
        current_fisher_weights_dict=None,
        current_selection_lipschitz_score_weights_dict=None,
        current_parameter_score_weights_dict=None,
        current_block_parameter_score_weights_dict=None,
        current_block_interaction_score_weights_dict=None,
        aggregation_score_source=None,
        segment_importance_active=True,
):
    method_instance.simulate_method(
        current_model_idx_list=current_trainable_list,
        current_global_round=current_round,
        current_local_round_dict=current_round_dict,
        current_scores_dict=current_scores_dict,
        current_fisher_weights_dict=current_fisher_weights_dict,
        current_selection_lipschitz_score_weights_dict=current_selection_lipschitz_score_weights_dict,
        current_parameter_score_weights_dict=current_parameter_score_weights_dict,
        current_block_parameter_score_weights_dict=current_block_parameter_score_weights_dict,
        current_block_interaction_score_weights_dict=current_block_interaction_score_weights_dict,
        aggregation_score_source=aggregation_score_source,
        segment_importance_active=segment_importance_active,
    )
    return method_instance.get_models()
