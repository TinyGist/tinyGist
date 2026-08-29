"""Internal scoring identifiers and strict validators.

User-facing schema-v2 metric names live in ``config_metrics``.  Internal APIs
accept only the canonical identifiers below; legacy aliases are intentionally
not maintained.
"""

DIRECT_PARAMETER_SCORE_METHOD = "weight_abs"
WEIGHT_ABS_EMA_ONLINE_METHOD = "weight_abs_ema_online"
GRADIENT_ABS_EMA_ONLINE_METHOD = "gradient_abs_ema_online"
GRADIENT_ABS_ROUND_STEP_EMA_ONLINE_METHOD = "gradient_abs_round_step_ema_online"
FISHER_DIAGONAL_EMA_ONLINE_METHOD = "fisher_diagonal_ema_online"
GRADIENT_ABS_POST_METHOD = "gradient_abs_post"
GRADIENT_WEIGHT_ABS_POST_METHOD = "gradient_weight_abs_post"
GRADIENT_SIGNAL_PRESERVATION_POST_METHOD = "gradient_signal_preservation_post"
FISHER_EMPIRICAL_DIAGONAL_POST_METHOD = "fisher_empirical_diagonal_post"
FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD = "fisher_empirical_diagonal_weight_post"
HUTCHINSON_DIAGONAL_POST_METHOD = "hutchinson_diagonal_post"
HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD = "hutchinson_diagonal_weight_post"
TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD = "taylor_first_abs_step_ema_online"
TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD = "taylor_first_abs_current_online"
TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD = "taylor_second_step_ema_online"
FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD = "fisher_taylor_second_current_online"
HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD = "hessian_taylor_second_exact_post"
HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD = "hessian_taylor_second_step_ema_online"
HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD = "hessian_taylor_second_current_online"

PARAMETER_SCORE_METHODS = {
    DIRECT_PARAMETER_SCORE_METHOD,
    WEIGHT_ABS_EMA_ONLINE_METHOD,
    GRADIENT_ABS_EMA_ONLINE_METHOD,
    GRADIENT_ABS_ROUND_STEP_EMA_ONLINE_METHOD,
    FISHER_DIAGONAL_EMA_ONLINE_METHOD,
    GRADIENT_ABS_POST_METHOD,
    GRADIENT_WEIGHT_ABS_POST_METHOD,
    GRADIENT_SIGNAL_PRESERVATION_POST_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
    HUTCHINSON_DIAGONAL_POST_METHOD,
    HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD,
    TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD,
    TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD,
    TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
    HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD,
    HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
}

PARAMETER_SCORE_RECORD_DEPENDENCIES = {
    TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD: (GRADIENT_ABS_EMA_ONLINE_METHOD,),
    FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD: (FISHER_DIAGONAL_EMA_ONLINE_METHOD,),
}

HESSIAN_PARAMETER_SCORE_METHODS = {
    HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD,
    HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
}

HESSIAN_EMA_PARAMETER_SCORE_METHODS = {
    HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
}

POST_TRAINING_PARAMETER_SCORE_METHODS = {
    GRADIENT_ABS_POST_METHOD,
    GRADIENT_WEIGHT_ABS_POST_METHOD,
    GRADIENT_SIGNAL_PRESERVATION_POST_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
    HUTCHINSON_DIAGONAL_POST_METHOD,
    HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD,
    HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD,
}

BUFFERED_PARAMETER_SCORE_METHODS = (
    PARAMETER_SCORE_METHODS
    - {DIRECT_PARAMETER_SCORE_METHOD}
    - set(PARAMETER_SCORE_RECORD_DEPENDENCIES)
)

BLOCK_GROUPING_METHODS = {"none", "sensitivity_aligned", "sensitivity_diverse"}
SCORE_PROJECT_METHODS = {"mean_abs", "l2", "rms"}
SCORE_COMBINE_METHODS = {"mean", "sum", "max", "l2", "rms"}


def _strict_value(value, supported_values, field_name):
    value = str(value).strip().lower()
    if value not in supported_values:
        raise ValueError(
            f"Invalid {field_name} [{value}], supported values are {sorted(supported_values)}"
        )
    return value


def canonical_parameter_score_method(parameter_score_method: str) -> str:
    return _strict_value(
        parameter_score_method,
        PARAMETER_SCORE_METHODS,
        "parameter score method",
    )


def canonical_score_project_method(project_method: str) -> str:
    return _strict_value(project_method, SCORE_PROJECT_METHODS, "score project")


def canonical_score_combine_method(combine_method: str) -> str:
    return _strict_value(combine_method, SCORE_COMBINE_METHODS, "score combine")


def parameter_score_record_dependencies(parameter_score_method: str) -> list[str]:
    parameter_score_method = canonical_parameter_score_method(parameter_score_method)
    if parameter_score_method == DIRECT_PARAMETER_SCORE_METHOD:
        return []
    return list(
        PARAMETER_SCORE_RECORD_DEPENDENCIES.get(
            parameter_score_method,
            (parameter_score_method,),
        )
    )


def canonical_block_grouping_method(block_grouping_method: str) -> str:
    return _strict_value(
        block_grouping_method,
        BLOCK_GROUPING_METHODS,
        "block grouping method",
    )


def canonical_block_group_size(block_group_size) -> int:
    block_group_size = int(block_group_size)
    if block_group_size <= 0:
        raise ValueError("block grouping size must be a positive integer")
    return block_group_size
