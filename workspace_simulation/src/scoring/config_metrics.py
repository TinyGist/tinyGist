"""Strict, user-facing metric names used by configuration schema v2.

The scoring implementation still uses compact internal identifiers.  Keeping
the translation here makes temporal lifetime part of the configured metric
name and removes the need for public EMA/reset switches.
"""

from dataclasses import dataclass

from .definitions import (
    DIRECT_PARAMETER_SCORE_METHOD,
    FISHER_DIAGONAL_EMA_ONLINE_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
    FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
    GRADIENT_ABS_EMA_ONLINE_METHOD,
    GRADIENT_ABS_ROUND_STEP_EMA_ONLINE_METHOD,
    GRADIENT_ABS_POST_METHOD,
    GRADIENT_SIGNAL_PRESERVATION_POST_METHOD,
    GRADIENT_WEIGHT_ABS_POST_METHOD,
    HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD,
    HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD,
    HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    HUTCHINSON_DIAGONAL_POST_METHOD,
    HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD,
    TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD,
    TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD,
    TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD,
    WEIGHT_ABS_EMA_ONLINE_METHOD,
)


@dataclass(frozen=True)
class ImportanceMetric:
    name: str
    internal_name: str
    reset_each_round: bool = False
    lipschitz_ema: bool = False

    @property
    def is_block_metric(self) -> bool:
        return self.internal_name in {
            "mean_abs",
            "rms",
            "lipschitz",
            "fisher_lipschitz_cooperation",
        }


def _metric(name, internal_name, *, reset=False, lipschitz_ema=False):
    return ImportanceMetric(
        name=name,
        internal_name=internal_name,
        reset_each_round=reset,
        lipschitz_ema=lipschitz_ema,
    )


# Names deliberately describe both the mathematical quantity and its temporal
# accumulation.  Only these exact names are accepted by schema v2.
IMPORTANCE_METRICS = {
    metric.name: metric
    for metric in (
        _metric("parameter_magnitude.current", DIRECT_PARAMETER_SCORE_METHOD),
        _metric("parameter_magnitude.round_step_ema", WEIGHT_ABS_EMA_ONLINE_METHOD, reset=True),
        _metric("parameter_magnitude.cross_round_step_ema", WEIGHT_ABS_EMA_ONLINE_METHOD),
        _metric(
            "gradient_magnitude.round_step_ema",
            GRADIENT_ABS_ROUND_STEP_EMA_ONLINE_METHOD,
            reset=True,
        ),
        _metric("gradient_magnitude.cross_round_step_ema", GRADIENT_ABS_EMA_ONLINE_METHOD),
        _metric("fisher_diagonal.round_step_ema", FISHER_DIAGONAL_EMA_ONLINE_METHOD, reset=True),
        _metric("fisher_diagonal.cross_round_step_ema", FISHER_DIAGONAL_EMA_ONLINE_METHOD),
        _metric("gradient_magnitude.round_sample_mean", GRADIENT_ABS_POST_METHOD, reset=True),
        _metric("gradient_weight_magnitude.round_sample_mean", GRADIENT_WEIGHT_ABS_POST_METHOD, reset=True),
        _metric("gradient_signal_preservation.round", GRADIENT_SIGNAL_PRESERVATION_POST_METHOD, reset=True),
        _metric("empirical_fisher_diagonal.round_sample_mean", FISHER_EMPIRICAL_DIAGONAL_POST_METHOD, reset=True),
        _metric("empirical_fisher_weighted.round_sample_mean", FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD, reset=True),
        _metric("hutchinson_diagonal.round_estimate", HUTCHINSON_DIAGONAL_POST_METHOD, reset=True),
        _metric("hutchinson_weighted.round_estimate", HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD, reset=True),
        _metric("taylor_first.round_step_ema", TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD, reset=True),
        _metric("taylor_first.cross_round_step_ema", TAYLOR_FIRST_ABS_STEP_EMA_ONLINE_METHOD),
        _metric("taylor_first_current.round_step_ema", TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD, reset=True),
        _metric("taylor_first_current.cross_round_step_ema", TAYLOR_FIRST_ABS_CURRENT_ONLINE_METHOD),
        _metric("taylor_second.round_step_ema", TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD, reset=True),
        _metric("taylor_second.cross_round_step_ema", TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD),
        _metric("fisher_taylor_current.round_step_ema", FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD, reset=True),
        _metric("fisher_taylor_current.cross_round_step_ema", FISHER_TAYLOR_SECOND_CURRENT_ONLINE_METHOD),
        _metric("hessian_taylor_exact.round", HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD, reset=True),
        _metric("hessian_taylor.round_step_ema", HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD, reset=True),
        _metric("hessian_taylor.cross_round_step_ema", HESSIAN_TAYLOR_SECOND_STEP_EMA_ONLINE_METHOD),
        _metric("hessian_taylor_current.round_step_ema", HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD, reset=True),
        _metric("hessian_taylor_current.cross_round_step_ema", HESSIAN_TAYLOR_SECOND_CURRENT_ONLINE_METHOD),
        _metric("block_magnitude.mean_absolute", "mean_abs"),
        _metric("block_magnitude.root_mean_square", "rms"),
        _metric("block_lipschitz.current", "lipschitz"),
        _metric("block_lipschitz.cross_round_ema", "lipschitz", lipschitz_ema=True),
        _metric("fisher_lipschitz_interaction.round_step_ema", "fisher_lipschitz_cooperation", reset=True),
        _metric("fisher_lipschitz_interaction.cross_round_step_ema", "fisher_lipschitz_cooperation"),
    )
}


def resolve_importance_metric(value, field_name="metric") -> ImportanceMetric:
    name = str(value).strip().lower()
    try:
        return IMPORTANCE_METRICS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {field_name} [{value}]. Supported values are "
            f"{sorted(IMPORTANCE_METRICS)}"
        ) from exc


@dataclass(frozen=True)
class AggregationMetric:
    name: str
    source: str
    fisher_calculation: str = "square"
    reset_each_round: bool = False
    lipschitz_ema: bool = False


AGGREGATION_METRICS = {
    metric.name: metric
    for metric in (
        AggregationMetric("uniform", "uniform"),
        AggregationMetric("validation_accuracy", "val_acc"),
        AggregationMetric("fisher_diagonal.round_step_ema", "fisher", reset_each_round=True),
        AggregationMetric("fisher_diagonal.cross_round_step_ema", "fisher"),
        AggregationMetric(
            "gradient_magnitude.round_step_ema",
            "fisher",
            fisher_calculation="abs",
            reset_each_round=True,
        ),
        AggregationMetric(
            "gradient_magnitude.cross_round_step_ema",
            "fisher",
            fisher_calculation="abs",
        ),
        AggregationMetric(
            "fisher_lipschitz_interaction.round_step_ema",
            "fisher_lipschitz",
            reset_each_round=True,
        ),
        AggregationMetric(
            "fisher_lipschitz_interaction.cross_round_step_ema",
            "fisher_lipschitz",
        ),
        AggregationMetric(
            "fisher_lipschitz_interaction.cross_round_ema",
            "fisher_lipschitz",
            lipschitz_ema=True,
        ),
    )
}


def resolve_aggregation_metric(value, field_name="metric") -> AggregationMetric:
    name = str(value).strip().lower()
    try:
        return AGGREGATION_METRICS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {field_name} [{value}]. Supported values are "
            f"{sorted(AGGREGATION_METRICS)}"
        ) from exc
