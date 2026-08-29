from dataclasses import dataclass, field

from src.scoring.config_metrics import ImportanceMetric, resolve_importance_metric


REFERENCE_SCORE_NAMES = (
    "gradient_magnitude.round_sample_mean",
    "gradient_weight_magnitude.round_sample_mean",
    "empirical_fisher_diagonal.round_sample_mean",
    "empirical_fisher_weighted.round_sample_mean",
    "hutchinson_diagonal.round_estimate",
    "hutchinson_weighted.round_estimate",
)
MEASUREMENT_NAMES = ("spearman", "top_k_overlap", "top_k_jaccard")
IMPORTANCE_CORRELATION_STAGES = {"after_training", "after_aggregation"}
IMPORTANCE_CORRELATION_KEYS = {
    "stage",
    "comparison_baseline",
    "evaluation_rounds",
    "reference_scores",
    "measurements",
    "hutchinson",
}


@dataclass(frozen=True)
class HutchinsonProbeConfig:
    probe_vectors: int = 5
    batch_limit: int = 0


@dataclass(frozen=True)
class ImportanceCorrelationProbeConfig:
    stage: str
    comparison_baseline: ImportanceMetric
    comparison_baseline_source: str
    reference_scores: tuple[ImportanceMetric, ...]
    measurements: tuple[str, ...]
    top_k: tuple[float, ...]
    evaluation_rounds: tuple[int, ...] | None
    parameter_scope: str
    bn_mode: str
    hutchinson: HutchinsonProbeConfig = field(default_factory=HutchinsonProbeConfig)

    @property
    def required_internal_score_methods(self) -> tuple[str, ...]:
        methods = [self.comparison_baseline.internal_name]
        methods.extend(metric.internal_name for metric in self.reference_scores)
        return tuple(dict.fromkeys(methods))

    @property
    def round_reset_parameter_score_methods(self) -> tuple[str, ...]:
        metrics = (self.comparison_baseline, *self.reference_scores)
        return tuple(dict.fromkeys(
            metric.internal_name for metric in metrics if metric.reset_each_round
        ))

    def should_evaluate(self, round_idx: int, stage: str | None = None) -> bool:
        round_matches = (
            self.evaluation_rounds is None or round_idx in self.evaluation_rounds
        )
        return round_matches and (stage is None or stage == self.stage)


@dataclass(frozen=True)
class ExperimentProbeConfig:
    importance_correlation: ImportanceCorrelationProbeConfig | None = None


def load_experiment_probe_config(settings) -> ExperimentProbeConfig:
    section = _probe_section(settings.parser)
    unknown = sorted(set(section) - {"importance_correlation"})
    if unknown:
        raise ValueError(f"Unsupported key(s) in probes: {unknown}")

    correlation_section = section.get("importance_correlation")
    if correlation_section is None:
        return ExperimentProbeConfig()
    if not isinstance(correlation_section, dict):
        raise ValueError("probes.importance_correlation must be a mapping or null")
    return ExperimentProbeConfig(
        importance_correlation=_importance_correlation_config(
            correlation_section,
            settings,
        ),
    )


def _probe_section(snapshot) -> dict:
    data = snapshot.source_data or {}
    section = data.get("probes") or {}
    if not isinstance(section, dict):
        raise ValueError("probes must be a mapping")
    return section


def _importance_correlation_config(section: dict, settings) -> ImportanceCorrelationProbeConfig:
    unknown = sorted(set(section) - IMPORTANCE_CORRELATION_KEYS)
    if unknown:
        raise ValueError(
            f"Unsupported key(s) in probes.importance_correlation: {unknown}"
        )
    if settings.fl_method.segment_unit != "parameter":
        raise ValueError(
            "probes.importance_correlation currently requires method.partition.unit=parameter"
        )

    baseline_value = section.get("comparison_baseline", "method")
    if str(baseline_value).strip().lower() == "method":
        baseline_source = "method"
        baseline = resolve_importance_metric(
            settings.fl_method.segment_importance_name,
            "probes.importance_correlation.comparison_baseline",
        )
    else:
        baseline_source = "explicit"
        baseline = resolve_importance_metric(
            baseline_value,
            "probes.importance_correlation.comparison_baseline",
        )
    if baseline.is_block_metric:
        raise ValueError(
            "probes.importance_correlation.comparison_baseline must resolve to a "
            "parameter-level metric"
        )

    references = tuple(
        resolve_importance_metric(name, "probes.importance_correlation.reference_scores")
        for name in _enabled_reference_score_names(section.get("reference_scores"))
    )
    measurements, top_k = _measurement_config(section.get("measurements"))

    return ImportanceCorrelationProbeConfig(
        stage=_correlation_stage(section.get("stage", "after_training")),
        comparison_baseline=baseline,
        comparison_baseline_source=baseline_source,
        reference_scores=references,
        measurements=measurements,
        top_k=top_k,
        evaluation_rounds=_evaluation_rounds(
            section.get("evaluation_rounds", "all"),
            settings.training.rounds,
        ),
        parameter_scope=settings.fl_method.parameter_scope,
        bn_mode=(
            settings.fl_method.bn_mode
            if settings.fl_method.bn_process_as_base_unit
            else "none"
        ),
        hutchinson=_hutchinson_config(section.get("hutchinson")),
    )


def _correlation_stage(value) -> str:
    stage = str(value).strip().lower()
    if stage not in IMPORTANCE_CORRELATION_STAGES:
        raise ValueError(
            "probes.importance_correlation.stage must be after_training "
            "or after_aggregation"
        )
    return stage


def _enabled_reference_score_names(value) -> tuple[str, ...]:
    if value is None:
        return REFERENCE_SCORE_NAMES
    if not isinstance(value, dict):
        raise ValueError(
            "probes.importance_correlation.reference_scores must be a mapping"
        )
    unknown = sorted(set(value) - set(REFERENCE_SCORE_NAMES))
    if unknown:
        raise ValueError(f"Unsupported importance reference score(s): {unknown}")
    enabled = []
    for name in REFERENCE_SCORE_NAMES:
        selected = value.get(name, False)
        if not isinstance(selected, bool):
            raise ValueError(
                f"probes.importance_correlation.reference_scores.{name} must be a boolean"
            )
        if selected:
            enabled.append(name)
    if not enabled:
        raise ValueError(
            "probes.importance_correlation.reference_scores must enable at least one score"
        )
    return tuple(enabled)


def _measurement_config(value) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if value is None:
        return MEASUREMENT_NAMES, (0.05, 0.1, 0.2, 0.3)
    if not isinstance(value, dict):
        raise ValueError("probes.importance_correlation.measurements must be a mapping")
    unknown = sorted(set(value) - {*MEASUREMENT_NAMES, "top_k"})
    if unknown:
        raise ValueError(f"Unsupported importance correlation measurement key(s): {unknown}")

    enabled = []
    for name in MEASUREMENT_NAMES:
        selected = value.get(name, False)
        if not isinstance(selected, bool):
            raise ValueError(
                f"probes.importance_correlation.measurements.{name} must be a boolean"
            )
        if selected:
            enabled.append(name)
    if not enabled:
        raise ValueError(
            "probes.importance_correlation.measurements must enable at least one measurement"
        )

    raw_top_k = value.get("top_k", (0.05, 0.1, 0.2, 0.3))
    if not isinstance(raw_top_k, (list, tuple)):
        raise ValueError("probes.importance_correlation.measurements.top_k must be a list")
    top_k = []
    for raw_value in raw_top_k:
        fraction = float(raw_value)
        if not 0 < fraction <= 1:
            raise ValueError(
                "probes.importance_correlation.measurements.top_k values must be in (0, 1]"
            )
        if fraction not in top_k:
            top_k.append(fraction)
    if ({"top_k_overlap", "top_k_jaccard"} & set(enabled)) and not top_k:
        raise ValueError(
            "top_k must contain at least one value when a top-k measurement is enabled"
        )
    return tuple(enabled), tuple(top_k)


def _evaluation_rounds(value, total_rounds: int) -> tuple[int, ...] | None:
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "probes.importance_correlation.evaluation_rounds must be 'all' or a list"
        )
    rounds = []
    for raw_round in value:
        if isinstance(raw_round, bool) or not isinstance(raw_round, int):
            raise ValueError("importance correlation evaluation rounds must be integers")
        if not 1 <= raw_round <= total_rounds:
            raise ValueError(
                f"importance correlation evaluation round [{raw_round}] is outside 1..{total_rounds}"
            )
        if raw_round not in rounds:
            rounds.append(raw_round)
    if not rounds:
        raise ValueError("importance correlation evaluation_rounds must not be empty")
    return tuple(rounds)


def _hutchinson_config(value) -> HutchinsonProbeConfig:
    if value is None:
        return HutchinsonProbeConfig()
    if not isinstance(value, dict):
        raise ValueError("probes.importance_correlation.hutchinson must be a mapping")
    unknown = sorted(set(value) - {"probe_vectors", "batch_limit"})
    if unknown:
        raise ValueError(f"Unsupported Hutchinson probe key(s): {unknown}")
    probe_vectors = int(value.get("probe_vectors", 5))
    batch_limit = int(value.get("batch_limit", 0))
    if probe_vectors <= 0:
        raise ValueError("hutchinson.probe_vectors must be a positive integer")
    if batch_limit < 0:
        raise ValueError("hutchinson.batch_limit must be a non-negative integer")
    return HutchinsonProbeConfig(
        probe_vectors=probe_vectors,
        batch_limit=batch_limit,
    )
