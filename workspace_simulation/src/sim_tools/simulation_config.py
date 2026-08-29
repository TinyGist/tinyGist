"""Strict parser for simulation configuration schema version 2.

Schema v2 intentionally does not read legacy configuration files.  Runtime
objects remain flat where that keeps the simulation manager simple, while the
YAML is grouped by responsibility and optional features are enabled by the
presence of their section rather than by scattered boolean switches.
"""

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path

import yaml

from src.differential_privacy.config import (
    DifferentialPrivacyConfig,
    parse_differential_privacy_config,
)
from src.fl_methods.definitions import (
    BLOCK_ALIGNED_AGGREGATION_SCORE_SOURCES,
    BLOCK_SEGMENT_UNITS,
    GROUPABLE_SEGMENT_UNITS,
    REFINABLE_SEGMENT_UNITS,
)
from src.models.definitions import canonical_parameter_scope
from src.scoring.config_metrics import (
    ImportanceMetric,
    resolve_aggregation_metric,
    resolve_importance_metric,
)
from src.scoring.definitions import (
    canonical_score_combine_method,
    canonical_score_project_method,
    parameter_score_record_dependencies,
)

from .definitions import canonical_experiment_task, canonical_method_name, log_value_slug


SCHEMA_VERSION = 2
DEFAULT_DEVICE_PREFIX = "device_"
DEFAULT_LOG_RUN_DIR_NAME_MAX_LENGTH = 96


@dataclass(frozen=True)
class EarlyStopCeilingConfig:
    enabled: bool
    metric: str
    value: float


@dataclass(frozen=True)
class EarlyStopPlateauConfig:
    enabled: bool
    metric: str
    patience: int
    min_delta: float
    near_best_ratio: float


@dataclass(frozen=True)
class EarlyStopRecordConfig:
    scope: str
    decay: float


@dataclass(frozen=True)
class EarlyStopConfig:
    enabled: bool
    scope: str
    min_epoch: int
    ceiling: EarlyStopCeilingConfig
    plateau: EarlyStopPlateauConfig
    record: EarlyStopRecordConfig


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int
    device_count: int
    device_indicator_prefix: str
    optimizer_name: str
    initial_lr: float
    weight_decay: float
    epoch_per_round: int
    max_batch_per_epoch: int
    early_stop: EarlyStopConfig
    loss_function_name: str


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    training_data_per_device: int
    labels_per_device: int
    label_allocating_method: str
    label_allocating_loop_step: int
    data_allocating_method: str
    data_allocating_alpha: float
    test_data_size_total: int
    valid_data_size_total: int
    train_batch_size: int
    test_batch_size: int
    valid_batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    input_size: int
    torch_random_seed: int
    output_class_number: int


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    repeat_count: int
    random_seed: int


@dataclass(frozen=True)
class FLMethodConfig:
    fl_method_name: str
    segment_divided_number: int
    segment_communicating_number: int
    segment_create_method: str
    segment_chosen_method: str
    segment_pick_exp_normalize: bool
    segment_pick_exp_base: float
    aggregating_threshold: int
    largest_seg_stored_num: int
    parameter_scope: str
    segment_unit: str
    channel_length: int
    block_refinement_enabled: bool
    block_refinement_targets: tuple[str, ...]
    linear_chunk_size: int
    pointwise_chunk_size: int
    base_unit_bias_mode: str
    bn_mode: str
    bn_process_mode: str
    bn_aggregation_source: str
    group_enabled: bool
    block_grouping_method: str
    block_group_size: int
    group_criterion_name: str
    group_criterion_metric: str
    group_criterion_parameter_to_unit: str
    segment_importance_name: str
    segment_importance_metric: str
    importance_parameter_to_unit: str
    importance_unit_to_group: str
    importance_group_to_segment: str
    segment_compose_method: str
    segment_importance_start_round: int
    aggregation_weight_name: str
    aggregation_score_source: str
    aggregation_weight_start_round: int
    refresh_batch_norm_from_validation: bool
    aggregation_lipschitz_score_ema_enabled: bool
    fisher_cal: str
    fisher_block: bool
    fisher_granularity: str
    fisher_block_reduce_method: str
    aggregation_weight_normalization: str
    aggregation_weight_exp_normalize: bool
    aggregation_weight_exp_base: float
    aggregation_update_unit_l2_mode: str
    aggregation_update_unit_l2_multiplier: float | None
    local_update_unit_l2_mode: str
    local_update_unit_l2_multiplier: float | None
    local_update_l2_start_round: int
    unit_l2_mode: str
    unit_l2_multiplier: float | None
    fisher_reset_buffer_each_round: bool
    round_reset_parameter_score_methods: tuple[str, ...]
    reset_block_interaction_each_round: bool
    selection_lipschitz_score_ema_enabled: bool
    group_criterion_lipschitz_score_ema_enabled: bool
    hutchinson_z_time: int
    hutchinson_batch_limit: int
    recipient_pick_method: str
    recipient_balance_strength: float

    @property
    def preserve_unit_l2(self) -> bool:
        """Compatibility view for callers that only distinguish enabled/disabled."""
        return self.unit_l2_mode != "none"

    @property
    def bn_process_as_base_unit(self) -> bool:
        return self.bn_process_mode == "base_unit"

    @property
    def block_refinement(self) -> dict:
        return {
            "enabled": self.block_refinement_enabled,
            "targets": self.block_refinement_targets,
            "linear_chunk_size": self.linear_chunk_size,
            "pointwise_chunk_size": self.pointwise_chunk_size,
            "bias": self.base_unit_bias_mode,
        }


@dataclass(frozen=True)
class UtilsConfig:
    use_adaptive_lr: bool
    use_AdaLoss: bool
    use_AdaStair: bool
    topology_shape: str
    topology_position_change: bool
    com_stability_mean: float
    com_stability_std: float
    com_highest_stability: float
    com_lowest_stability: float
    stale_sim_method: str
    stale_sim_distribution: str
    stale_gauss_mean: float
    stale_gauss_std: float
    stale_chi_square_k: float
    stale_uniform_multiplier: float
    stale_highest_probability: float
    stale_lowest_probability: float
    log_run_dir_name_max_length: int


class ConfigSnapshot:
    def __init__(self, source_data):
        self.source_data = source_data

    def write(self, fp):
        yaml.safe_dump(self.source_data, fp, sort_keys=False)


@dataclass(frozen=True)
class SimulationConfig:
    parser: ConfigSnapshot
    experiment: ExperimentConfig
    training: TrainingConfig
    dataset: DatasetConfig
    model: ModelConfig
    fl_method: FLMethodConfig
    differential_privacy: DifferentialPrivacyConfig
    utils: UtilsConfig


def load_simulation_config(config_file) -> SimulationConfig:
    path = Path(config_file)
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Unsupported config format [{path.suffix}]. Use .yml or .yaml.")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    data = _mapping(data, "config")
    _keys(
        data,
        {
            "schema_version",
            "experiment",
            "federation",
            "dataset",
            "model",
            "training",
            "method",
            "differential_privacy",
            "probes",
            "output",
        },
        "config",
    )
    version = _positive_int(_required(data, "schema_version", "config"), "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}; legacy schemas are intentionally unsupported"
        )

    experiment_section = _section(data, "experiment")
    federation_section = _section(data, "federation")
    dataset_section = _section(data, "dataset")
    model_section = _section(data, "model")
    training_section = _section(data, "training")
    output_section = _section(data, "output", required=False)
    method_section = _section(data, "method")

    experiment = _yaml_experiment(experiment_section)
    training = _yaml_training(federation_section, training_section)
    dataset = _yaml_dataset(dataset_section)
    model = _yaml_model(model_section, experiment.random_seed)
    fl_method = _yaml_fl_method(method_section, training.rounds)
    differential_privacy = parse_differential_privacy_config(
        data.get("differential_privacy")
    )
    utils = _yaml_utils(federation_section, training_section, output_section)
    _validate_differential_privacy_compatibility(
        differential_privacy,
        fl_method,
        training,
        experiment,
        utils,
    )
    return SimulationConfig(
        parser=ConfigSnapshot(data),
        experiment=experiment,
        training=training,
        dataset=dataset,
        model=model,
        fl_method=fl_method,
        differential_privacy=differential_privacy,
        utils=utils,
    )


def _validate_differential_privacy_compatibility(
        dp_config,
        method_config,
        training_config,
        experiment_config,
        utils_config,
):
    if not dp_config.enabled:
        return
    incompatible = []
    if method_config.parameter_scope != "all":
        incompatible.append("method.parameters.scope=all")
    if method_config.segment_unit != "parameter":
        incompatible.append("method.partition.unit=parameter")
    if method_config.bn_mode != "affine":
        incompatible.append("method.parameters.batch_norm.include=affine")
    if method_config.aggregation_score_source != "val_acc":
        incompatible.append(
            "method.aggregation.weight.metric=validation_accuracy"
        )
    if (
        method_config.segment_create_method == "importance"
        and method_config.segment_importance_metric != "weight_abs"
    ):
        incompatible.append(
            "method.segment_importance.metric=parameter_magnitude.current "
            "for importance-based construction"
        )
    if dp_config.mode == "local_dp_sgd":
        if training_config.optimizer_name != "sgd":
            incompatible.append("training.optimizer.name=sgd")
        if training_config.early_stop.enabled:
            incompatible.append("training.local.early_stop=null")
        if utils_config.use_adaptive_lr:
            incompatible.append("training.learning_rate_adaptation.strategies=[]")
        if experiment_config.task != "Classification":
            incompatible.append("experiment.task=classification")
    if incompatible:
        raise ValueError(
            f"differential_privacy.mode={dp_config.mode} currently requires: "
            + ", ".join(incompatible)
        )


def _yaml_experiment(section: dict) -> ExperimentConfig:
    _keys(section, {"task", "repeat_count", "seed", "repeat_index", "base_seed"}, "experiment")
    seed = _integer(_required(section, "seed", "experiment"), "experiment.seed")
    return ExperimentConfig(
        task=canonical_experiment_task(_required(section, "task", "experiment")),
        repeat_count=_positive_int(section.get("repeat_count", 1), "experiment.repeat_count"),
        random_seed=seed,
    )


def _yaml_training(federation: dict, section: dict) -> TrainingConfig:
    _keys(federation, {"rounds", "clients", "network"}, "federation")
    clients = _section(federation, "clients")
    _keys(clients, {"count", "availability"}, "federation.clients")
    _keys(section, {"optimizer", "local", "loss", "learning_rate_adaptation"}, "training")
    optimizer = _section(section, "optimizer")
    _keys(optimizer, {"name", "learning_rate", "weight_decay"}, "training.optimizer")
    local = _section(section, "local")
    _keys(local, {"epochs_per_round", "max_batches_per_epoch", "early_stop"}, "training.local")

    optimizer_name = str(_required(optimizer, "name", "training.optimizer")).strip().lower()
    supported_optimizers = {"adam_cross_round", "adam_round", "sgd"}
    if optimizer_name not in supported_optimizers:
        raise ValueError(
            "training.optimizer.name must be adam_cross_round, adam_round, or sgd"
        )

    loss_name = str(_required(section, "loss", "training")).strip()
    loss_name = {"cross_entropy": "ce_loss"}.get(loss_name.lower(), loss_name)
    return TrainingConfig(
        rounds=_positive_int(_required(federation, "rounds", "federation"), "federation.rounds"),
        device_count=_bounded_int(
            _required(clients, "count", "federation.clients"),
            "federation.clients.count",
            2,
            2 ** 31 - 1,
        ),
        device_indicator_prefix=DEFAULT_DEVICE_PREFIX,
        optimizer_name=optimizer_name,
        initial_lr=_positive_float(_required(optimizer, "learning_rate", "training.optimizer"), "training.optimizer.learning_rate"),
        weight_decay=_non_negative_float(optimizer.get("weight_decay", 5e-4), "training.optimizer.weight_decay"),
        epoch_per_round=_positive_int(_required(local, "epochs_per_round", "training.local"), "training.local.epochs_per_round"),
        max_batch_per_epoch=_positive_int(
            _required(local, "max_batches_per_epoch", "training.local"),
            "training.local.max_batches_per_epoch",
        ),
        early_stop=_yaml_early_stop(local.get("early_stop")),
        loss_function_name=loss_name,
    )


def _yaml_early_stop(raw) -> EarlyStopConfig:
    if raw is None:
        return EarlyStopConfig(
            enabled=False,
            scope="local_train",
            min_epoch=0,
            ceiling=EarlyStopCeilingConfig(False, "acc", 1.0),
            plateau=EarlyStopPlateauConfig(False, "acc", 3, 0.0, 0.9),
            record=EarlyStopRecordConfig("round", 1.0),
        )
    section = _mapping(raw, "training.local.early_stop")
    _keys(section, {"min_epochs", "ceiling", "plateau", "record"}, "training.local.early_stop")
    ceiling = section.get("ceiling")
    plateau = section.get("plateau")
    record = _mapping(section.get("record") or {}, "training.local.early_stop.record")
    _keys(record, {"scope", "decay"}, "training.local.early_stop.record")
    ceiling_config = _yaml_early_stop_ceiling(ceiling)
    plateau_config = _yaml_early_stop_plateau(plateau)
    if not ceiling_config.enabled and not plateau_config.enabled:
        raise ValueError("training.local.early_stop must define ceiling and/or plateau")
    record_scope = str(record.get("scope", "round")).strip().lower()
    if record_scope not in {"round", "client"}:
        raise ValueError("training.local.early_stop.record.scope must be round or client")
    return EarlyStopConfig(
        enabled=True,
        scope="local_train",
        min_epoch=_non_negative_int(section.get("min_epochs", 0), "training.local.early_stop.min_epochs"),
        ceiling=ceiling_config,
        plateau=plateau_config,
        record=EarlyStopRecordConfig(
            "device" if record_scope == "client" else "round",
            _ratio(record.get("decay", 1.0), "training.local.early_stop.record.decay"),
        ),
    )


def _yaml_early_stop_ceiling(raw) -> EarlyStopCeilingConfig:
    if raw is None:
        return EarlyStopCeilingConfig(False, "acc", 1.0)
    section = _mapping(raw, "training.local.early_stop.ceiling")
    _keys(section, {"metric", "threshold"}, "training.local.early_stop.ceiling")
    metric = _early_stop_metric(_required(section, "metric", "training.local.early_stop.ceiling"))
    threshold = _early_stop_threshold(
        metric,
        _required(section, "threshold", "training.local.early_stop.ceiling"),
        "training.local.early_stop.ceiling.threshold",
    )
    return EarlyStopCeilingConfig(True, metric, threshold)


def _yaml_early_stop_plateau(raw) -> EarlyStopPlateauConfig:
    if raw is None:
        return EarlyStopPlateauConfig(False, "acc", 3, 0.0, 0.9)
    section = _mapping(raw, "training.local.early_stop.plateau")
    _keys(section, {"metric", "patience", "min_delta", "near_best_ratio"}, "training.local.early_stop.plateau")
    return EarlyStopPlateauConfig(
        True,
        _early_stop_metric(_required(section, "metric", "training.local.early_stop.plateau")),
        _positive_int(_required(section, "patience", "training.local.early_stop.plateau"), "training.local.early_stop.plateau.patience"),
        _non_negative_float(section.get("min_delta", 0.0), "training.local.early_stop.plateau.min_delta"),
        _ratio(section.get("near_best_ratio", 0.9), "training.local.early_stop.plateau.near_best_ratio"),
    )


def _yaml_dataset(section: dict) -> DatasetConfig:
    _keys(section, {"name", "partition", "evaluation", "batches"}, "dataset")
    partition = _section(section, "partition")
    evaluation = _section(section, "evaluation")
    batches = _section(section, "batches")
    _keys(partition, {"samples_per_client", "labels_per_client", "label_assignment", "sample_assignment"}, "dataset.partition")
    label_assignment = _section(partition, "label_assignment")
    sample_assignment = _section(partition, "sample_assignment")
    _keys(label_assignment, {"strategy", "loop_step"}, "dataset.partition.label_assignment")
    _keys(sample_assignment, {"strategy", "dirichlet_alpha"}, "dataset.partition.sample_assignment")
    _keys(evaluation, {"test_samples", "validation_samples"}, "dataset.evaluation")
    _keys(batches, {"train_size", "test_size", "validation_size"}, "dataset.batches")
    label_strategy = str(_required(label_assignment, "strategy", "dataset.partition.label_assignment")).strip().lower()
    if label_strategy not in {"ordered", "random", "loop"}:
        raise ValueError("dataset.partition.label_assignment.strategy must be ordered, random, or loop")
    sample_strategy = str(_required(sample_assignment, "strategy", "dataset.partition.sample_assignment")).strip().lower()
    if sample_strategy not in {"dirichlet", "uniform", "random"}:
        raise ValueError("dataset.partition.sample_assignment.strategy must be dirichlet, uniform, or random")

    return DatasetConfig(
        dataset_name=str(_required(section, "name", "dataset")),
        training_data_per_device=_positive_int(_required(partition, "samples_per_client", "dataset.partition"), "dataset.partition.samples_per_client"),
        labels_per_device=_positive_int(_required(partition, "labels_per_client", "dataset.partition"), "dataset.partition.labels_per_client"),
        label_allocating_method=label_strategy,
        label_allocating_loop_step=_positive_int(label_assignment.get("loop_step", 1), "dataset.partition.label_assignment.loop_step"),
        data_allocating_method=sample_strategy,
        data_allocating_alpha=_positive_float(sample_assignment.get("dirichlet_alpha", 1.0), "dataset.partition.sample_assignment.dirichlet_alpha"),
        test_data_size_total=_positive_int(_required(evaluation, "test_samples", "dataset.evaluation"), "dataset.evaluation.test_samples"),
        valid_data_size_total=_positive_int(_required(evaluation, "validation_samples", "dataset.evaluation"), "dataset.evaluation.validation_samples"),
        train_batch_size=_positive_int(_required(batches, "train_size", "dataset.batches"), "dataset.batches.train_size"),
        test_batch_size=_positive_int(_required(batches, "test_size", "dataset.batches"), "dataset.batches.test_size"),
        valid_batch_size=_positive_int(_required(batches, "validation_size", "dataset.batches"), "dataset.batches.validation_size"),
    )


def _yaml_model(section: dict, random_seed: int) -> ModelConfig:
    _keys(section, {"name", "input_size", "num_classes"}, "model")
    return ModelConfig(
        model_name=str(_required(section, "name", "model")),
        input_size=_positive_int(_required(section, "input_size", "model"), "model.input_size"),
        torch_random_seed=int(random_seed),
        output_class_number=_positive_int(_required(section, "num_classes", "model"), "model.num_classes"),
    )


def _yaml_fl_method(section: dict, total_rounds: int) -> FLMethodConfig:
    _keys(
        section,
        {"name", "activation_schedule", "parameters", "partition", "grouping", "segment_importance", "segments", "exchange", "local_update", "aggregation", "estimators"},
        "method",
    )
    method_name = canonical_method_name(_required(section, "name", "method"))
    activation_schedule = _section(section, "activation_schedule", required=False)
    parameters = _section(section, "parameters", required=False)
    partition = _section(section, "partition", required=False)
    segments = _section(section, "segments", required=False)
    exchange = _section(section, "exchange", required=False)
    local_update = _section(section, "local_update", required=False)
    aggregation = _section(section, "aggregation", required=False)
    estimators = _section(section, "estimators", required=False)

    parameter_scope, bn_mode, bn_process_mode, bn_aggregation_source = _yaml_method_parameters(parameters)
    (
        segment_unit,
        channel_length,
        bias_mode,
        refinement_enabled,
        refinement_targets,
        linear_chunk_size,
        pointwise_chunk_size,
    ) = _yaml_partition(partition)
    group_config = _yaml_grouping(section.get("grouping"))
    importance_config = _yaml_segment_importance(section.get("segment_importance"))
    segment_config = _yaml_segments(segments, method_name)
    exchange_config = _yaml_exchange(exchange, method_name)
    local_update_unit_l2_mode, local_update_unit_l2_multiplier = _yaml_local_update(
        local_update,
        segment_unit,
        method_name,
    )
    aggregation_config = _yaml_aggregation(
        aggregation,
        segment_unit,
        method_name,
    )
    hutchinson_z_time, hutchinson_batch_limit = _yaml_estimators(estimators)
    (
        segment_importance_start_round,
        aggregation_weight_start_round,
        local_update_l2_start_round,
    ) = _yaml_activation_schedule(activation_schedule, total_rounds)

    if refinement_enabled and segment_unit not in REFINABLE_SEGMENT_UNITS:
        raise ValueError("method.partition.refinement requires unit kernel or channel")
    if group_config[0] and segment_unit not in GROUPABLE_SEGMENT_UNITS:
        raise ValueError("method.grouping requires method.partition.unit kernel or channel")
    if segment_config[2] == "importance" and importance_config is None:
        raise ValueError("method.segment_importance is required for importance-based construction")
    if segment_config[3] == "probabilistic" and importance_config is None:
        raise ValueError("method.segment_importance is required for importance-weighted selection")
    if group_config[0] and segment_config[2] != "importance":
        raise ValueError("method.grouping is only used by importance-based segment construction")
    if method_name == "Gist" and segment_config[2:4] != ("importance", "probabilistic"):
        raise ValueError(
            "Gist requires construction=importance_balanced_by_payload and "
            "selection.strategy=importance_weighted"
        )
    if method_name == "DFA" and (segment_config[0] != 1 or segment_config[2:4] != ("consistent", "uniform")):
        raise ValueError("DFA requires one fixed segment with random_uniform selection")
    if method_name == "SegmentPulling" and segment_config[3] != "uniform":
        raise ValueError("SegmentPulling supports random_uniform segment selection only")
    if importance_config is None:
        default_metric = resolve_importance_metric("parameter_magnitude.current")
        importance_config = (default_metric, "mean_abs", "mean", "mean")

    importance_metric, parameter_to_unit, unit_to_group, group_to_segment = importance_config
    group_enabled, grouping_method, group_size, criterion_metric, criterion_projection = group_config
    if (
        group_enabled
        and importance_metric.internal_name == criterion_metric.internal_name
        and importance_metric.reset_each_round != criterion_metric.reset_each_round
    ):
        raise ValueError(
            "grouping.criterion and segment_importance cannot use different lifetimes "
            "of the same underlying metric in one run"
        )
    reset_methods = _round_reset_methods(importance_metric, criterion_metric if group_enabled else None)
    reset_interaction = any(
        metric is not None
        and metric.internal_name == "fisher_lipschitz_cooperation"
        and metric.reset_each_round
        for metric in (importance_metric, criterion_metric if group_enabled else None)
    )

    (
        aggregation_metric,
        fisher_granularity,
        fisher_reduction,
        refresh_batch_norm_from_validation,
        aggregation_weight_normalization,
        aggregation_weight_exp_normalize,
        aggregation_weight_exp_base,
        aggregation_update_unit_l2_mode,
        aggregation_update_unit_l2_multiplier,
        unit_l2_mode,
        unit_l2_multiplier,
    ) = aggregation_config
    if aggregation_metric.source in BLOCK_ALIGNED_AGGREGATION_SCORE_SOURCES and segment_unit not in BLOCK_SEGMENT_UNITS:
        raise ValueError(
            f"method.aggregation.weight.metric={aggregation_metric.name} requires block partitioning"
        )

    return FLMethodConfig(
        fl_method_name=method_name,
        segment_divided_number=segment_config[0],
        segment_communicating_number=exchange_config[0],
        segment_create_method=segment_config[2],
        segment_chosen_method=segment_config[3],
        segment_pick_exp_normalize=segment_config[4],
        segment_pick_exp_base=segment_config[5],
        aggregating_threshold=exchange_config[3],
        largest_seg_stored_num=exchange_config[4],
        parameter_scope=parameter_scope,
        segment_unit=segment_unit,
        channel_length=channel_length,
        block_refinement_enabled=refinement_enabled,
        block_refinement_targets=refinement_targets,
        linear_chunk_size=linear_chunk_size,
        pointwise_chunk_size=pointwise_chunk_size,
        base_unit_bias_mode=bias_mode,
        bn_mode=bn_mode,
        bn_process_mode=bn_process_mode,
        bn_aggregation_source=bn_aggregation_source,
        group_enabled=group_enabled,
        block_grouping_method=grouping_method,
        block_group_size=group_size,
        group_criterion_name=criterion_metric.name,
        group_criterion_metric=criterion_metric.internal_name,
        group_criterion_parameter_to_unit=criterion_projection,
        segment_importance_name=importance_metric.name,
        segment_importance_metric=importance_metric.internal_name,
        importance_parameter_to_unit=parameter_to_unit,
        importance_unit_to_group=unit_to_group,
        importance_group_to_segment=group_to_segment,
        segment_compose_method=segment_config[1],
        segment_importance_start_round=segment_importance_start_round,
        aggregation_weight_name=aggregation_metric.name,
        aggregation_score_source=aggregation_metric.source,
        aggregation_weight_start_round=aggregation_weight_start_round,
        refresh_batch_norm_from_validation=refresh_batch_norm_from_validation,
        aggregation_lipschitz_score_ema_enabled=aggregation_metric.lipschitz_ema,
        fisher_cal=aggregation_metric.fisher_calculation,
        fisher_block=fisher_granularity == "block",
        fisher_granularity=fisher_granularity,
        fisher_block_reduce_method=fisher_reduction,
        aggregation_weight_normalization=aggregation_weight_normalization,
        aggregation_weight_exp_normalize=aggregation_weight_exp_normalize,
        aggregation_weight_exp_base=aggregation_weight_exp_base,
        aggregation_update_unit_l2_mode=aggregation_update_unit_l2_mode,
        aggregation_update_unit_l2_multiplier=aggregation_update_unit_l2_multiplier,
        local_update_unit_l2_mode=local_update_unit_l2_mode,
        local_update_unit_l2_multiplier=local_update_unit_l2_multiplier,
        local_update_l2_start_round=local_update_l2_start_round,
        unit_l2_mode=unit_l2_mode,
        unit_l2_multiplier=unit_l2_multiplier,
        fisher_reset_buffer_each_round=aggregation_metric.reset_each_round,
        round_reset_parameter_score_methods=reset_methods,
        reset_block_interaction_each_round=reset_interaction,
        selection_lipschitz_score_ema_enabled=importance_metric.lipschitz_ema,
        group_criterion_lipschitz_score_ema_enabled=(
            group_enabled and criterion_metric.lipschitz_ema
        ),
        hutchinson_z_time=hutchinson_z_time,
        hutchinson_batch_limit=hutchinson_batch_limit,
        recipient_pick_method=exchange_config[1],
        recipient_balance_strength=exchange_config[2],
    )


def _yaml_method_parameters(section: dict):
    _keys(section, {"scope", "batch_norm"}, "method.parameters")
    bn = _section(section, "batch_norm", required=False)
    _keys(bn, {"include", "distribution", "aggregation_weight"}, "method.parameters.batch_norm")
    include = str(bn.get("include", "affine")).strip().lower()
    distribution = str(bn.get("distribution", "as_base_unit")).strip().lower()
    aggregation_weight = str(bn.get("aggregation_weight", "same_as_model")).strip().lower()
    try:
        bn_mode = {
            "none": "none",
            "affine": "affine",
            "affine_and_running_stats": "affine_running_stats",
        }[include]
    except KeyError as exc:
        raise ValueError("method.parameters.batch_norm.include must be none, affine, or affine_and_running_stats") from exc
    try:
        bn_process = {
            "as_base_unit": "base_unit",
            "with_each_segment": "separate_per_segment",
            "once_per_recipient": "separate_per_recipient",
        }[distribution]
    except KeyError as exc:
        raise ValueError("Invalid method.parameters.batch_norm.distribution") from exc
    try:
        bn_aggregation = {"same_as_model": "score", "uniform": "uniform"}[aggregation_weight]
    except KeyError as exc:
        raise ValueError("method.parameters.batch_norm.aggregation_weight must be same_as_model or uniform") from exc
    return canonical_parameter_scope(section.get("scope", "all")), bn_mode, bn_process, bn_aggregation
def _yaml_partition(section: dict):
    _keys(section, {"unit", "channel_size", "bias", "refinement"}, "method.partition")
    unit = str(section.get("unit", "parameter")).strip().lower()
    if unit not in {"parameter", "kernel", "channel", "layer"}:
        raise ValueError("method.partition.unit must be parameter, kernel, channel, or layer")
    if unit == "layer":
        invalid_fields = {"channel_size", "bias", "refinement"} & set(section)
        if invalid_fields:
            raise ValueError(
                "method.partition.unit=layer does not use: "
                + ", ".join(sorted(invalid_fields))
            )
    channel_size = _non_negative_int(section.get("channel_size", 0), "method.partition.channel_size")
    bias_name = str(section.get("bias", "repeat_per_chunk")).strip().lower()
    try:
        bias = {"repeat_per_chunk": "each_chunk", "separate": "separate", "local_only": "local"}[bias_name]
    except KeyError as exc:
        raise ValueError("method.partition.bias must be repeat_per_chunk, separate, or local_only") from exc
    refinement = section.get("refinement")
    if refinement is None:
        return unit, channel_size, bias, False, (), 16, 16
    refinement = _mapping(refinement, "method.partition.refinement")
    _keys(refinement, {"targets", "linear_chunk_size", "pointwise_chunk_size"}, "method.partition.refinement")
    targets = tuple(str(item).strip().lower() for item in _sequence(_required(refinement, "targets", "method.partition.refinement")))
    invalid_targets = set(targets) - {"linear", "pointwise"}
    if invalid_targets or not targets:
        raise ValueError("method.partition.refinement.targets must contain linear and/or pointwise")
    return (
        unit,
        channel_size,
        bias,
        True,
        targets,
        _positive_int(refinement.get("linear_chunk_size", 16), "method.partition.refinement.linear_chunk_size"),
        _positive_int(refinement.get("pointwise_chunk_size", 16), "method.partition.refinement.pointwise_chunk_size"),
    )


def _yaml_grouping(raw):
    default = resolve_importance_metric("gradient_magnitude.round_step_ema")
    if raw is None:
        return False, "none", 1, default, "mean_abs"
    section = _mapping(raw, "method.grouping")
    _keys(section, {"strategy", "arrangement", "size", "criterion"}, "method.grouping")
    strategy = str(_required(section, "strategy", "method.grouping")).strip().lower()
    if strategy != "within_layer_by_criterion":
        raise ValueError("method.grouping.strategy must be within_layer_by_criterion")
    arrangement = str(section.get("arrangement", "sensitivity_aligned")).strip().lower()
    if arrangement not in {"sensitivity_aligned", "sensitivity_diverse"}:
        raise ValueError(
            "method.grouping.arrangement must be sensitivity_aligned or sensitivity_diverse"
        )
    criterion = _section(section, "criterion")
    _keys(criterion, {"metric", "parameter_to_unit"}, "method.grouping.criterion")
    metric = resolve_importance_metric(_required(criterion, "metric", "method.grouping.criterion"), "method.grouping.criterion.metric")
    return (
        True,
        arrangement,
        _positive_int(_required(section, "size", "method.grouping"), "method.grouping.size"),
        metric,
        canonical_score_project_method(criterion.get("parameter_to_unit", "mean_abs")),
    )


def _yaml_segment_importance(raw):
    if raw is None:
        return None
    section = _mapping(raw, "method.segment_importance")
    _keys(section, {"metric", "reductions"}, "method.segment_importance")
    reductions = _section(section, "reductions", required=False)
    _keys(reductions, {"parameter_to_unit", "unit_to_group", "group_to_segment"}, "method.segment_importance.reductions")
    return (
        resolve_importance_metric(_required(section, "metric", "method.segment_importance"), "method.segment_importance.metric"),
        canonical_score_project_method(reductions.get("parameter_to_unit", "mean_abs")),
        canonical_score_combine_method(reductions.get("unit_to_group", "mean")),
        canonical_score_combine_method(reductions.get("group_to_segment", "mean")),
    )


def _yaml_segments(section: dict, method_name: str):
    _keys(section, {"count", "construction", "selection"}, "method.segments")
    defaults = {
        "Centralized": (1, "fixed", "random_uniform"),
        "DFA": (1, "fixed", "random_uniform"),
        "SegmentPulling": (3, "fixed", "random_uniform"),
        "SDFA": (3, "fixed", "random_uniform"),
        "Gist": (3, "importance_balanced_by_payload", "importance_weighted"),
    }
    count_default, construction_default, selection_default = defaults[method_name]
    construction_name = str(section.get("construction", construction_default)).strip().lower()
    try:
        construction = {
            "fixed": "consistent",
            "reshuffle_once": "random_same",
            "reshuffle_each_round": "random_each",
            "importance_balanced_by_payload": "importance",
        }[construction_name]
    except KeyError as exc:
        raise ValueError("Invalid method.segments.construction") from exc
    selection = section.get("selection") or {}
    selection = _mapping(selection, "method.segments.selection")
    _keys(
        selection,
        {"strategy", "exponential_normalization", "exponential_base"},
        "method.segments.selection",
    )
    selection_name = str(selection.get("strategy", selection_default)).strip().lower()
    try:
        chosen = {"random_uniform": "uniform", "importance_weighted": "probabilistic"}[selection_name]
    except KeyError as exc:
        raise ValueError("method.segments.selection.strategy must be random_uniform or importance_weighted") from exc
    exponential_base = _finite_float(
        selection.get("exponential_base", math.e),
        "method.segments.selection.exponential_base",
    )
    if exponential_base <= 1:
        raise ValueError("method.segments.selection.exponential_base must be greater than 1")
    return (
        _positive_int(section.get("count", count_default), "method.segments.count"),
        "score_sorted_balanced_payload",
        construction,
        chosen,
        _bool(selection.get("exponential_normalization", False)),
        exponential_base,
    )


def _yaml_exchange(section: dict, method_name: str):
    _keys(section, {"sends_per_client", "pulls_per_segment", "recipient_selection", "receive_queue"}, "method.exchange")
    default_communication = 2 if method_name == "SegmentPulling" else 6
    if method_name == "SegmentPulling":
        communication_number = _positive_int(section.get("pulls_per_segment", default_communication), "method.exchange.pulls_per_segment")
        if "sends_per_client" in section:
            raise ValueError("SegmentPulling uses pulls_per_segment, not sends_per_client")
    else:
        communication_number = _non_negative_int(section.get("sends_per_client", default_communication), "method.exchange.sends_per_client")
        if "pulls_per_segment" in section:
            raise ValueError(f"{method_name} uses sends_per_client, not pulls_per_segment")
    recipient = _section(section, "recipient_selection", required=False)
    _keys(recipient, {"strategy", "balance_strength"}, "method.exchange.recipient_selection")
    recipient_name = str(recipient.get("strategy", "random_with_replacement")).strip().lower()
    try:
        recipient_method = {
            "random_with_replacement": "uniform_with_replacement",
            "balanced_probabilistic": "grouped_probabilistic",
            "balanced_unique_probabilistic": "grouped_unique_probabilistic",
            "balanced_round_robin": "grouped_round_robin",
            "random_without_replacement": "uniform_without_replacement",
        }[recipient_name]
    except KeyError as exc:
        raise ValueError("Invalid method.exchange.recipient_selection.strategy") from exc
    queue = _section(section, "receive_queue", required=False)
    _keys(queue, {"aggregation_threshold", "capacity"}, "method.exchange.receive_queue")
    return (
        communication_number,
        recipient_method,
        _non_negative_float(recipient.get("balance_strength", 1.0), "method.exchange.recipient_selection.balance_strength"),
        _positive_int(queue.get("aggregation_threshold", 5), "method.exchange.receive_queue.aggregation_threshold"),
        _positive_int(queue.get("capacity", 50), "method.exchange.receive_queue.capacity"),
    )


def _yaml_unit_l2(section, field_name, allowed_modes, minimum=0):
    _keys(section, {"mode", "multiplier"}, field_name)
    mode = str(section.get("mode", "none")).strip().lower()
    if mode not in allowed_modes:
        allowed = ", ".join(sorted(allowed_modes))
        raise ValueError(f"{field_name}.mode must be one of: {allowed}")
    if mode != "bounded":
        if "multiplier" in section:
            raise ValueError(
                f"{field_name}.multiplier is valid only in bounded mode"
            )
        return mode, None
    if "multiplier" not in section:
        raise ValueError(f"{field_name}.multiplier is required in bounded mode")
    multiplier = _positive_float(
        section["multiplier"],
        f"{field_name}.multiplier",
    )
    if multiplier < minimum:
        raise ValueError(
            f"{field_name}.multiplier must be at least {minimum:g}"
        )
    return mode, multiplier


def _validate_block_unit_l2(mode, field_name, segment_unit, method_name):
    if mode == "none":
        return
    if method_name == "Centralized":
        raise ValueError(f"{field_name} is not supported by Centralized")
    if segment_unit not in BLOCK_SEGMENT_UNITS:
        raise ValueError(
            f"{field_name} requires method.partition.unit "
            "to be kernel, channel, or layer"
        )


def _yaml_activation_schedule(section, total_rounds):
    field_names = (
        "segment_importance_start_round",
        "aggregation_weight_start_round",
        "local_update_l2_start_round",
    )
    _keys(section, set(field_names), "method.activation_schedule")
    return tuple(
        _activation_start_round(
            section.get(field_name, 1),
            total_rounds,
            f"method.activation_schedule.{field_name}",
        )
        for field_name in field_names
    )


def _activation_start_round(value, total_rounds, field_name):
    value = _finite_float(value, field_name)
    if value <= 0:
        raise ValueError(
            f"{field_name} must be a positive whole round or a ratio in (0, 1)"
        )
    if value < 1:
        return max(1, math.ceil(total_rounds * value))
    if not value.is_integer():
        raise ValueError(
            f"{field_name} must be a whole round when its value is at least 1"
        )
    return int(value)


def _yaml_local_update(section, segment_unit, method_name):
    _keys(section, {"unit_l2"}, "method.local_update")
    mode, multiplier = _yaml_unit_l2(
        _section(section, "unit_l2", required=False),
        "method.local_update.unit_l2",
        {"none", "bounded"},
    )
    _validate_block_unit_l2(
        mode,
        "method.local_update.unit_l2",
        segment_unit,
        method_name,
    )
    return mode, multiplier


def _yaml_aggregation(section, segment_unit, method_name):
    _keys(
        section,
        {"weight", "unit_l2", "update_unit_l2", "preserve_unit_l2"},
        "method.aggregation",
    )
    if "unit_l2" in section and "preserve_unit_l2" in section:
        raise ValueError(
            "method.aggregation.unit_l2 and the legacy preserve_unit_l2 key "
            "cannot be used together"
        )
    if "unit_l2" in section:
        unit_l2_mode, unit_l2_multiplier = _yaml_unit_l2(
            _section(section, "unit_l2"),
            "method.aggregation.unit_l2",
            {"none", "exact", "bounded"},
            minimum=1,
        )
    else:
        unit_l2_mode = (
            "exact"
            if _bool(section.get("preserve_unit_l2", False))
            else "none"
        )
        unit_l2_multiplier = None
    update_mode, update_multiplier = _yaml_unit_l2(
        _section(section, "update_unit_l2", required=False),
        "method.aggregation.update_unit_l2",
        {"none", "bounded"},
    )
    _validate_block_unit_l2(
        unit_l2_mode,
        "method.aggregation.unit_l2",
        segment_unit,
        method_name,
    )
    _validate_block_unit_l2(
        update_mode,
        "method.aggregation.update_unit_l2",
        segment_unit,
        method_name,
    )

    weight = _section(section, "weight", required=False)
    _keys(
        weight,
        {
            "metric",
            "granularity",
            "block_reduction",
            "normalization",
            "refresh_batch_norm_from_validation",
            "exponential_normalization",
            "exponential_base",
        },
        "method.aggregation.weight",
    )
    metric = resolve_aggregation_metric(
        weight.get("metric", "uniform"),
        "method.aggregation.weight.metric",
    )
    refresh_key = "refresh_batch_norm_from_validation"
    refresh_batch_norm = (
        _bool(weight.get(refresh_key, True))
        if metric.source == "val_acc"
        else False
    )
    granularity = str(weight.get("granularity", "parameter")).strip().lower()
    if granularity not in {"parameter", "block"}:
        raise ValueError(
            "method.aggregation.weight.granularity must be parameter or block"
        )
    reduction = str(weight.get("block_reduction", "mean")).strip().lower()
    if reduction not in {"mean", "rms", "l2"}:
        raise ValueError(
            "method.aggregation.weight.block_reduction must be mean, rms, or l2"
        )
    normalization = str(weight.get("normalization", "none")).strip().lower()
    if normalization not in {"none", "device_layer_l2"}:
        raise ValueError(
            "method.aggregation.weight.normalization must be "
            "none or device_layer_l2"
        )
    if normalization == "device_layer_l2":
        if not metric.name.startswith("fisher_diagonal."):
            raise ValueError(
                "method.aggregation.weight.normalization=device_layer_l2 "
                "requires a fisher_diagonal metric"
            )
        if segment_unit not in GROUPABLE_SEGMENT_UNITS or granularity != "block":
            raise ValueError(
                "method.aggregation.weight.normalization=device_layer_l2 "
                "requires kernel/channel partitioning with granularity=block"
            )
        if reduction != "l2":
            raise ValueError(
                "method.aggregation.weight.normalization=device_layer_l2 "
                "requires block_reduction=l2"
            )
    exponential_base = _finite_float(
        weight.get("exponential_base", math.e),
        "method.aggregation.weight.exponential_base",
    )
    if exponential_base <= 1:
        raise ValueError(
            "method.aggregation.weight.exponential_base must be greater than 1"
        )
    return (
        metric,
        granularity,
        reduction,
        refresh_batch_norm,
        normalization,
        _bool(weight.get("exponential_normalization", False)),
        exponential_base,
        update_mode,
        update_multiplier,
        unit_l2_mode,
        unit_l2_multiplier,
    )

def _yaml_estimators(section: dict):
    _keys(section, {"hutchinson"}, "method.estimators")
    raw = section.get("hutchinson")
    if raw is None:
        return 5, 0
    hutchinson = _mapping(raw, "method.estimators.hutchinson")
    _keys(hutchinson, {"probe_vectors", "batch_limit"}, "method.estimators.hutchinson")
    return (
        _positive_int(hutchinson.get("probe_vectors", 5), "method.estimators.hutchinson.probe_vectors"),
        _non_negative_int(hutchinson.get("batch_limit", 0), "method.estimators.hutchinson.batch_limit"),
    )


def _round_reset_methods(*metrics: ImportanceMetric | None) -> tuple[str, ...]:
    methods = []
    for metric in metrics:
        if metric is None or metric.is_block_metric or not metric.reset_each_round:
            continue
        for dependency in parameter_score_record_dependencies(metric.internal_name):
            if dependency not in methods:
                methods.append(dependency)
    return tuple(methods)


def _yaml_utils(federation: dict, training: dict, output: dict) -> UtilsConfig:
    clients = _section(federation, "clients")
    availability = _section(clients, "availability", required=False)
    network = _section(federation, "network", required=False)
    adaptation = _section(training, "learning_rate_adaptation", required=False)
    _keys(adaptation, {"strategies"}, "training.learning_rate_adaptation")
    strategies = tuple(str(item).strip().lower() for item in _sequence(adaptation.get("strategies", ())))
    invalid_strategies = set(strategies) - {"loss", "staircase"}
    if invalid_strategies:
        raise ValueError(f"Invalid learning-rate adaptation strategies: {sorted(invalid_strategies)}")

    _keys(network, {"topology", "reshuffle_each_round", "reliability"}, "federation.network")
    reliability = _section(network, "reliability", required=False)
    _keys(reliability, {"mean", "std", "minimum", "maximum"}, "federation.network.reliability")
    _keys(
        availability,
        {"strategy", "distribution", "gaussian_mean", "gaussian_std", "chi_square_k", "uniform_multiplier", "minimum", "maximum"},
        "federation.clients.availability",
    )
    _keys(output, {"max_run_directory_name_length"}, "output")
    topology = str(network.get("topology", "mesh_full")).strip().lower()
    supported_topologies = {
        "ring",
        "line",
        "star",
        "mesh_full",
        "tree_random",
        "tree_binary",
        "custom_clustered",
    }
    if topology not in supported_topologies:
        raise ValueError(f"federation.network.topology must be one of {sorted(supported_topologies)}")

    reliability_mean = _probability(reliability.get("mean", 0.8), "federation.network.reliability.mean")
    reliability_std = _non_negative_float(reliability.get("std", 0.1), "federation.network.reliability.std")
    reliability_min = _ratio(reliability.get("minimum", 1.0), "federation.network.reliability.minimum")
    reliability_max = _ratio(reliability.get("maximum", 1.0), "federation.network.reliability.maximum")
    if reliability_min > reliability_max:
        raise ValueError("federation.network.reliability.minimum must not exceed maximum")

    stale_method = str(availability.get("strategy", "probabilistic")).strip().lower()
    if stale_method not in {"probabilistic", "fixed_round"}:
        raise ValueError("federation.clients.availability.strategy must be probabilistic or fixed_round")
    stale_distribution = str(availability.get("distribution", "gaussian")).strip().lower()
    if stale_distribution not in {"gaussian", "uniform", "chi_square"}:
        raise ValueError("federation.clients.availability.distribution must be gaussian, uniform, or chi_square")
    stale_min = _ratio(availability.get("minimum", 1.0), "federation.clients.availability.minimum")
    stale_max = _ratio(availability.get("maximum", 1.0), "federation.clients.availability.maximum")
    if stale_min > stale_max:
        raise ValueError("federation.clients.availability.minimum must not exceed maximum")
    gaussian_mean_parser = _probability if stale_method == "probabilistic" else _positive_float

    return UtilsConfig(
        use_adaptive_lr=bool(strategies),
        use_AdaLoss="loss" in strategies,
        use_AdaStair="staircase" in strategies,
        topology_shape=topology,
        topology_position_change=_bool(network.get("reshuffle_each_round", False)),
        com_stability_mean=reliability_mean,
        com_stability_std=reliability_std,
        com_highest_stability=reliability_max,
        com_lowest_stability=reliability_min,
        stale_sim_method=stale_method,
        stale_sim_distribution=stale_distribution,
        stale_gauss_mean=gaussian_mean_parser(
            availability.get("gaussian_mean", 0.8),
            "federation.clients.availability.gaussian_mean",
        ),
        stale_gauss_std=_non_negative_float(
            availability.get("gaussian_std", 0.1),
            "federation.clients.availability.gaussian_std",
        ),
        stale_chi_square_k=_positive_float(
            availability.get("chi_square_k", 2),
            "federation.clients.availability.chi_square_k",
        ),
        stale_uniform_multiplier=_positive_float(
            availability.get("uniform_multiplier", 2),
            "federation.clients.availability.uniform_multiplier",
        ),
        stale_highest_probability=stale_max,
        stale_lowest_probability=stale_min,
        log_run_dir_name_max_length=_bounded_int(
            output.get("max_run_directory_name_length", DEFAULT_LOG_RUN_DIR_NAME_MAX_LENGTH),
            "output.max_run_directory_name_length",
            80,
            255,
        ),
    )


def resolve_method_abbreviation(method_config: FLMethodConfig, utils_config: UtilsConfig) -> str:
    parts = [log_value_slug(method_config.fl_method_name)]
    if utils_config.use_AdaStair:
        parts.append("adastair")
    if utils_config.use_AdaLoss:
        parts.append("ada_loss")
    if method_config.parameter_scope != "all":
        parts.append(method_config.parameter_scope)
    if method_config.fl_method_name != "Centralized":
        parts.extend((
            f"s{method_config.segment_divided_number}",
            log_value_slug(method_config.segment_unit),
            log_value_slug(method_config.segment_create_method),
            log_value_slug(method_config.segment_chosen_method),
        ))
        if method_config.segment_chosen_method == "probabilistic" and method_config.segment_pick_exp_normalize:
            parts.append(f"expb{method_config.segment_pick_exp_base:g}")
        if method_config.segment_create_method == "importance":
            parts.append(f"importance_{log_value_slug(method_config.segment_importance_name)}")
        if method_config.group_enabled:
            parts.append(
                f"group{method_config.block_group_size}_"
                f"{log_value_slug(method_config.block_grouping_method)}_"
                f"{log_value_slug(method_config.group_criterion_name)}"
            )
        if method_config.block_refinement_enabled:
            parts.append("ref_" + "_".join(method_config.block_refinement_targets))
        if method_config.fl_method_name in {"DFA", "SDFA", "Gist"}:
            parts.append(f"recipient_{log_value_slug(method_config.recipient_pick_method)}")
            if method_config.recipient_pick_method in {
                "grouped_probabilistic",
                "grouped_unique_probabilistic",
            }:
                parts.append(f"balance{method_config.recipient_balance_strength:g}")
    if method_config.segment_importance_start_round != 1:
        parts.append(f"segment_importance_from_r{method_config.segment_importance_start_round}")
    if method_config.aggregation_score_source != "uniform":
        parts.append(f"agg_{log_value_slug(method_config.aggregation_weight_name)}")
    if method_config.aggregation_weight_start_round != 1:
        parts.append(f"aggregation_weight_from_r{method_config.aggregation_weight_start_round}")
    if method_config.aggregation_weight_normalization != "none":
        parts.append(
            f"agg_norm_{log_value_slug(method_config.aggregation_weight_normalization)}"
        )
    if method_config.aggregation_weight_exp_normalize:
        parts.append(f"agg_weight_expb{method_config.aggregation_weight_exp_base:g}")
    if method_config.local_update_unit_l2_mode == "bounded":
        parts.append(
            f"local_unit_l2_bound_{method_config.local_update_unit_l2_multiplier:g}"
        )
    if method_config.local_update_l2_start_round != 1:
        parts.append(f"local_update_l2_from_r{method_config.local_update_l2_start_round}")
    if method_config.aggregation_update_unit_l2_mode == "bounded":
        parts.append(
            "agg_update_unit_l2_bound_"
            f"{method_config.aggregation_update_unit_l2_multiplier:g}"
        )
    if method_config.unit_l2_mode == "exact":
        parts.append("agg_unit_l2_preserve")
    elif method_config.unit_l2_mode == "bounded":
        parts.append(f"agg_unit_l2_bound_{method_config.unit_l2_multiplier:g}")
    if method_config.aggregation_score_source == "val_acc":
        refresh_state = "on" if method_config.refresh_batch_norm_from_validation else "off"
        parts.append(f"val_bn_stats_update_{refresh_state}")
    if method_config.hutchinson_z_time != 5:
        parts.append(f"hutchz{method_config.hutchinson_z_time}")
    if method_config.hutchinson_batch_limit:
        parts.append(f"hutchb{method_config.hutchinson_batch_limit}")
    return "+".join(parts)


def resolve_training_abbreviation(training_config: TrainingConfig) -> str:
    early_stop = training_config.early_stop
    parts = [f"optimizer_{log_value_slug(training_config.optimizer_name)}"]
    if not early_stop.enabled:
        return "+".join([*parts, "local_es_off"])
    parts.extend(["local_es_on", f"min_epoch{early_stop.min_epoch}"])
    if early_stop.ceiling.enabled:
        parts.append(f"ceil_{early_stop.ceiling.metric}{early_stop.ceiling.value:g}")
    if early_stop.plateau.enabled:
        parts.append(
            f"plat_{early_stop.plateau.metric}_p{early_stop.plateau.patience}"
            f"_d{early_stop.plateau.min_delta:g}_r{early_stop.plateau.near_best_ratio:g}"
        )
    if early_stop.record.scope != "round" or early_stop.record.decay != 1.0:
        parts.append(f"rec_{early_stop.record.scope}_decay{early_stop.record.decay:g}")
    return "+".join(parts)


def _section(parent: dict, key: str, *, required=True) -> dict:
    if key not in parent:
        if required:
            raise KeyError(f"Missing required config section: {key}")
        return {}
    value = parent[key]
    if value is None and not required:
        return {}
    return _mapping(value, key)


def _mapping(value, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _keys(section: dict, allowed: set[str], field_name: str):
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported key(s) in {field_name}: {unknown}. "
            "Legacy configuration keys are not accepted by schema v2."
        )


def _required(section: dict, key: str, field_name: str):
    if key not in section:
        raise KeyError(f"Missing required config key: {field_name}.{key}")
    return section[key]


def _sequence(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError("Expected a list")


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected a boolean, got {value!r}")


def _positive_int(value, field_name: str) -> int:
    value = _integer(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _non_negative_int(value, field_name: str) -> int:
    value = _integer(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _positive_float(value, field_name: str) -> float:
    value = _finite_float(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _non_negative_float(value, field_name: str) -> float:
    value = _finite_float(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _ratio(value, field_name: str) -> float:
    value = _finite_float(value, field_name)
    if not 0 < value <= 1:
        raise ValueError(f"{field_name} must be in (0, 1]")
    return value


def _bounded_int(value, field_name: str, minimum: int, maximum: int) -> int:
    value = _integer(value, field_name)
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return value


def _integer(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _finite_float(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _probability(value, field_name: str) -> float:
    value = _finite_float(value, field_name)
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return value


def _early_stop_metric(value) -> str:
    value = str(value).strip().lower()
    if value not in {"accuracy", "loss"}:
        raise ValueError("early-stop metric must be accuracy or loss")
    return "acc" if value == "accuracy" else "loss"


def _early_stop_threshold(metric: str, value, field_name: str) -> float:
    value = _finite_float(value, field_name)
    if metric == "acc" and not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be in [0, 1]")
    if metric == "loss" and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value
