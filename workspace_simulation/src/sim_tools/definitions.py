_MISSING = object()

METHOD_NAMES = {
    "centralized": "Centralized",
    "segmentpulling": "SegmentPulling",
    "gist": "Gist",
    "dfa": "DFA",
    "sdfa": "SDFA",
}

METHOD_SLUGS = {
    "Centralized": "centralized",
    "SegmentPulling": "segment_pulling",
    "Gist": "gist",
    "DFA": "dfa",
    "SDFA": "sdfa",
}

GIST_FAMILY_METHODS = {"Gist", "DFA", "SDFA"}

EXPERIMENT_TASKS = {
    "classification": "Classification",
    "object_detection": "Object_Detection",
}

FISHER_CAL_METHODS = {"square", "abs"}
FISHER_GRANULARITIES = {"parameter", "block"}
FISHER_BLOCK_REDUCE_METHODS = {"mean", "rms", "l2"}

UTILS_LAZY_IMPORTS = {
    "AdaptiveDFLLearningRate": ("src.utils.adaptive_loss_stair_tool", "AdaptiveDFLLearningRate"),
    "CommunicationSimulator": ("src.utils.communication_simulator_tool", "CommunicationSimulator"),
    "CommunicationRecorder": ("src.utils.communication_recorder", "CommunicationRecorder"),
    "PacketPayload": ("src.utils.communication_recorder", "PacketPayload"),
    "StaleTrainingSimulator": ("src.utils.stale_training_tool", "StaleTrainingSimulator"),
    "EarlyStopController": ("src.utils.early_stop", "EarlyStopController"),
    "config_abbreviation": ("src.sim_tools.config_abbreviation", "config_abbreviation"),
    "get_default_device": ("src.sim_tools.device", "get_default_device"),
    "pick_gpu_with_most_free_mem": ("src.sim_tools.device", "pick_gpu_with_most_free_mem"),
    "round_filter": ("src.sim_tools.logging_utils", "round_filter"),
    "setup_round_logging": ("src.sim_tools.logging_utils", "setup_round_logging"),
    "build_detection_metrics": ("src.utils.object_detection_tools", "build_detection_metrics"),
    "infer_object_detection_task": ("src.utils.object_detection_tools", "infer_object_detection_task"),
    "SimulationConfig": ("src.sim_tools.simulation_config", "SimulationConfig"),
    "load_simulation_config": ("src.sim_tools.simulation_config", "load_simulation_config"),
    "MetricValues": ("src.sim_tools.simulation_metrics", "MetricValues"),
    "SimulationMetricsRecorder": ("src.sim_tools.simulation_metrics", "SimulationMetricsRecorder"),
    "FederatedLearningSim": ("src.sim_tools.simulation_manager_tool", "FederatedLearningSim"),
}

OBJECT_DETECTION_METRIC_IMPORTS = {
    "yolo": {
        ("vehicle", False): ("src.models", "YoLoMAPVehicle"),
        ("vehicle", True): ("src.models", "YoLoMAPVehicleBinary"),
        ("person", False): ("src.models", "YoLoMAPPerson"),
    },
    "fomo": {
        ("vehicle", False): ("src.models", "FOMOMetricsVehicle"),
        ("vehicle", True): ("src.models", "FOMOMetricsVehicleBinary"),
        ("person", False): ("src.models", "FOMOMetricsPerson"),
    },
}

LOG_VALUE_SLUGS = {
    "Centralized": "cen",
    "SegmentPulling": "sp",
    "DFA": "dfa",
    "SDFA": "sdfa",
    "Gist": "gist",
    "parameter": "param",
    "kernel": "kernel",
    "channel": "channel",
    "layer": "layer",
    "consistent": "cons",
    "random_same": "rsame",
    "random_each": "reach",
    "importance": "imp",
    "uniform": "uni",
    "probabilistic": "prob",
    "uniform_with_replacement": "uwrep",
    "uniform_without_replacement": "uworep",
    "grouped_probabilistic": "gprob",
    "grouped_unique_probabilistic": "guprob",
    "grouped_round_robin": "grr",
    "sensitivity_aligned": "sensalign",
    "sensitivity_diverse": "sensdiv",
    "weight_abs": "wabs",
    "weight_abs_ema_online": "wema",
    "gradient_abs_ema_online": "gradema",
    "gradient_abs_round_step_ema_online": "gradrema",
    "gradient_abs_post": "gradpost",
    "gradient_weight_abs_post": "gradwpost",
    "gradient_signal_preservation_post": "gradsigpost",
    "fisher_diagonal_ema_online": "fisherema",
    "fisher_empirical_diagonal_post": "fishempdiag",
    "fisher_empirical_diagonal_weight_post": "fishempwdiag",
    "taylor_first_abs_step_ema_online": "tfirststep",
    "taylor_first_abs_current_online": "tfirstcur",
    "taylor_second_step_ema_online": "tsecondstep",
    "fisher_taylor_second_current_online": "ftsecondcur",
    "hessian_taylor_second_exact_post": "htsecondexact",
    "hessian_taylor_second_step_ema_online": "htsecondstep",
    "hessian_taylor_second_current_online": "htsecondcur",
    "hutchinson_diagonal_post": "hutchdiag",
    "hutchinson_diagonal_weight_post": "hutchwdiag",
    "mean_abs": "mag",
    "rms": "rms",
    "lipschitz": "lip",
    "fisher_lipschitz_cooperation": "fishlipcoop",
    "fisher": "fisherema",
    "fisher_lipschitz": "fishlip",
    "val_acc": "val",
    "square": "sq",
    "abs": "abs",
    "block": "block",
    "mean": "mean",
    "l2": "l2",
}


def normalize_definition_key(value) -> str:
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def log_value_slug(value) -> str:
    return LOG_VALUE_SLUGS.get(value, str(value).replace(" ", "_"))


def canonical_method_name(method_name: str) -> str:
    key = normalize_definition_key(method_name)
    try:
        return METHOD_NAMES[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid method_name [{method_name}], "
            f"supported methods are {sorted(METHOD_SLUGS)}"
        ) from exc


def canonical_experiment_task(task: str) -> str:
    key = normalize_definition_key(task)
    try:
        return EXPERIMENT_TASKS[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid experiment task [{task}], supported tasks are "
            f"{sorted(EXPERIMENT_TASKS.values())}"
        ) from exc


def _strict_fisher_value(value, supported_values, field_name: str) -> str:
    key = normalize_definition_key(value)
    if key not in supported_values:
        raise ValueError(
            f"Invalid {field_name} [{value}], "
            f"supported values are {sorted(supported_values)}"
        )
    return key


def canonical_fisher_cal(fisher_cal: str) -> str:
    return _strict_fisher_value(
        fisher_cal,
        FISHER_CAL_METHODS,
        "fisher_cal",
    )


def canonical_fisher_granularity(fisher_granularity) -> str:
    return _strict_fisher_value(
        fisher_granularity,
        FISHER_GRANULARITIES,
        "fisher granularity",
    )


def canonical_fisher_block_reduce_method(reduce_method) -> str:
    return _strict_fisher_value(
        reduce_method,
        FISHER_BLOCK_REDUCE_METHODS,
        "fisher block_reduce_method",
    )
