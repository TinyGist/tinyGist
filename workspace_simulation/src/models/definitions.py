PARAMETER_SCOPES = {"all", "conv", "fc", "custom"}
CONV_BLOCK_MODES = {"channel", "kernel", "layer"}
BN_AGGREGATION_MODES = {"none", "affine", "affine_running_stats"}
BN_PROCESS_MODES = {"base_unit", "separate_per_segment", "separate_per_recipient"}
BLOCK_REFINEMENT_TARGETS = {"linear", "pointwise"}
BLOCK_BIAS_MODES = {"each_chunk", "separate", "local"}
BLOCK_REFINEMENT_BIAS_MODES = BLOCK_BIAS_MODES

DEFAULT_BLOCK_REFINEMENT = {
    "enabled": False,
    "targets": (),
    "linear_chunk_size": 0,
    "pointwise_chunk_size": 0,
    "bias": "each_chunk",
}

MODEL_REGISTRATION_NAMES = {
    "FCN": "FCN",
    "DeepFC": "DeepFC",
    "BasicConv": "BasicConv",
    "LeNet5": "LeNet5",
    "ConvolutionalNet": "ConvolutionalNet",
    "MobileNetV1Small": "MobileNetV1Small",
    "MobileNetV1SmallNoBN": "MobileNetV1SmallNoBN",
    "MobileNetV1SmallGroupNorm": "MobileNetV1SmallGroupNorm",
    "MobileNetV1SmallLayerNorm": "MobileNetV1SmallLayerNorm",
    "MobileNetV4Small": "MobileNetV4Small",
    "MobileNetV4SmallGroupNorm": "MobileNetV4SmallGroupNorm",
    "MobileNetV4SmallLayerNorm": "MobileNetV4SmallLayerNorm",
    "MobileNetV2Small": "MobileNetV2Small",
    "MobileNetV2SmallGroupNorm": "MobileNetV2SmallGroupNorm",
    "MobileNetV2SmallLayerNorm": "MobileNetV2SmallLayerNorm",
    "MobileNetV2Baseline": "MobileNetV2Baseline",
    "MobileNetV2BaselineGroupNorm": "MobileNetV2BaselineGroupNorm",
    "MobileNetV2BaselineLayerNorm": "MobileNetV2BaselineLayerNorm",
    "MobileNetV2Alpha035": "MobileNetV2Alpha035",
    "MobileNetV2Alpha035GroupNorm": "MobileNetV2Alpha035GroupNorm",
    "MobileNetV2Alpha035LayerNorm": "MobileNetV2Alpha035LayerNorm",
    "FOMOMNv2Baseline": "FOMOMNv2Baseline",
    "FOMOMNv2BaselineGroupNorm": "FOMOMNv2BaselineGroupNorm",
    "FOMOMNv2BaselineLayerNorm": "FOMOMNv2BaselineLayerNorm",
    "FOMOMNv2Alpha035": "FOMOMNv2Alpha035",
    "FOMOMNv2Alpha035GroupNorm": "FOMOMNv2Alpha035GroupNorm",
    "FOMOMNv2Alpha035LayerNorm": "FOMOMNv2Alpha035LayerNorm",
    "uYOLO": "MircoYOLO",
    "uYOLOGroupNorm": "MircoYOLOGroupNorm",
    "uYOLOLayerNorm": "MircoYOLOLayerNorm",
}

CRITERIA_REGISTRATION_NAMES = {
    "fomo_person_loss": "FOMOLossPerson",
    "fomo_vehicle_loss": "FOMOLossVehicle",
    "fomo_vehicle_binary_loss": "FOMOLossVehicleBinary",
    "fomo_person_metrics": "FOMOMetricsPerson",
    "fomo_vehicle_metrics": "FOMOMetricsVehicle",
    "fomo_vehicle_binary_metrics": "FOMOMetricsVehicleBinary",
    "yolo_person_loss": "YoLoLossPerson",
    "yolo_vehicle_loss": "YoLoLossVehicle",
    "yolo_vehicle_binary_loss": "YoLoLossVehicleBinary",
    "yolo_person_map": "YoLoMAPPerson",
    "yolo_vehicle_map": "YoLoMAPVehicle",
    "yolo_vehicle_binary_map": "YoLoMAPVehicleBinary",
}


def normalize_definition_key(value) -> str:
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def _canonical_definition_value(value, supported_values, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    key = normalize_definition_key(value)
    if key not in supported_values:
        raise ValueError(
            f"Invalid {field_name} [{value}], "
            f"supported values are {sorted(supported_values)}"
        )
    return key


def canonical_parameter_scope(parameter_scope: str) -> str:
    return _canonical_definition_value(
        parameter_scope,
        PARAMETER_SCOPES,
        "parameter_scope",
    )


def canonical_bn_aggregation_mode(bn_mode) -> str:
    return _canonical_definition_value(
        bn_mode,
        BN_AGGREGATION_MODES,
        "bn_mode",
    )


def canonical_bn_process_mode(process_mode) -> str:
    if isinstance(process_mode, bool):
        return "base_unit" if process_mode else "separate_per_segment"
    return _canonical_definition_value(
        process_mode,
        BN_PROCESS_MODES,
        "BN process_as_base_unit mode",
    )


def normalize_channel_length(channel_length):
    if channel_length is None:
        return None
    channel_length = int(channel_length)
    if channel_length < 0:
        raise ValueError("channel_length must be greater than or equal to 0")
    if channel_length == 0:
        return None
    return channel_length


def input_channel_slices(input_channel_count: int, channel_length=None):
    channel_length = normalize_channel_length(channel_length)
    if channel_length is None or channel_length >= input_channel_count:
        return [slice(0, input_channel_count)]
    return [
        slice(start, min(start + channel_length, input_channel_count))
        for start in range(0, input_channel_count, channel_length)
    ]


def fixed_chunk_slices(length: int, chunk_size: int):
    length = int(length)
    chunk_size = int(chunk_size)
    if length < 0:
        raise ValueError("chunk length must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if length == 0:
        return []
    return [
        slice(start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def canonical_block_refinement_target(target) -> str:
    return _canonical_definition_value(
        target,
        BLOCK_REFINEMENT_TARGETS,
        "block refinement target",
    )


def canonical_block_refinement_targets(
        targets,
        enabled=False,
) -> tuple[str, ...]:
    if targets is None:
        targets = ("linear", "pointwise") if enabled else ()
    if not isinstance(targets, (list, tuple, set)):
        raise ValueError("block refinement targets must be a list")
    canonical_targets = []
    for target in targets:
        canonical_target = canonical_block_refinement_target(target)
        if canonical_target not in canonical_targets:
            canonical_targets.append(canonical_target)
    return tuple(canonical_targets)


def canonical_block_bias_mode(bias_mode) -> str:
    return _canonical_definition_value(
        bias_mode,
        BLOCK_BIAS_MODES,
        "block bias mode",
    )


def canonical_block_refinement_enabled(value) -> bool:
    if not isinstance(value, bool):
        raise ValueError("block refinement enabled must be a boolean")
    return value


def canonical_block_refinement_config(value=None) -> dict:
    config = dict(DEFAULT_BLOCK_REFINEMENT)
    if value is None:
        return config
    if not isinstance(value, dict):
        raise ValueError("block refinement config must be a mapping or None")

    supported_keys = {
        "enabled",
        "targets",
        "linear_chunk_size",
        "pointwise_chunk_size",
        "bias",
    }
    unknown = sorted(set(value) - supported_keys)
    if unknown:
        raise ValueError(f"Unsupported block refinement key(s): {unknown}")

    enabled = canonical_block_refinement_enabled(value.get("enabled", False))
    targets = canonical_block_refinement_targets(
        value.get("targets"),
        enabled=enabled,
    )
    linear_chunk_size = int(
        value.get("linear_chunk_size", 16 if enabled else 0)
    )
    pointwise_chunk_size = int(
        value.get("pointwise_chunk_size", 16 if enabled else 0)
    )
    if enabled and not targets:
        raise ValueError("block refinement requires at least one target")
    if linear_chunk_size < 0 or pointwise_chunk_size < 0:
        raise ValueError("block refinement chunk sizes must be non-negative")
    if enabled and "linear" in targets and linear_chunk_size <= 0:
        raise ValueError(
            "linear block refinement requires linear_chunk_size > 0"
        )
    if enabled and "pointwise" in targets and pointwise_chunk_size <= 0:
        raise ValueError(
            "pointwise block refinement requires pointwise_chunk_size > 0"
        )

    config.update({
        "enabled": enabled,
        "targets": targets,
        "linear_chunk_size": linear_chunk_size,
        "pointwise_chunk_size": pointwise_chunk_size,
        "bias": canonical_block_bias_mode(value.get("bias", "each_chunk")),
    })
    return config


def block_refinement_enabled(config, target: str) -> bool:
    config = canonical_block_refinement_config(config)
    target = canonical_block_refinement_target(target)
    return bool(config["enabled"] and target in config["targets"])
