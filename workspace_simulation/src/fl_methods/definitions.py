SEGMENT_CREATE_METHODS = {"random_same", "random_each", "consistent", "importance"}
SEGMENT_CHOSEN_METHODS = {"uniform", "probabilistic"}
SEGMENT_UNITS = {"parameter", "kernel", "channel", "layer"}
BLOCK_SEGMENT_UNITS = {"kernel", "channel", "layer"}
REFINABLE_SEGMENT_UNITS = {"kernel", "channel"}
GROUPABLE_SEGMENT_UNITS = {"kernel", "channel"}
BLOCK_SCORE_METHODS = {"mean_abs", "rms", "lipschitz", "fisher_lipschitz_cooperation"}
MODEL_BLOCK_SCORE_METHODS = {"mean_abs", "rms", "lipschitz"}
LIPSCHITZ_BLOCK_SCORE_METHODS = {"lipschitz", "fisher_lipschitz_cooperation"}
AGGREGATION_SCORE_SOURCES = {"uniform", "val_acc", "fisher", "fisher_lipschitz"}
FISHER_AGGREGATION_SCORE_SOURCES = {"fisher", "fisher_lipschitz"}
BLOCK_ALIGNED_AGGREGATION_SCORE_SOURCES = {"fisher_lipschitz"}

FL_METHOD_CLASS_NAMES = {
    "Centralized": "Centralized",
    "SegmentPulling": "SegmentPulling",
    "Gist": "Gist",
    "DFA": "DFA",
    "SDFA": "SDFA",
}

def _canonical_value(value, supported_values, field_name):
    value = str(value).strip().lower()
    if value not in supported_values:
        raise ValueError(
            f"Invalid {field_name} [{value}], supported values are {sorted(supported_values)}"
        )
    return value


def canonical_segment_unit(segment_unit: str) -> str:
    return _canonical_value(segment_unit, SEGMENT_UNITS, "segment unit")


def canonical_block_score_method(block_score_method: str) -> str:
    return _canonical_value(block_score_method, BLOCK_SCORE_METHODS, "block score method")


def canonical_channel_length(channel_length) -> int:
    if channel_length is None:
        return 0
    channel_length = int(channel_length)
    if channel_length < 0:
        raise ValueError("channel_length must be greater than or equal to 0")
    return channel_length


def canonical_aggregation_score_source(aggregation_score_source: str) -> str:
    return _canonical_value(
        aggregation_score_source,
        AGGREGATION_SCORE_SOURCES,
        "aggregation score source",
    )
