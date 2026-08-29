"""Run-directory naming helpers independent of simulation state."""

import hashlib
import re


RUN_DIR_HASH_LENGTH = 12
RUN_DIR_PREFIX_BUDGET_FRACTION = 0.28
TRAILING_RUN_HASH_PATTERN = re.compile(
    rf"^(?P<core>.*)_h(?P<digest>[0-9a-f]{{{RUN_DIR_HASH_LENGTH}}})$",
    re.IGNORECASE,
)
PARALLEL_RUN_LABEL_PATTERN = re.compile(
    r"^seed_index(?P<seed_index>\d+)_repeat(?P<repeat>\d+)_seed(?P<seed>-?\d+)$"
)
RUN_CONFIG_SIGNAL_PATTERNS = (
    ("ll2", re.compile(r"(?:^|_)local_l2_([0-9]+(?:_[0-9]+)*)")),
    ("al2", re.compile(r"(?:^|_)agg_update_l2_([0-9]+(?:_[0-9]+)*)")),
    ("eps", re.compile(r"(?:^|_)eps([0-9]+(?:_[0-9]+)*)")),
    ("delta", re.compile(r"(?:^|_)delta(1e-\d+)")),
)


def sanitize_path_part(raw_value: str, max_characters: int | None = None) -> str:
    safe_value = "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_"
        for char in str(raw_value).strip()
    ).strip("._-")
    if max_characters is not None:
        max_characters = int(max_characters)
        if max_characters <= 0:
            return ""
        if len(safe_value) > max_characters:
            if max_characters < 3:
                return safe_value[-max_characters:]
            head_length = (max_characters - 1) // 2
            tail_length = max_characters - head_length - 1
            safe_value = (
                f"{safe_value[:head_length]}_"
                f"{safe_value[-tail_length:]}"
            )
    return safe_value


def path_component_hash(component: str) -> str:
    return hashlib.sha1(component.encode("utf-8")).hexdigest()[:RUN_DIR_HASH_LENGTH]


def _token_boundary_prefix(encoded: bytes, byte_budget: int) -> str:
    end = min(byte_budget, len(encoded))
    prefix = encoded[:end].decode("utf-8", errors="ignore")
    if end < len(encoded) and encoded[end:end + 1] != b"_":
        boundary = prefix.rfind("_")
        if boundary > 0:
            prefix = prefix[:boundary]
    return prefix.rstrip("._-+")


def _token_boundary_suffix(encoded: bytes, byte_budget: int) -> str:
    start = max(0, len(encoded) - byte_budget)
    suffix = encoded[start:].decode("utf-8", errors="ignore")
    if start > 0 and encoded[start - 1:start] != b"_":
        boundary = suffix.find("_")
        if 0 <= boundary < len(suffix) - 1:
            suffix = suffix[boundary + 1:]
    return suffix.lstrip("._-+")


def compact_run_label(run_label: str) -> str:
    match = PARALLEL_RUN_LABEL_PATTERN.fullmatch(run_label)
    if match:
        return (
            f"si{match.group('seed_index')}_"
            f"r{match.group('repeat')}_"
            f"s{match.group('seed')}"
        )
    return sanitize_path_part(run_label, 24)


def summarize_run_config_stem(config_stem: str) -> str:
    """Keep the readable experiment identity and compact sweep values."""
    encoded = config_stem.encode("utf-8")
    if len(encoded) <= 48:
        return config_stem
    signals = []
    for label, pattern in RUN_CONFIG_SIGNAL_PATTERNS:
        match = pattern.search(config_stem)
        if match:
            separator = "_" if label in {"ll2", "al2"} else ""
            signals.append(f"{label}{separator}{match.group(1)}")
    prefix_budget = 24 if len(signals) >= 2 else 30
    prefix = _token_boundary_prefix(encoded, prefix_budget)
    return "_".join([prefix, *signals])


def shorten_path_component(component: str, max_length: int) -> str:
    max_bytes = max(80, min(int(max_length), 255))
    encoded = component.encode("utf-8")
    if len(encoded) <= max_bytes:
        return component

    trailing_hash = TRAILING_RUN_HASH_PATTERN.fullmatch(component)
    if trailing_hash:
        component = trailing_hash.group("core")
        digest = trailing_hash.group("digest").lower()
    else:
        digest = path_component_hash(component)
    marker = f"_h{digest}_"
    marker_bytes = len(marker.encode("utf-8"))
    budget = max_bytes - marker_bytes
    head_budget = max(1, int(budget * RUN_DIR_PREFIX_BUDGET_FRACTION))
    tail_budget = max(1, budget - head_budget)
    core_encoded = component.encode("utf-8")
    head = _token_boundary_prefix(core_encoded, head_budget)
    tail = _token_boundary_suffix(core_encoded, tail_budget)
    return f"{head}{marker}{tail}"
