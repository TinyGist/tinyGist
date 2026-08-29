"""Strict, standalone configuration for differential privacy mechanisms."""

from dataclasses import dataclass
import math
from numbers import Real


MODEL_UPDATE_REPLACE_ONE_SENSITIVITY_FACTOR = 2.0


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    enabled: bool = False
    mode: str = "model_update"
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5

    @property
    def adjacency(self) -> str:
        if self.mode == "model_update":
            return "device_update_replace_one"
        return "sample_add_remove_within_device"

    @property
    def l2_sensitivity(self) -> float:
        if self.mode == "model_update":
            return (
                MODEL_UPDATE_REPLACE_ONE_SENSITIVITY_FACTOR
                * self.clipping_norm
            )
        return self.clipping_norm

    @property
    def noise_std(self) -> float:
        return self.l2_sensitivity * self.noise_multiplier


def parse_differential_privacy_config(raw) -> DifferentialPrivacyConfig:
    if raw is None:
        return DifferentialPrivacyConfig()
    if not isinstance(raw, dict):
        raise ValueError("differential_privacy must be a mapping or null")

    supported_keys = {"mode", "clipping_norm", "noise_multiplier", "delta"}
    unknown = sorted(set(raw) - supported_keys)
    if unknown:
        raise ValueError(
            f"Unsupported key(s) in differential_privacy: {unknown}"
        )

    mode = str(_required(raw, "mode")).strip().lower().replace("-", "_")
    if mode not in {"model_update", "local_dp_sgd"}:
        raise ValueError(
            "differential_privacy.mode must be model_update or local_dp_sgd"
        )
    return DifferentialPrivacyConfig(
        enabled=True,
        mode=mode,
        clipping_norm=_positive_float(
            _required(raw, "clipping_norm"),
            "differential_privacy.clipping_norm",
        ),
        noise_multiplier=_positive_float(
            _required(raw, "noise_multiplier"),
            "differential_privacy.noise_multiplier",
        ),
        delta=_probability_open_interval(
            _required(raw, "delta"),
            "differential_privacy.delta",
        ),
    )


def _required(section, key):
    if key not in section:
        raise ValueError(f"Missing required config key differential_privacy.{key}")
    return section[key]


def _positive_float(value, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path} must be a positive finite number")
    return result


def _probability_open_interval(value, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be strictly between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise ValueError(f"{path} must be strictly between 0 and 1")
    return result
