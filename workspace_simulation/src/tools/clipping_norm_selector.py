"""Select candidate clipping norms from non-private calibration norm samples."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

from src.sim_tools.simulation_config import load_simulation_config


CALIBRATION_QUANTILES = (
    ("p50", 0.50),
    ("p75", 0.75),
    ("p80", 0.80),
    ("p90", 0.90),
    ("p95", 0.95),
)


@dataclass(frozen=True)
class ClippingCandidate:
    label: str
    quantile: float
    clipping_norm: float
    observed_clipping_fraction: float
    mechanism_noise_std: float
    post_average_noise_std: float | None
    mean_clipped_norm: float
    mean_retained_ratio: float
    mean_clipping_residual: float
    expected_noise_l2: float | None


@dataclass(frozen=True)
class ClippingNormSummary:
    sample_count: int
    minimum: float
    mean: float
    maximum: float
    current_clipping_norm: float
    current_clipping_fraction: float
    candidates: tuple[ClippingCandidate, ...]
    recommended: ClippingCandidate


def load_norm_samples(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "l2_norm" not in reader.fieldnames:
            raise ValueError("Calibration CSV must contain an l2_norm column")
        values = []
        for row_number, row in enumerate(reader, start=2):
            raw_value = row.get("l2_norm")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid l2_norm at CSV row {row_number}: {raw_value!r}"
                ) from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"l2_norm must be finite and non-negative at row {row_number}"
                )
            values.append(value)
    if not values:
        raise ValueError("Calibration CSV contains no norm samples")
    return values


def summarize_clipping_norms(
        norms,
        dp_config,
        *,
        expected_batch_size=None,
        parameter_count=None,
):
    values = []
    for index, raw_value in enumerate(norms):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid norm at index {index}: {raw_value!r}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Norm at index {index} must be finite and non-negative")
        values.append(value)
    if not values:
        raise ValueError("At least one calibration norm is required")
    values.sort()
    if parameter_count is not None and (
            isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count <= 0
    ):
        raise ValueError("parameter_count must be a positive integer")

    sensitivity_factor = 2.0 if dp_config.mode == "model_update" else 1.0
    if dp_config.mode == "local_dp_sgd":
        if (
                isinstance(expected_batch_size, bool)
                or not isinstance(expected_batch_size, int)
                or expected_batch_size <= 0
        ):
            raise ValueError(
                "local_dp_sgd calibration requires a positive expected_batch_size"
            )
    else:
        expected_batch_size = None

    candidates = []
    for label, quantile in CALIBRATION_QUANTILES:
        clipping_norm = _percentile(values, quantile)
        mechanism_noise_std = (
            sensitivity_factor
            * clipping_norm
            * dp_config.noise_multiplier
        )
        candidates.append(ClippingCandidate(
            label=label,
            quantile=quantile,
            clipping_norm=clipping_norm,
            observed_clipping_fraction=_clipping_fraction(values, clipping_norm),
            mechanism_noise_std=mechanism_noise_std,
            post_average_noise_std=(
                mechanism_noise_std / expected_batch_size
                if expected_batch_size is not None
                else None
            ),
            mean_clipped_norm=statistics.fmean(
                min(value, clipping_norm)
                for value in values
            ),
            mean_retained_ratio=statistics.fmean(
                1.0 if value == 0 else min(1.0, clipping_norm / value)
                for value in values
            ),
            mean_clipping_residual=statistics.fmean(
                max(0.0, value - clipping_norm)
                for value in values
            ),
            expected_noise_l2=(
                (
                    mechanism_noise_std / expected_batch_size
                    if expected_batch_size is not None
                    else mechanism_noise_std
                )
                * math.sqrt(parameter_count)
                if parameter_count is not None
                else None
            ),
        ))

    return ClippingNormSummary(
        sample_count=len(values),
        minimum=values[0],
        mean=statistics.fmean(values),
        maximum=values[-1],
        current_clipping_norm=dp_config.clipping_norm,
        current_clipping_fraction=_clipping_fraction(
            values,
            dp_config.clipping_norm,
        ),
        candidates=tuple(candidates),
        recommended=next(
            candidate for candidate in candidates
            if candidate.label == "p80"
        ),
    )


def _percentile(sorted_values, quantile):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _clipping_fraction(values, clipping_norm):
    return sum(value > clipping_norm for value in values) / len(values)


def _format_number(value):
    return format(float(value), ".10g")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize non-private norm samples and propose clipping_norm "
            "candidates for one DP config."
        )
    )
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument(
        "--norm-file",
        required=True,
        type=Path,
        help="CSV containing an l2_norm column",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_simulation_config(args.config_file)
        dp_config = config.differential_privacy
        if not dp_config.enabled:
            raise ValueError("Differential privacy is disabled in the config")
        norms = load_norm_samples(args.norm_file)
        expected_batch_size = (
            min(
                config.dataset.train_batch_size,
                config.dataset.training_data_per_device,
            )
            if dp_config.mode == "local_dp_sgd"
            else None
        )
        summary = summarize_clipping_norms(
            norms,
            dp_config,
            expected_batch_size=expected_batch_size,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    semantics = (
        "whole-model client update L2 norm"
        if dp_config.mode == "model_update"
        else "per-sample whole-model gradient L2 norm"
    )
    print(f"config: {args.config_file.resolve()}")
    print(f"norm_file: {args.norm_file.resolve()}")
    print(f"mode: {dp_config.mode}")
    print(f"required_norm_semantics: {semantics}")
    print("calibration_scope: non-private/public-or-independent data")
    print(f"sample_count: {summary.sample_count}")
    print(f"minimum_norm: {_format_number(summary.minimum)}")
    print(f"mean_norm: {_format_number(summary.mean)}")
    print(f"maximum_norm: {_format_number(summary.maximum)}")
    print(f"current_clipping_norm: {_format_number(summary.current_clipping_norm)}")
    print(
        "current_observed_clipping_fraction: "
        f"{_format_number(summary.current_clipping_fraction)}"
    )
    for candidate in summary.candidates:
        prefix = f"candidate_{candidate.label}"
        print(f"{prefix}_clipping_norm: {_format_number(candidate.clipping_norm)}")
        print(
            f"{prefix}_observed_clipping_fraction: "
            f"{_format_number(candidate.observed_clipping_fraction)}"
        )
        print(
            f"{prefix}_mechanism_noise_std: "
            f"{_format_number(candidate.mechanism_noise_std)}"
        )
        print(
            f"{prefix}_mean_clipped_norm: "
            f"{_format_number(candidate.mean_clipped_norm)}"
        )
        print(
            f"{prefix}_mean_retained_ratio: "
            f"{_format_number(candidate.mean_retained_ratio)}"
        )
        print(
            f"{prefix}_mean_clipping_residual: "
            f"{_format_number(candidate.mean_clipping_residual)}"
        )
        if candidate.post_average_noise_std is not None:
            print(
                f"{prefix}_post_average_noise_std: "
                f"{_format_number(candidate.post_average_noise_std)}"
            )
    print("recommended_starting_quantile: p80")
    print(
        "recommended_starting_clipping_norm: "
        f"{_format_number(summary.recommended.clipping_norm)}"
    )
    if summary.sample_count < 100:
        print("warning: fewer than 100 norm samples; quantiles may be unstable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
