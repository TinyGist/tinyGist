"""Report privacy cost or solve sigma from a target epsilon and delta."""

import argparse
import math
from pathlib import Path

from src.differential_privacy.estimator import (
    estimate_privacy_from_config,
    solve_noise_multiplier_from_config,
)
from src.sim_tools.simulation_config import load_simulation_config


def _positive_int(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _probability(value):
    result = _positive_float(value)
    if result >= 1:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return result


def _format_number(value):
    return format(float(value), ".10g")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate one device's privacy cost from a validated config, or "
            "solve the noise multiplier required by a target epsilon."
        )
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=Path,
        help="schema-v2 YAML experiment config",
    )
    parser.add_argument(
        "--participations",
        type=_positive_int,
        help=(
            "known active rounds for one device; defaults to federation.rounds "
            "as a per-run upper bound"
        ),
    )
    parser.add_argument(
        "--delta",
        type=_probability,
        help="override differential_privacy.delta for this calculation",
    )
    parser.add_argument(
        "--target-epsilon",
        type=_positive_float,
        help="also solve the minimum noise_multiplier meeting this epsilon",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_simulation_config(args.config_file)
        estimate = estimate_privacy_from_config(
            config,
            participations=args.participations,
            delta=args.delta,
        )
        solution = (
            solve_noise_multiplier_from_config(
                config,
                args.target_epsilon,
                delta=args.delta,
                participations=args.participations,
            )
            if args.target_epsilon is not None
            else None
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    dp_config = config.differential_privacy
    cost = estimate.privacy_cost
    basis = (
        "configured-round upper bound"
        if estimate.uses_configured_round_upper_bound
        else "explicit active-round count"
    )
    print(f"config: {args.config_file.resolve()}")
    print(f"mode: {estimate.mode}")
    print(f"adjacency: {estimate.adjacency}")
    print(f"participation_basis: {basis}")
    print(f"participations_per_device: {estimate.participations}")
    print(f"mechanisms_per_participation: {estimate.mechanisms_per_participation}")
    print(f"total_releases_or_steps: {estimate.total_mechanisms}")
    if estimate.sample_rate is not None:
        print(f"dataset_size_per_device: {estimate.dataset_size}")
        print(f"expected_batch_size: {estimate.expected_batch_size}")
        print(f"sample_rate: {_format_number(estimate.sample_rate)}")
    print(f"clipping_norm: {_format_number(dp_config.clipping_norm)}")
    print(f"l2_sensitivity: {_format_number(dp_config.l2_sensitivity)}")
    print(f"noise_multiplier: {_format_number(estimate.noise_multiplier)}")
    print(
        "noise_std: "
        f"{_format_number(dp_config.l2_sensitivity * estimate.noise_multiplier)}"
    )
    print(f"delta: {_format_number(cost.delta)}")
    print(f"candidate_order_count: {len(estimate.candidate_orders)}")
    print(f"optimal_alpha: {_format_number(cost.optimal_alpha)}")
    print(f"epsilon: {_format_number(cost.epsilon)}")
    if estimate.continuous_optimal_alpha is not None:
        print(
            "continuous_optimal_alpha: "
            f"{_format_number(estimate.continuous_optimal_alpha)}"
        )
        print(
            "continuous_epsilon: "
            f"{_format_number(estimate.continuous_epsilon)}"
        )
    if solution is not None:
        solved = solution.privacy_estimate
        print(f"requested_epsilon: {_format_number(solution.target_epsilon)}")
        print(f"target_delta: {_format_number(solution.target_delta)}")
        print(
            "required_noise_multiplier: "
            f"{_format_number(solution.required_noise_multiplier)}"
        )
        print(
            "required_noise_std: "
            f"{_format_number(dp_config.l2_sensitivity * solution.required_noise_multiplier)}"
        )
        print(f"required_optimal_alpha: {_format_number(solved.privacy_cost.optimal_alpha)}")
        print(f"achieved_epsilon: {_format_number(solved.privacy_cost.epsilon)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
