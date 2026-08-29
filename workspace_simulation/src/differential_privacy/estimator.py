"""Static privacy-cost estimates derived from a validated simulation config."""

from dataclasses import dataclass
import math

from .accountant import (
    DP_SGD_RDP_ORDERS,
    RDP_ORDERS,
    PerDeviceRDPAccountant,
    PerDeviceSampledGaussianAccountant,
    PrivacyCost,
)


@dataclass(frozen=True)
class ConfigPrivacyEstimate:
    mode: str
    adjacency: str
    participations: int
    uses_configured_round_upper_bound: bool
    mechanisms_per_participation: int
    total_mechanisms: int
    sample_rate: float | None
    expected_batch_size: int | None
    dataset_size: int | None
    noise_multiplier: float
    candidate_orders: tuple[float, ...]
    privacy_cost: PrivacyCost
    continuous_optimal_alpha: float | None
    continuous_epsilon: float | None


@dataclass(frozen=True)
class NoiseMultiplierEstimate:
    target_epsilon: float
    target_delta: float
    required_noise_multiplier: float
    privacy_estimate: ConfigPrivacyEstimate


def estimate_privacy_from_config(
        config,
        *,
        participations=None,
        noise_multiplier=None,
        delta=None,
):
    """Estimate one device's per-run privacy cost without running training.

    With no explicit participation count, every configured global round is
    treated as active for the device. This is an upper bound for configurations
    whose availability process may skip that device in some rounds.
    """
    dp_config = config.differential_privacy
    if not dp_config.enabled:
        raise ValueError("Differential privacy is disabled in the config")

    noise_multiplier = _positive_finite(
        dp_config.noise_multiplier if noise_multiplier is None else noise_multiplier,
        "noise_multiplier",
    )
    delta = _open_probability(
        dp_config.delta if delta is None else delta,
        "delta",
    )
    participations, uses_upper_bound = _resolve_participations(
        config,
        participations,
    )

    if dp_config.mode == "model_update":
        accountant = PerDeviceRDPAccountant(noise_multiplier, delta)
        privacy_cost = accountant.step(
            "estimated_device",
            releases=participations,
        )
        continuous_alpha, continuous_epsilon = _continuous_gaussian_optimum(
            participations,
            noise_multiplier,
            delta,
        )
        return ConfigPrivacyEstimate(
            mode=dp_config.mode,
            adjacency=dp_config.adjacency,
            participations=participations,
            uses_configured_round_upper_bound=uses_upper_bound,
            mechanisms_per_participation=1,
            total_mechanisms=participations,
            sample_rate=None,
            expected_batch_size=None,
            dataset_size=None,
            noise_multiplier=noise_multiplier,
            candidate_orders=accountant.orders,
            privacy_cost=privacy_cost,
            continuous_optimal_alpha=continuous_alpha,
            continuous_epsilon=continuous_epsilon,
        )

    dataset_size = config.dataset.training_data_per_device
    expected_batch_size = min(config.dataset.train_batch_size, dataset_size)
    sample_rate = expected_batch_size / dataset_size
    steps_per_participation = (
        config.training.epoch_per_round
        * config.training.max_batch_per_epoch
    )
    total_steps = participations * steps_per_participation
    accountant = PerDeviceSampledGaussianAccountant(noise_multiplier, delta)
    privacy_cost = accountant.step(
        "estimated_device",
        sample_rate,
        steps=total_steps,
    )
    return ConfigPrivacyEstimate(
        mode=dp_config.mode,
        adjacency=dp_config.adjacency,
        participations=participations,
        uses_configured_round_upper_bound=uses_upper_bound,
        mechanisms_per_participation=steps_per_participation,
        total_mechanisms=total_steps,
        sample_rate=sample_rate,
        expected_batch_size=expected_batch_size,
        dataset_size=dataset_size,
        noise_multiplier=noise_multiplier,
        candidate_orders=accountant.orders,
        privacy_cost=privacy_cost,
        continuous_optimal_alpha=None,
        continuous_epsilon=None,
    )


def solve_noise_multiplier_from_config(
        config,
        target_epsilon,
        *,
        delta=None,
        participations=None,
):
    """Find the smallest sigma meeting a target epsilon for this accountant."""
    dp_config = config.differential_privacy
    if not dp_config.enabled:
        raise ValueError("Differential privacy is disabled in the config")
    target_epsilon = _positive_finite(target_epsilon, "target_epsilon")
    target_delta = _open_probability(
        dp_config.delta if delta is None else delta,
        "delta",
    )
    orders = (
        RDP_ORDERS
        if dp_config.mode == "model_update"
        else DP_SGD_RDP_ORDERS
    )
    conversion_floor = math.log(1.0 / target_delta) / (max(orders) - 1.0)
    if target_epsilon <= conversion_floor:
        raise ValueError(
            f"target_epsilon must exceed {conversion_floor:.10g} for the "
            "current finite RDP order set"
        )

    def estimate(sigma):
        return estimate_privacy_from_config(
            config,
            participations=participations,
            noise_multiplier=sigma,
            delta=target_delta,
        )

    high = max(float(dp_config.noise_multiplier), 1e-6)
    high_estimate = estimate(high)
    while high_estimate.privacy_cost.epsilon > target_epsilon:
        high *= 2.0
        if not math.isfinite(high) or high > 1e12:
            raise RuntimeError("Could not bracket a finite noise multiplier")
        high_estimate = estimate(high)

    low = 0.0
    for _ in range(64):
        middle = (low + high) / 2.0
        middle_estimate = estimate(middle)
        if middle_estimate.privacy_cost.epsilon <= target_epsilon:
            high = middle
            high_estimate = middle_estimate
        else:
            low = middle
        if high - low <= 1e-12 * max(1.0, high):
            break

    return NoiseMultiplierEstimate(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        required_noise_multiplier=high,
        privacy_estimate=high_estimate,
    )


def _resolve_participations(config, participations):
    configured_rounds = config.training.rounds
    uses_upper_bound = participations is None
    if participations is None:
        participations = configured_rounds
    if (
            isinstance(participations, bool)
            or not isinstance(participations, int)
            or participations <= 0
    ):
        raise ValueError("participations must be a positive integer")
    if participations > configured_rounds:
        raise ValueError(
            "participations cannot exceed federation.rounds "
            f"({configured_rounds}) for one experiment run"
        )
    return participations, uses_upper_bound


def _positive_finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _open_probability(value, name):
    result = _positive_finite(value, name)
    if result >= 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return result


def _continuous_gaussian_optimum(release_count, noise_multiplier, delta):
    log_inverse_delta = math.log(1.0 / delta)
    coefficient = release_count / (2.0 * noise_multiplier ** 2)
    alpha = 1.0 + math.sqrt(log_inverse_delta / coefficient)
    epsilon = coefficient * alpha + log_inverse_delta / (alpha - 1.0)
    return alpha, epsilon
