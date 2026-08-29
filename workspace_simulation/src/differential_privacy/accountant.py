"""Per-device Renyi-DP accounting for the supported Gaussian mechanisms."""

from dataclasses import dataclass
from functools import lru_cache
import math


RDP_ORDERS = (
    1.25, 1.5, 1.75,
    2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 24.0,
    32.0, 48.0, 64.0, 96.0, 128.0, 256.0,
)

DP_SGD_RDP_ORDERS = (
    2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 24.0,
    32.0, 48.0, 64.0, 96.0, 128.0, 256.0,
)


@dataclass(frozen=True)
class PrivacyCost:
    device_id: str
    release_count: int
    epsilon: float
    delta: float
    optimal_alpha: float
    rdp_values: tuple[float, ...]


def _validate_accountant_inputs(
        noise_multiplier,
        delta,
        orders,
        *,
        integer_orders=False,
):
    noise_multiplier = float(noise_multiplier)
    delta = float(delta)
    orders = tuple(float(order) for order in orders)
    if noise_multiplier <= 0 or not math.isfinite(noise_multiplier):
        raise ValueError("noise_multiplier must be a positive finite number")
    if not 0 < delta < 1:
        raise ValueError("delta must be strictly between 0 and 1")
    if not orders or any(
            not math.isfinite(order)
            or order <= 1
            or (integer_orders and not order.is_integer())
            for order in orders
    ):
        requirement = (
            "integers greater than 1"
            if integer_orders
            else "finite and greater than 1"
        )
        raise ValueError(f"RDP orders must be {requirement}")
    return noise_multiplier, delta, orders


def _privacy_cost(
        device_id,
        release_count,
        delta,
        log_inverse_delta,
        orders,
        rdp_values,
):
    if release_count == 0:
        return PrivacyCost(
            device_id=str(device_id),
            release_count=0,
            epsilon=0.0,
            delta=delta,
            optimal_alpha=orders[-1],
            rdp_values=rdp_values,
        )
    epsilon_candidates = tuple(
        rdp + log_inverse_delta / (order - 1.0)
        for order, rdp in zip(orders, rdp_values)
    )
    optimal_index = min(
        range(len(epsilon_candidates)),
        key=epsilon_candidates.__getitem__,
    )
    return PrivacyCost(
        device_id=str(device_id),
        release_count=release_count,
        epsilon=epsilon_candidates[optimal_index],
        delta=delta,
        optimal_alpha=orders[optimal_index],
        rdp_values=rdp_values,
    )


class PerDeviceRDPAccountant:
    def __init__(self, noise_multiplier, delta, orders=RDP_ORDERS):
        (
            self.noise_multiplier,
            self.delta,
            self.orders,
        ) = _validate_accountant_inputs(noise_multiplier, delta, orders)
        self._log_inverse_delta = math.log(1.0 / self.delta)
        self._rdp_per_release = tuple(
            order / (2.0 * self.noise_multiplier ** 2)
            for order in self.orders
        )
        self._release_counts = {}

    def step(self, device_id, releases=1) -> PrivacyCost:
        if (
                isinstance(releases, bool)
                or not isinstance(releases, int)
                or releases <= 0
        ):
            raise ValueError("releases must be a positive integer")
        device_id = str(device_id)
        release_count = self._release_counts.get(device_id, 0) + releases
        self._release_counts[device_id] = release_count
        rdp_values = tuple(
            release_count * value
            for value in self._rdp_per_release
        )
        return _privacy_cost(
            device_id,
            release_count,
            self.delta,
            self._log_inverse_delta,
            self.orders,
            rdp_values,
        )

    def release_count(self, device_id) -> int:
        return self._release_counts.get(str(device_id), 0)

    def privacy_cost(self, device_id) -> PrivacyCost:
        device_id = str(device_id)
        release_count = self.release_count(device_id)
        rdp_values = tuple(
            release_count * value
            for value in self._rdp_per_release
        )
        return _privacy_cost(
            device_id,
            release_count,
            self.delta,
            self._log_inverse_delta,
            self.orders,
            rdp_values,
        )


@lru_cache(maxsize=None)
def _log_binomial_coefficients(integer_order):
    return tuple(
        math.lgamma(integer_order + 1)
        - math.lgamma(count + 1)
        - math.lgamma(integer_order - count + 1)
        for count in range(integer_order + 1)
    )


def sampled_gaussian_rdp(order, sample_rate, noise_multiplier):
    """Exact integer-order RDP for the Poisson-sampled Gaussian mechanism."""
    order = float(order)
    sample_rate = float(sample_rate)
    noise_multiplier = float(noise_multiplier)
    if not order.is_integer() or order <= 1:
        raise ValueError("Sampled-Gaussian RDP requires integer orders greater than 1")
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if noise_multiplier <= 0 or not math.isfinite(noise_multiplier):
        raise ValueError("noise_multiplier must be a positive finite number")
    if sample_rate == 1:
        return order / (2.0 * noise_multiplier ** 2)

    integer_order = int(order)
    log_q = math.log(sample_rate)
    log_one_minus_q = math.log1p(-sample_rate)
    log_terms = []
    for count, log_binomial in enumerate(
            _log_binomial_coefficients(integer_order)):
        privacy_term = (
            count * (count - 1)
            / (2.0 * noise_multiplier ** 2)
        )
        log_terms.append(
            log_binomial
            + count * log_q
            + (integer_order - count) * log_one_minus_q
            + privacy_term
        )
    maximum = max(log_terms)
    log_moment = maximum + math.log(
        sum(math.exp(term - maximum) for term in log_terms)
    )
    return log_moment / (integer_order - 1)


class PerDeviceSampledGaussianAccountant:
    """Compose Poisson-sampled Gaussian DP-SGD steps independently per device."""

    def __init__(self, noise_multiplier, delta, orders=DP_SGD_RDP_ORDERS):
        (
            self.noise_multiplier,
            self.delta,
            self.orders,
        ) = _validate_accountant_inputs(
            noise_multiplier,
            delta,
            orders,
            integer_orders=True,
        )
        self._log_inverse_delta = math.log(1.0 / self.delta)
        self._zero_rdp = tuple(0.0 for _ in self.orders)
        self._rdp_per_step_cache = {}
        self._step_counts = {}
        self._rdp_values = {}

    def _rdp_per_step(self, sample_rate):
        sample_rate = float(sample_rate)
        if not math.isfinite(sample_rate) or not 0 < sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        cached = self._rdp_per_step_cache.get(sample_rate)
        if cached is None:
            cached = tuple(
                sampled_gaussian_rdp(
                    order,
                    sample_rate,
                    self.noise_multiplier,
                )
                for order in self.orders
            )
            self._rdp_per_step_cache[sample_rate] = cached
        return cached

    def step(self, device_id, sample_rate, steps=1):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        device_id = str(device_id)
        per_step = self._rdp_per_step(sample_rate)
        current = self._rdp_values.get(device_id, self._zero_rdp)
        self._rdp_values[device_id] = tuple(
            value + steps * increment
            for value, increment in zip(current, per_step)
        )
        self._step_counts[device_id] = self._step_counts.get(device_id, 0) + steps
        return self.privacy_cost(device_id)

    def privacy_cost(self, device_id):
        device_id = str(device_id)
        step_count = self._step_counts.get(device_id, 0)
        rdp_values = self._rdp_values.get(device_id, self._zero_rdp)
        return _privacy_cost(
            device_id,
            step_count,
            self.delta,
            self._log_inverse_delta,
            self.orders,
            rdp_values,
        )
