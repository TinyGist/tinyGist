import math
from dataclasses import dataclass


@dataclass
class EarlyStopState:
    best_value: float
    has_best: bool = False
    plateau_count: int = 0


@dataclass(frozen=True)
class EarlyStopDecision:
    should_stop: bool
    reason: str = ""
    metric: str = ""
    metric_value: float | None = None
    best_value: float | None = None
    plateau_count: int = 0


class EarlyStopController:
    def __init__(self, config):
        self.config = config
        self._device_records = {}

    def create_state(self, device_idx) -> EarlyStopState:
        if not self.config.enabled or not self.config.plateau.enabled:
            return EarlyStopState(best_value=0.0)

        metric = self.config.plateau.metric
        if self.config.record.scope == "device":
            device_record = self._device_records.get(device_idx, {})
            if metric in device_record:
                return EarlyStopState(
                    best_value=self._decayed_record(metric, device_record[metric]),
                    has_best=True,
                )

        return EarlyStopState(best_value=self._initial_best(metric))

    def update(self, device_idx, state: EarlyStopState, completed_epoch_count: int, metrics: dict) -> EarlyStopDecision:
        if not self.config.enabled:
            return EarlyStopDecision(should_stop=False)

        self._update_device_records(device_idx, metrics)
        if self.config.plateau.enabled:
            self._update_plateau_state(state, metrics)

        if completed_epoch_count < self.config.min_epoch:
            return EarlyStopDecision(should_stop=False)

        ceiling_decision = self._ceiling_decision(metrics)
        if ceiling_decision.should_stop:
            return ceiling_decision

        if self.config.plateau.enabled and state.plateau_count >= self.config.plateau.patience:
            metric = self.config.plateau.metric
            return EarlyStopDecision(
                should_stop=True,
                reason="plateau",
                metric=metric,
                metric_value=self._metric_value(metrics, metric),
                best_value=state.best_value,
                plateau_count=state.plateau_count,
            )

        return EarlyStopDecision(should_stop=False)

    def _update_device_records(self, device_idx, metrics: dict):
        tracked_metrics = set()
        if self.config.ceiling.enabled:
            tracked_metrics.add(self.config.ceiling.metric)
        if self.config.plateau.enabled:
            tracked_metrics.add(self.config.plateau.metric)

        if not tracked_metrics:
            return

        device_record = self._device_records.setdefault(device_idx, {})
        for metric in tracked_metrics:
            value = self._metric_value(metrics, metric)
            if metric not in device_record or self._strictly_better(metric, value, device_record[metric]):
                device_record[metric] = value

    def _update_plateau_state(self, state: EarlyStopState, metrics: dict):
        plateau = self.config.plateau
        metric = plateau.metric
        value = self._metric_value(metrics, metric)

        if not state.has_best or self._improved(metric, value, state.best_value, plateau.min_delta):
            state.best_value = value
            state.has_best = True
            state.plateau_count = 0
            return

        if self._near_best(metric, value, state.best_value, plateau.near_best_ratio):
            state.plateau_count += 1
        else:
            state.plateau_count = 0

    def _ceiling_decision(self, metrics: dict) -> EarlyStopDecision:
        ceiling = self.config.ceiling
        if not ceiling.enabled:
            return EarlyStopDecision(should_stop=False)

        metric = ceiling.metric
        value = self._metric_value(metrics, metric)
        if not self._ceiling_reached(metric, value, ceiling.value):
            return EarlyStopDecision(should_stop=False)

        return EarlyStopDecision(
            should_stop=True,
            reason="ceiling",
            metric=metric,
            metric_value=value,
            best_value=ceiling.value,
        )

    def _decayed_record(self, metric: str, value: float) -> float:
        decay = self.config.record.decay
        if decay == 0:
            return self._initial_best(metric)
        if self._higher_is_better(metric):
            return value * decay
        return value / decay

    @staticmethod
    def _metric_value(metrics: dict, metric: str) -> float:
        if metric not in metrics:
            raise KeyError(f"Missing early_stop metric [{metric}]")
        return float(metrics[metric])

    @staticmethod
    def _initial_best(metric: str) -> float:
        if EarlyStopController._higher_is_better(metric):
            return float("-inf")
        return float("inf")

    @staticmethod
    def _higher_is_better(metric: str) -> bool:
        return metric != "loss"

    @staticmethod
    def _strictly_better(metric: str, value: float, best_value: float) -> bool:
        if EarlyStopController._higher_is_better(metric):
            return value > best_value
        return value < best_value

    @staticmethod
    def _improved(metric: str, value: float, best_value: float, min_delta: float) -> bool:
        if EarlyStopController._higher_is_better(metric):
            return value > best_value + min_delta
        return value < best_value - min_delta

    @staticmethod
    def _near_best(metric: str, value: float, best_value: float, near_best_ratio: float) -> bool:
        if not EarlyStopController._is_finite(best_value):
            return False
        if EarlyStopController._higher_is_better(metric):
            return value >= best_value * near_best_ratio
        if near_best_ratio == 0:
            return True
        return value <= best_value / near_best_ratio

    @staticmethod
    def _ceiling_reached(metric: str, value: float, ceiling_value: float) -> bool:
        if EarlyStopController._higher_is_better(metric):
            return value >= ceiling_value
        return value <= ceiling_value

    @staticmethod
    def _is_finite(value: float) -> bool:
        return math.isfinite(value)
