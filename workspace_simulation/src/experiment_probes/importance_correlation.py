import csv
from contextlib import contextmanager
import math
from pathlib import Path
import random

import numpy as np
import torch

from .config import ImportanceCorrelationProbeConfig


OUTPUT_FILE = "importance_correlation.csv"


class ImportanceCorrelationProbe:
    BASE_FIELDNAMES = (
        "round",
        "stage",
        "device",
        "parameter_count",
        "comparison_baseline",
        "reference_score",
    )

    def __init__(self, config: ImportanceCorrelationProbeConfig, log_dir):
        self.config = config
        self.fieldnames = (*self.BASE_FIELDNAMES, *self._measurement_fieldnames())
        self.output_path = Path(log_dir) / OUTPUT_FILE
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            with self.output_path.open("w", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=self.fieldnames).writeheader()

    def record(self, round_idx, stage, gradient_buffer, model_idx_list):
        if not self.config.should_evaluate(round_idx, stage):
            return
        rows = []
        for device_idx in model_idx_list:
            baseline = self._score_vector(
                gradient_buffer,
                device_idx,
                self.config.comparison_baseline.internal_name,
            )
            for reference_metric in self.config.reference_scores:
                reference = self._score_vector(
                    gradient_buffer,
                    device_idx,
                    reference_metric.internal_name,
                )
                if baseline.size != reference.size:
                    raise ValueError(
                        "Importance correlation score-vector lengths differ for "
                        f"device [{device_idx}]: {baseline.size} != {reference.size}"
                    )
                rows.append(self._measurement_row(
                    round_idx,
                    stage,
                    device_idx,
                    baseline,
                    reference,
                    reference_metric.name,
                ))
        if rows:
            with self.output_path.open("a", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=self.fieldnames)
                writer.writerows(rows)
                output.flush()

    def close(self):
        return None

    def _score_vector(self, gradient_buffer, device_idx, score_method):
        return (
            gradient_buffer.get_parameter_score_vector(
                device_idx,
                parameter_score_method=score_method,
                parameter_scope=self.config.parameter_scope,
                bn_mode=self.config.bn_mode,
                target_device="cpu",
            )
            .detach()
            .to(dtype=torch.float64)
            .numpy()
        )

    def _measurement_fieldnames(self):
        fieldnames = []
        if "spearman" in self.config.measurements:
            fieldnames.append("spearman")
        for measurement in ("top_k_overlap", "top_k_jaccard"):
            if measurement not in self.config.measurements:
                continue
            fieldnames.extend(
                _top_k_column_name(measurement, top_k)
                for top_k in self.config.top_k
            )
        return tuple(fieldnames)

    def _measurement_row(
            self,
            round_idx,
            stage,
            device_idx,
            baseline,
            reference,
            reference_name,
    ):
        row = {
            "round": round_idx,
            "stage": stage,
            "device": device_idx,
            "parameter_count": int(baseline.size),
            "comparison_baseline": self.config.comparison_baseline.name,
            "reference_score": reference_name,
        }
        if "spearman" in self.config.measurements:
            row["spearman"] = _spearman(baseline, reference)
        if (
                "top_k_overlap" in self.config.measurements
                or "top_k_jaccard" in self.config.measurements
        ):
            for top_k in self.config.top_k:
                overlap, jaccard = _top_k_measurements(
                    baseline,
                    reference,
                    top_k,
                )
                if "top_k_overlap" in self.config.measurements:
                    row[_top_k_column_name("top_k_overlap", top_k)] = overlap
                if "top_k_jaccard" in self.config.measurements:
                    row[_top_k_column_name("top_k_jaccard", top_k)] = jaccard
        return row


def _top_k_column_name(measurement, fraction):
    return f"{measurement}_{fraction:g}"


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _spearman(left, right):
    if left.size < 2 or right.size != left.size:
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _top_k_measurements(left, right, fraction):
    if left.size == 0 or right.size != left.size:
        return float("nan"), float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan"), float("nan")
    count = min(left.size, max(1, math.ceil(fraction * left.size)))
    coordinate = np.arange(left.size)
    left_top = set(np.lexsort((coordinate, -left))[:count].tolist())
    right_top = set(np.lexsort((coordinate, -right))[:count].tolist())
    intersection = len(left_top & right_top)
    union = len(left_top | right_top)
    return intersection / count, intersection / union


@contextmanager
def preserve_random_state(dataloaders=()):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generator_states = []
    seen_generators = set()
    for dataloader in dataloaders:
        candidates = (
            getattr(dataloader, "generator", None),
            getattr(getattr(dataloader, "sampler", None), "generator", None),
            getattr(
                getattr(getattr(dataloader, "batch_sampler", None), "sampler", None),
                "generator",
                None,
            ),
        )
        for generator in candidates:
            if generator is None or id(generator) in seen_generators:
                continue
            seen_generators.add(id(generator))
            generator_states.append((generator, generator.get_state()))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for generator, state in generator_states:
            generator.set_state(state)
