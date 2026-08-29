"""Privacy-accounting output independent from other experiment recorders."""

import csv
from pathlib import Path


def _alpha_column(alpha):
    return f"rdp_alpha_{format(float(alpha), 'g').replace('.', '_')}"


class PrivacyRecorder:
    def __init__(self, output_path, orders):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.orders = tuple(orders)
        self._alpha_columns = tuple(_alpha_column(order) for order in self.orders)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._fieldnames = (
            "round",
            "device_id",
            "mode",
            "adjacency",
            "release_count",
            "steps_this_round",
            "sample_rate",
            "expected_batch_size",
            "dataset_size",
            "clipping_norm",
            "l2_sensitivity",
            "noise_multiplier",
            "noise_std",
            "delta",
            "epsilon",
            "optimal_alpha",
            *self._alpha_columns,
        )
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._closed = False

    def record(
            self,
            global_round,
            config,
            privacy_cost,
            *,
            steps_this_round=None,
            sample_rate=None,
            expected_batch_size=None,
            dataset_size=None,
    ):
        if self._closed:
            raise RuntimeError("PrivacyRecorder is already closed")
        if len(privacy_cost.rdp_values) != len(self.orders):
            raise ValueError(
                "Privacy cost RDP values do not match recorder orders"
            )
        if privacy_cost.optimal_alpha not in self.orders:
            raise ValueError(
                "Privacy cost optimal alpha is absent from recorder orders"
            )
        row = {
            "round": int(global_round),
            "device_id": privacy_cost.device_id,
            "mode": config.mode,
            "adjacency": config.adjacency,
            "release_count": privacy_cost.release_count,
            "steps_this_round": steps_this_round,
            "sample_rate": sample_rate,
            "expected_batch_size": expected_batch_size,
            "dataset_size": dataset_size,
            "clipping_norm": config.clipping_norm,
            "l2_sensitivity": config.l2_sensitivity,
            "noise_multiplier": config.noise_multiplier,
            "noise_std": config.noise_std,
            "delta": privacy_cost.delta,
            "epsilon": privacy_cost.epsilon,
            "optimal_alpha": privacy_cost.optimal_alpha,
        }
        row.update({
            column: value
            for column, value in zip(self._alpha_columns, privacy_cost.rdp_values)
        })
        self._writer.writerow(row)

    def flush(self):
        if not self._closed:
            self._file.flush()

    def close(self):
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True
