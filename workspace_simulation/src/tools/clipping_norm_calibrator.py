"""Collect whole-model norm samples and propose clipping thresholds."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import math
import os
from pathlib import Path
import statistics
import time

import torch
import yaml

from src.data_getter import DATASETS
from src.differential_privacy.calibration import (
    model_update_l2_norm,
    per_sample_gradient_l2_norms,
    snapshot_trainable_parameters,
    trainable_parameter_count,
)
from src.differential_privacy.dp_sgd import freeze_batch_norm_statistics
from src.models import Criteria, NETWORKS
from src.models.model_registry import isolated_model_initialization_rng
from src.sim_tools.simulation_config import load_simulation_config
from src.tools.clipping_norm_selector import summarize_clipping_norms


DEFAULT_CALIBRATION_ROUNDS = 1
DEFAULT_SAMPLES_PER_BATCH = 8
DEFAULT_TIME_LIMIT_SECONDS = 1800
RAW_FIELDS = (
    "mode",
    "device_id",
    "calibration_round",
    "epoch",
    "batch",
    "sample_index",
    "l2_norm",
)


class CalibrationTimeLimit(RuntimeError):
    pass


def _positive_int(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _resolve_device(value):
    canonical = str(value).strip().lower()
    if canonical == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if canonical == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if canonical not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(canonical)


def _directory_string(path):
    return str(Path(path).resolve()) + os.sep


def _default_output_dir(config_file):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("calibration") / f"{timestamp}_{Path(config_file).stem}"


def _build_training_loaders(config, output_dir, pre_data_path, num_workers):
    dataset = config.dataset
    training = config.training
    getter_type = DATASETS[dataset.dataset_name]
    getter = getter_type(
        num_devices=training.device_count,
        label_per_device=dataset.labels_per_device,
        data_per_device=dataset.training_data_per_device,
        prefix_name=training.device_indicator_prefix,
    )
    if pre_data_path is not None:
        getter.get_data_from_store(
            _directory_string(pre_data_path),
            dataset.train_batch_size,
            dataset.test_batch_size,
            dataset.valid_batch_size,
            num_workers=num_workers,
        )
    else:
        data_dir = output_dir / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        getter.get_data_from_new(
            dataset.label_allocating_method,
            dataset.label_allocating_loop_step,
            dataset.data_allocating_method,
            dataset.data_allocating_alpha,
            dataset.test_data_size_total,
            dataset.valid_data_size_total,
            _directory_string(data_dir),
            dataset.train_batch_size,
            dataset.test_batch_size,
            dataset.valid_batch_size,
            num_workers=num_workers,
        )
    if (
            config.experiment.task == "Classification"
            and getter.get_total_classes() != config.model.output_class_number
    ):
        raise ValueError(
            "Configured model class count does not match the loaded dataset"
        )
    return getter.get_training_dataloader_dict()


def _build_model(config, device_index, runtime_device):
    model_config = config.model
    model_type = NETWORKS[model_config.model_name]
    with isolated_model_initialization_rng():
        model = model_type(
            model_config.input_size,
            device_num=device_index,
            random_seed=model_config.torch_random_seed,
            num_class=model_config.output_class_number,
        )
    return model.to(runtime_device)


def _build_loss(config, runtime_device):
    name = config.training.loss_function_name
    loss_type = torch.nn.CrossEntropyLoss if name == "ce_loss" else Criteria[name]
    loss = loss_type()
    return loss.to(runtime_device) if isinstance(loss, torch.nn.Module) else loss


def _build_optimizer(config, model):
    training = config.training
    optimizer_type = (
        torch.optim.SGD
        if training.optimizer_name == "sgd"
        else torch.optim.Adam
    )
    return optimizer_type(
        model.parameters(),
        lr=training.initial_lr,
        weight_decay=training.weight_decay,
    )


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise CalibrationTimeLimit("calibration time limit reached")


def _ordinary_training_step(model, optimizer, loss_function, data, labels):
    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    loss = loss_function(output, labels)
    if isinstance(loss, torch.Tensor) and loss.ndim:
        loss = loss.mean()
    loss.backward()
    optimizer.step()


def _collect_norm_samples(
        config,
        loaders,
        runtime_device,
        *,
        calibration_rounds,
        samples_per_batch,
        max_devices,
        deadline,
        rows,
):
    mode = config.differential_privacy.mode
    training = config.training
    selected = list(loaders.items())
    if max_devices is not None:
        selected = selected[:max_devices]
    if not selected:
        raise ValueError("No training devices are available for calibration")

    parameter_count = None
    fallback_devices = set()
    for device_index, (device_id, loader) in enumerate(selected):
        _check_deadline(deadline)
        model = _build_model(config, device_index, runtime_device)
        loss_function = _build_loss(config, runtime_device)
        optimizer = _build_optimizer(config, model)
        current_parameter_count = trainable_parameter_count(model)
        if parameter_count is None:
            parameter_count = current_parameter_count
        elif parameter_count != current_parameter_count:
            raise ValueError("Calibrated models have inconsistent parameter counts")

        for calibration_round in range(1, calibration_rounds + 1):
            _check_deadline(deadline)
            if training.optimizer_name == "adam_round":
                optimizer = _build_optimizer(config, model)
            baseline = (
                snapshot_trainable_parameters(model)
                if mode == "model_update"
                else None
            )
            model.train()
            if mode == "local_dp_sgd":
                freeze_batch_norm_statistics(model)

            for epoch in range(1, training.epoch_per_round + 1):
                for batch_index, batch in enumerate(loader, start=1):
                    if batch_index > training.max_batch_per_epoch:
                        break
                    _check_deadline(deadline)
                    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                        raise ValueError(
                            "Clipping calibration currently requires classification "
                            "batches shaped as (data, labels)"
                        )
                    data, labels = batch
                    data = data.to(runtime_device, non_blocking=True)
                    labels = labels.to(runtime_device, non_blocking=True)

                    if mode == "local_dp_sgd":
                        measured = min(samples_per_batch, int(labels.shape[0]))
                        norms, used_fallback, reason = per_sample_gradient_l2_norms(
                            model,
                            loss_function,
                            data[:measured],
                            labels[:measured],
                            force_fallback=device_id in fallback_devices,
                        )
                        if used_fallback and device_id not in fallback_devices:
                            fallback_devices.add(device_id)
                            print(
                                f"warning: {device_id} uses per-sample fallback: "
                                f"{reason or 'forced after first failure'}",
                                flush=True,
                            )
                        for sample_index, norm in enumerate(
                                norms.detach().cpu().tolist()
                        ):
                            rows.append({
                                "mode": mode,
                                "device_id": device_id,
                                "calibration_round": calibration_round,
                                "epoch": epoch,
                                "batch": batch_index,
                                "sample_index": sample_index,
                                "l2_norm": float(norm),
                            })
                    _ordinary_training_step(
                        model,
                        optimizer,
                        loss_function,
                        data,
                        labels,
                    )

            if mode == "model_update":
                rows.append({
                    "mode": mode,
                    "device_id": device_id,
                    "calibration_round": calibration_round,
                    "epoch": training.epoch_per_round,
                    "batch": training.max_batch_per_epoch,
                    "sample_index": "",
                    "l2_norm": float(
                        model_update_l2_norm(model, baseline).detach().cpu()
                    ),
                })
            print(
                f"calibration progress: device={device_id}, "
                f"round={calibration_round}/{calibration_rounds}, "
                f"norm_samples={len(rows)}",
                flush=True,
            )
        del optimizer, loss_function, model
        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, parameter_count, len(selected), tuple(sorted(fallback_devices))


def _percentile(sorted_values, quantile):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _distribution(values):
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": len(sorted_values),
        "minimum": sorted_values[0],
        "mean": statistics.fmean(sorted_values),
        "maximum": sorted_values[-1],
        "p50": _percentile(sorted_values, 0.50),
        "p75": _percentile(sorted_values, 0.75),
        "p80": _percentile(sorted_values, 0.80),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
    }


def _candidate_dict(candidate):
    return {
        "source": candidate.label,
        "quantile": candidate.quantile,
        "clipping_norm": candidate.clipping_norm,
        "observed_clipping_fraction": candidate.observed_clipping_fraction,
        "mean_clipped_norm": candidate.mean_clipped_norm,
        "mean_retained_ratio": candidate.mean_retained_ratio,
        "mean_clipping_residual": candidate.mean_clipping_residual,
        "mechanism_noise_std": candidate.mechanism_noise_std,
        "post_average_noise_std": candidate.post_average_noise_std,
        "expected_noise_l2": candidate.expected_noise_l2,
    }


def _write_outputs(
        output_dir,
        config_file,
        config,
        rows,
        parameter_count,
        device_count,
        fallback_devices,
        *,
        calibration_rounds,
        samples_per_batch,
        runtime_device,
        partial,
        elapsed_seconds,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "clipping_norm_samples.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    norms = [row["l2_norm"] for row in rows]
    expected_batch_size = (
        min(
            config.dataset.train_batch_size,
            config.dataset.training_data_per_device,
        )
        if config.differential_privacy.mode == "local_dp_sgd"
        else None
    )
    summary = summarize_clipping_norms(
        norms,
        config.differential_privacy,
        expected_batch_size=expected_batch_size,
        parameter_count=parameter_count,
    )
    by_device = defaultdict(list)
    for row in rows:
        by_device[row["device_id"]].append(row["l2_norm"])
    summary_data = {
        "config_file": str(Path(config_file).resolve()),
        "mode": config.differential_privacy.mode,
        "status": "partial" if partial else "complete",
        "privacy_warning": (
            "This is non-private calibration. Reusing protected training data "
            "without accounting for calibration does not provide an end-to-end "
            "DP guarantee."
        ),
        "runtime_device": str(runtime_device),
        "elapsed_seconds": elapsed_seconds,
        "calibration_rounds": calibration_rounds,
        "samples_per_batch": (
            samples_per_batch
            if config.differential_privacy.mode == "local_dp_sgd"
            else None
        ),
        "devices_calibrated": device_count,
        "fallback_devices": list(fallback_devices),
        "trainable_parameter_count": parameter_count,
        "current_clipping_norm": summary.current_clipping_norm,
        "current_observed_clipping_fraction": summary.current_clipping_fraction,
        "norm_distribution": _distribution(norms),
        "per_device_norm_distribution": {
            device_id: _distribution(device_norms)
            for device_id, device_norms in sorted(by_device.items())
        },
        "recommended_starting_quantile": summary.recommended.label,
        "recommended_starting_clipping_norm": summary.recommended.clipping_norm,
        "candidates": [
            _candidate_dict(candidate)
            for candidate in summary.candidates
        ],
        "raw_samples_file": raw_path.name,
    }
    summary_path = output_dir / "clipping_norm_summary.yml"
    with summary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(summary_data, file, sort_keys=False)
    with (output_dir / "used_config.yml").open("w", encoding="utf-8") as file:
        config.parser.write(file)
    return raw_path, summary_path, summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run short non-private local training with the configured model and "
            "dataset, then report whole-model clipping_norm candidates."
        )
    )
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument(
        "--calibration-rounds",
        type=_positive_int,
        default=DEFAULT_CALIBRATION_ROUNDS,
    )
    parser.add_argument(
        "--samples-per-batch",
        type=_positive_int,
        default=DEFAULT_SAMPLES_PER_BATCH,
        help="local_dp_sgd only; samples measured from each training batch",
    )
    parser.add_argument("--max-devices", type=_positive_int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--pre-data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=_positive_int,
        default=DEFAULT_TIME_LIMIT_SECONDS,
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    started = time.monotonic()
    rows = []
    partial = False
    try:
        config = load_simulation_config(args.config_file)
        if not config.differential_privacy.enabled:
            raise ValueError("Differential privacy is disabled in the config")
        if config.experiment.task != "Classification":
            raise ValueError(
                "Clipping calibration currently supports classification only"
            )
        runtime_device = _resolve_device(args.device)
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else _default_output_dir(args.config_file)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "warning: this tool performs non-private calibration; use public, "
            "independent, or previously approved pilot data for a complete DP claim",
            flush=True,
        )
        loaders = _build_training_loaders(
            config,
            output_dir,
            args.pre_data_path,
            args.dataloader_workers,
        )
        deadline = started + args.time_limit_seconds
        try:
            (
                rows,
                parameter_count,
                device_count,
                fallback_devices,
            ) = _collect_norm_samples(
                config,
                loaders,
                runtime_device,
                rows=rows,
                calibration_rounds=args.calibration_rounds,
                samples_per_batch=args.samples_per_batch,
                max_devices=args.max_devices,
                deadline=deadline,
            )
        except (CalibrationTimeLimit, KeyboardInterrupt) as exc:
            partial = True
            print(f"warning: {exc}; writing partial calibration output", flush=True)
            if not rows:
                raise ValueError(
                    "Calibration stopped before any norm samples were collected"
                ) from exc
            parameter_count = None
            device_count = len({row["device_id"] for row in rows})
            fallback_devices = ()
        elapsed = time.monotonic() - started
        raw_path, summary_path, summary = _write_outputs(
            output_dir,
            args.config_file,
            config,
            rows,
            parameter_count,
            device_count,
            fallback_devices,
            calibration_rounds=args.calibration_rounds,
            samples_per_batch=args.samples_per_batch,
            runtime_device=runtime_device,
            partial=partial,
            elapsed_seconds=elapsed,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"raw_samples: {raw_path.resolve()}")
    print(f"summary: {summary_path.resolve()}")
    print(f"status: {'partial' if partial else 'complete'}")
    print(f"sample_count: {summary.sample_count}")
    print(f"current_clipping_fraction: {summary.current_clipping_fraction:.10g}")
    print(f"recommended_starting_quantile: {summary.recommended.label}")
    print(
        "recommended_starting_clipping_norm: "
        f"{summary.recommended.clipping_norm:.10g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
