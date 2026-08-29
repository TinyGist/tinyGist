import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from framework_runner import (
    ExperimentRun,
    QUICK_CHECK_ROUNDS,
    RUN_LABEL_LOCK_ENV_VAR,
    RUN_LABEL_ENV_VAR,
    SUPPORTED_CONFIG_PATTERNS,
    add_common_cli_arguments,
    build_experiment_runs,
    build_run_plan,
    describe_experiment_run,
    discover_config_files,
    expand_same_seed_repeats,
    experiment_run_label,
    parse_gpu_ids,
    resolve_config_folder,
    set_workspace,
)

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
RUNNER_SCRIPT = Path(__file__).resolve().with_name("framework_runner.py")

REPO_ROOT = Path(__file__).resolve().parent

# Compatibility alias for callers that imported the former parallel-only helper.
parallel_run_label = experiment_run_label


def resolve_gpu_ids(gpu_args) -> list[str]:
    requested_gpu_ids = parse_gpu_ids(gpu_args)
    if requested_gpu_ids is None:
        return visible_gpu_ids()
    if not requested_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return []
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(requested_gpu_ids)
    return requested_gpu_ids


def visible_gpu_ids() -> list[str]:
    try:
        import torch
    except Exception:
        return []

    if not torch.cuda.is_available():
        return []

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return []

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        value = cuda_visible.strip()
        if value in {"", "-1"} or value.lower() in {"none", "no", "false"}:
            return []
        if value.lower() != "all":
            gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
            if gpu_ids:
                return gpu_ids[:gpu_count]

    return [str(gpu_id) for gpu_id in range(gpu_count)]


def cuda_diagnostics() -> dict:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_import_error": repr(exc),
            "cuda_available": False,
            "cuda_device_count": 0,
        }

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }


def resolve_worker_count(max_workers, config_count: int, threads_per_worker: int, gpu_ids: list[str]) -> int:
    if config_count <= 0:
        raise ValueError("No configs to run")
    if threads_per_worker <= 0:
        raise ValueError("threads_per_worker must be greater than 0")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")

    if max_workers is not None:
        worker_count = max_workers
    elif gpu_ids:
        worker_count = len(gpu_ids)
    else:
        cpu_count = os.cpu_count() or 1
        worker_count = max(1, cpu_count // threads_per_worker)

    return min(worker_count, config_count)


def build_worker_env(slot_index: int, threads_per_worker: int, gpu_ids: list[str]) -> dict:
    env = os.environ.copy()
    for env_name in THREAD_ENV_VARS:
        env[env_name] = str(threads_per_worker)
    env["WORKSPACE_SIM_WORKER_ID"] = str(slot_index)

    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"

    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[slot_index % len(gpu_ids)]
    return env


def build_worker_command(
        run: ExperimentRun,
        replicate_experiment=False,
        pre_data_path=None,
        dataloader_workers: int = 0,
        aggregation_device: str = "cpu",
        quick_check: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--config-file",
        str(run.config_file),
        "--dataloader-workers",
        str(dataloader_workers),
        "--aggregation-device",
        aggregation_device,
    ]
    if run.uses_seed_offset or run.rewrite_config:
        command.extend([
            "--repeat-count",
            str(run.repeat_count),
        ])
    if run.uses_seed_offset:
        command.extend([
            "--repeat-index",
            str(run.repeat_index),
        ])
    if replicate_experiment:
        command.append("--replicate-experiment")
    if pre_data_path is not None:
        command.extend(["--pre-data-path", str(pre_data_path)])
    if quick_check:
        command.append("--quick-check")
    return command


def start_worker(run: ExperimentRun, slot_index: int, args, gpu_ids: list[str]) -> subprocess.Popen:
    env = build_worker_env(slot_index, args.threads_per_worker, gpu_ids)
    env.pop(RUN_LABEL_ENV_VAR, None)
    env.pop(RUN_LABEL_LOCK_ENV_VAR, None)
    quick_check = getattr(args, "quick_check", False)
    if run.uses_seed_offset or run.uses_same_seed_repeat:
        env[RUN_LABEL_ENV_VAR] = experiment_run_label(run)
        env[RUN_LABEL_LOCK_ENV_VAR] = "1"
    command = build_worker_command(
        run,
        replicate_experiment=args.replicate_experiment,
        pre_data_path=args.pre_data_path,
        dataloader_workers=args.dataloader_workers,
        aggregation_device=args.aggregation_device,
        quick_check=quick_check,
    )
    gpu_msg = env.get("CUDA_VISIBLE_DEVICES", "cpu") if gpu_ids else "cpu"
    print(f"[scheduler] start slot={slot_index} gpu={gpu_msg} {describe_experiment_run(run)}", flush=True)
    return subprocess.Popen(command, env=env)


def run_parallel(runs: list[ExperimentRun], args, gpu_ids: list[str]) -> int:
    quick_check = getattr(args, "quick_check", False)
    if args.dataloader_workers < 0:
        raise ValueError("dataloader_workers must be a non-negative integer")
    if (
            isinstance(args.poll_interval, bool)
            or not isinstance(args.poll_interval, (int, float))
            or not math.isfinite(args.poll_interval)
            or args.poll_interval <= 0
    ):
        raise ValueError("poll_interval must be a finite positive number")
    worker_count = resolve_worker_count(
        args.max_workers,
        config_count=len(runs),
        threads_per_worker=args.threads_per_worker,
        gpu_ids=gpu_ids,
    )
    quick_summary = (
        f"quick_check=true effective_rounds={QUICK_CHECK_ROUNDS}"
        if quick_check
        else "quick_check=false"
    )
    print(
        f"[scheduler] runs={len(runs)} workers={worker_count} "
        f"threads_per_worker={args.threads_per_worker} "
        f"dataloader_workers={args.dataloader_workers} "
        f"repeats_per_seed={args.repeats_per_seed} "
        f"aggregation_device={args.aggregation_device} "
        f"{quick_summary} "
        f"gpus={gpu_ids or 'none'}",
        flush=True,
    )
    print(f"[scheduler] cuda={cuda_diagnostics()}", flush=True)

    if args.dry_run:
        for run in runs:
            print(f"[scheduler] dry-run {describe_experiment_run(run)}")
        return 0

    pending = list(runs)
    running = {}
    failures = []
    completed = 0

    try:
        while pending or running:
            while pending and len(running) < worker_count:
                used_slots = {job["slot_index"] for job in running.values()}
                slot_index = next(slot for slot in range(worker_count) if slot not in used_slots)
                run = pending.pop(0)
                process = start_worker(run, slot_index, args, gpu_ids)
                running[process] = {
                    "run": run,
                    "slot_index": slot_index,
                    "started_at": time.monotonic(),
                }

            finished_processes = []
            for process, job in running.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                finished_processes.append(process)
                completed += 1
                elapsed = time.monotonic() - job["started_at"]
                run = job["run"]
                if return_code == 0:
                    print(
                        f"[scheduler] done {completed}/{len(runs)} "
                        f"{describe_experiment_run(run)} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[scheduler] failed {completed}/{len(runs)} "
                        f"{describe_experiment_run(run)} return_code={return_code} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    failures.append((run, return_code))

            for process in finished_processes:
                running.pop(process)

            if running and not finished_processes:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("[scheduler] interrupted; terminating running experiments", flush=True)
        for process in running:
            process.terminate()
        for process in running:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise

    if failures:
        print("[scheduler] failed runs:", flush=True)
        for run, return_code in failures:
            print(f"  return_code={return_code} {describe_experiment_run(run)}", flush=True)
        return 1

    print(f"[scheduler] all {len(runs)} runs completed", flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run federated learning simulations in parallel from YAML config files."
    )
    add_common_cli_arguments(
        parser,
        aggregation_device_default="cpu",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent experiments. Defaults to visible GPU count, otherwise CPU count / threads-per-worker.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="CPU thread limit exported to each worker process. Default: 2.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between scheduler checks. Default: 2.0.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    workspace_path = set_workspace(args.workspace)
    if workspace_path is not None:
        print(f"Workspace: {workspace_path}", flush=True)

    config_folder = resolve_config_folder(args.config_folder, workspace_path=workspace_path)
    runs = build_run_plan(
        config_file=args.config_file,
        config_folder=config_folder,
        repeat_count_override=args.seed_count,
        repeats_per_seed=args.repeats_per_seed,
    )
    gpu_ids = resolve_gpu_ids(args.gpus)
    return run_parallel(runs, args, gpu_ids)


if __name__ == "__main__":
    raise SystemExit(main())
