import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile

import yaml

from src.sim_tools.runtime_env import configure_runtime_cache_dirs


SUPPORTED_CONFIG_PATTERNS = ("*.yml", "*.yaml")
DEFAULT_CONFIG_FOLDER = "src/config_folder/"
WORKSPACE_CONFIG_FOLDER = "config"
RUN_LABEL_ENV_VAR = "WORKSPACE_SIM_RUN_LABEL"
RUN_LABEL_LOCK_ENV_VAR = "WORKSPACE_SIM_RUN_LABEL_LOCKED"
SOURCE_CONFIG_ENV_VAR = "WORKSPACE_SIM_SOURCE_CONFIG"
QUICK_CHECK_ROUNDS = 2

configure_runtime_cache_dirs()


@dataclass(frozen=True)
class ExperimentRun:
    config_file: Path
    repeat_count: int
    repeat_index: int | None
    base_seed: int
    run_seed: int
    rewrite_config: bool = False
    repeats_per_seed: int = 1
    same_seed_repeat_index: int | None = None

    @property
    def uses_seed_offset(self) -> bool:
        return self.repeat_index is not None

    @property
    def uses_same_seed_repeat(self) -> bool:
        return self.same_seed_repeat_index is not None


def set_workspace(workspace=None):
    if workspace is None:
        return None

    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace_path}")
    if not workspace_path.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace_path}")

    os.chdir(workspace_path)
    return workspace_path


def resolve_config_folder(config_folder=None, workspace_path=None) -> Path:
    if config_folder is not None:
        return Path(config_folder)
    if workspace_path is not None:
        return Path(WORKSPACE_CONFIG_FOLDER)
    return Path(DEFAULT_CONFIG_FOLDER)


def discover_config_files(config_file=None, config_folder=None) -> list[Path]:
    if config_file is not None:
        used_file = Path(config_file)
        if not used_file.exists():
            raise FileNotFoundError(f"Config file does not exist: {used_file}")
        if not used_file.is_file():
            raise FileNotFoundError(f"Config path is not a file: {used_file}")
        return [used_file]

    used_folder = Path(config_folder)
    if not used_folder.exists():
        raise FileNotFoundError(f"Config folder does not exist: {used_folder}")
    if not used_folder.is_dir():
        raise NotADirectoryError(f"Config folder is not a directory: {used_folder}")

    config_files = []
    for pattern in SUPPORTED_CONFIG_PATTERNS:
        config_files.extend(used_folder.rglob(pattern))
    config_files = sorted(config_files)
    if not config_files:
        raise FileNotFoundError(
            f"No config files found in {used_folder} matching {SUPPORTED_CONFIG_PATTERNS}"
        )
    return config_files


def parse_gpu_ids(gpu_args) -> list[str] | None:
    if gpu_args is None:
        return None

    raw_value = ",".join(str(item) for item in gpu_args).replace("，", ",")
    raw_value = raw_value.strip()
    if not raw_value or raw_value.lower() in {"all", "default"}:
        return None
    if raw_value.lower() in {"none", "no", "false", "cpu", "-1"}:
        return []

    gpu_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not gpu_ids:
        return None
    invalid_ids = [gpu_id for gpu_id in gpu_ids if not gpu_id.isdigit()]
    if invalid_ids:
        raise ValueError(f"Invalid GPU id(s): {invalid_ids}. Use values like --gpus 0 or --gpus 0,1.")
    return gpu_ids


def apply_gpu_selection(gpu_ids: list[str] | None):
    if gpu_ids is None:
        print("GPUs: all visible GPUs", flush=True)
        return
    if not gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPUs: disabled, running on CPU", flush=True)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)


def read_yaml_config(config_file: Path) -> dict:
    path = Path(config_file)
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _mapping(value, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _positive_int(value, field_name: str) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if int_value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int_value


def _non_negative_int(value, field_name: str) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if int_value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return int_value


def _repeat_count_from_data(data: dict, field_name: str) -> int:
    experiment = data.get("experiment", {}) or {}
    experiment = _mapping(experiment, "experiment")
    return _positive_int(experiment.get("repeat_count", 1), field_name)


def _base_seed_from_data(data: dict, config_file: Path) -> int:
    experiment = _mapping(data.get("experiment", {}) or {}, "experiment")
    if "seed" not in experiment:
        raise KeyError(f"Missing required config key experiment.seed in {config_file}")
    return int(experiment["seed"])


def build_experiment_runs(
        config_files: list[Path],
        repeat_count_override: int | None = None,
        repeat_index: int | None = None,
) -> list[ExperimentRun]:
    if repeat_count_override is not None:
        repeat_count_override = _positive_int(
            repeat_count_override,
            "--seed-count/--repeat-count",
        )
    if repeat_index is not None:
        repeat_index = _non_negative_int(repeat_index, "--repeat-index")

    runs = []
    for config_file in config_files:
        config_file = Path(config_file)
        data = read_yaml_config(config_file)
        source_repeat_count = _repeat_count_from_data(data, f"{config_file}: experiment.repeat_count")
        repeat_count = (
            repeat_count_override
            if repeat_count_override is not None
            else source_repeat_count
        )
        rewrite_config = repeat_count_override is not None and repeat_count_override != source_repeat_count
        base_seed = _base_seed_from_data(data, config_file)

        if repeat_index is not None:
            if repeat_index >= repeat_count:
                raise ValueError(
                    f"--repeat-index {repeat_index} is outside [0, {repeat_count}) for {config_file}"
                )
            runs.append(ExperimentRun(
                config_file=config_file,
                repeat_count=repeat_count,
                repeat_index=repeat_index,
                base_seed=base_seed,
                run_seed=base_seed + repeat_index,
                rewrite_config=rewrite_config,
            ))
            continue

        if repeat_count == 1:
            runs.append(ExperimentRun(
                config_file=config_file,
                repeat_count=repeat_count,
                repeat_index=None,
                base_seed=base_seed,
                run_seed=base_seed,
                rewrite_config=rewrite_config,
            ))
            continue

        for used_repeat_index in range(0, repeat_count):
            runs.append(ExperimentRun(
                config_file=config_file,
                repeat_count=repeat_count,
                repeat_index=used_repeat_index,
                base_seed=base_seed,
                run_seed=base_seed + used_repeat_index,
                rewrite_config=rewrite_config,
            ))

    return runs


def expand_same_seed_repeats(
        runs: list[ExperimentRun],
        repeats_per_seed: int,
) -> list[ExperimentRun]:
    repeats_per_seed = _positive_int(repeats_per_seed, "--repeats-per-seed")
    if repeats_per_seed == 1:
        return runs
    return [
        replace(
            run,
            repeats_per_seed=repeats_per_seed,
            same_seed_repeat_index=repeat_index,
        )
        for run in runs
        for repeat_index in range(repeats_per_seed)
    ]


def build_run_plan(
        config_file=None,
        config_folder=None,
        repeat_count_override: int | None = None,
        repeat_index: int | None = None,
        repeats_per_seed: int = 1,
) -> list[ExperimentRun]:
    config_files = discover_config_files(
        config_file=config_file,
        config_folder=config_folder,
    )
    seed_runs = build_experiment_runs(
        config_files,
        repeat_count_override=repeat_count_override,
        repeat_index=repeat_index,
    )
    return expand_same_seed_repeats(seed_runs, repeats_per_seed)


def experiment_run_label(run: ExperimentRun) -> str:
    seed_index = run.repeat_index if run.repeat_index is not None else 0
    same_seed_repeat_index = (
        run.same_seed_repeat_index
        if run.same_seed_repeat_index is not None
        else 0
    )
    return (
        f"seed_index{seed_index:03d}_repeat{same_seed_repeat_index:03d}_"
        f"seed{run.run_seed}"
    )


def describe_experiment_run(run: ExperimentRun) -> str:
    if not run.uses_seed_offset:
        description = (
            f"config={run.config_file} "
            f"model_initialization_seed={run.run_seed}"
        )
    else:
        description = (
            f"config={run.config_file} repeat={run.repeat_index}/{run.repeat_count} "
            f"base_model_initialization_seed={run.base_seed} "
            f"model_initialization_seed={run.run_seed}"
        )
    if run.uses_same_seed_repeat:
        description += (
            f" same_seed_repeat={run.same_seed_repeat_index}/"
            f"{run.repeats_per_seed}"
        )
    return description


@contextmanager
def prepared_config_file(run: ExperimentRun, quick_check: bool = False):
    if not run.uses_seed_offset and not run.rewrite_config and not quick_check:
        yield run.config_file
        return

    data = read_yaml_config(run.config_file)
    if run.uses_seed_offset or run.rewrite_config:
        experiment = data.setdefault("experiment", {})
        experiment = _mapping(experiment, "experiment")
        experiment["seed"] = run.run_seed
        experiment["repeat_count"] = run.repeat_count
        if run.uses_seed_offset:
            experiment["repeat_index"] = run.repeat_index
            experiment["base_seed"] = run.base_seed
        else:
            experiment.pop("repeat_index", None)
            experiment.pop("base_seed", None)

    if quick_check:
        federation = _mapping(data.get("federation"), "federation")
        if "rounds" not in federation:
            raise KeyError("Missing required config key federation.rounds")
        source_rounds = federation["rounds"]
        if (
                isinstance(source_rounds, bool)
                or not isinstance(source_rounds, int)
                or source_rounds <= 0
        ):
            raise ValueError("federation.rounds must be a positive integer.")
        federation["rounds"] = QUICK_CHECK_ROUNDS

    temp_prefix = "workspace_sim_quick_check_" if quick_check else "workspace_sim_repeat_"
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        quick_suffix = f"_quick_check{QUICK_CHECK_ROUNDS}r" if quick_check else ""
        if run.repeat_index is None:
            temp_name = (
                f"{run.config_file.stem}_repeat_count{run.repeat_count}_"
                f"seed{run.run_seed}{quick_suffix}.yml"
            )
        else:
            temp_name = (
                f"{run.config_file.stem}_repeat{run.repeat_index:03d}_"
                f"seed{run.run_seed}{quick_suffix}.yml"
            )
        temp_config = Path(temp_dir) / temp_name
        with temp_config.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=False)
        yield temp_config


@contextmanager
def repeat_run_label(run: ExperimentRun):
    if (
            os.environ.get(RUN_LABEL_LOCK_ENV_VAR) == "1"
            and RUN_LABEL_ENV_VAR in os.environ
    ):
        yield
        return
    if not (run.uses_seed_offset or run.uses_same_seed_repeat):
        yield
        return

    old_value = os.environ.get(RUN_LABEL_ENV_VAR)
    os.environ[RUN_LABEL_ENV_VAR] = experiment_run_label(run)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(RUN_LABEL_ENV_VAR, None)
        else:
            os.environ[RUN_LABEL_ENV_VAR] = old_value


@contextmanager
def source_config_path(run: ExperimentRun):
    old_value = os.environ.get(SOURCE_CONFIG_ENV_VAR)
    os.environ[SOURCE_CONFIG_ENV_VAR] = str(run.config_file)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(SOURCE_CONFIG_ENV_VAR, None)
        else:
            os.environ[SOURCE_CONFIG_ENV_VAR] = old_value


def run_experiment_run(
        run: ExperimentRun,
        replicate_experiment: bool = False,
        pre_data_path=None,
        dataloader_workers: int = 0,
        aggregation_device: str = "auto",
        quick_check: bool = False,
):
    from src.sim_tools.simulation_manager_tool import FederatedLearningSim

    quick_description = (
        f" quick_check_rounds={QUICK_CHECK_ROUNDS}" if quick_check else ""
    )
    print(f"Run {describe_experiment_run(run)}{quick_description}", flush=True)
    with prepared_config_file(run, quick_check=quick_check) as config_file:
        with source_config_path(run):
            with repeat_run_label(run):
                sim = FederatedLearningSim(
                    config_file,
                    replicate_experiment=replicate_experiment,
                    pre_data_path=pre_data_path,
                    dataloader_workers=dataloader_workers,
                    aggregation_device=aggregation_device,
                )
                sim.compile_all_settings()
                sim.run_simulation_multi()


def run_config_file(
        config_file: Path,
        replicate_experiment: bool = False,
        pre_data_path=None,
        repeat_count_override: int | None = None,
        repeat_index: int | None = None,
        dataloader_workers: int = 0,
        aggregation_device: str = "auto",
        repeats_per_seed: int = 1,
        quick_check: bool = False,
):
    runs = build_run_plan(
        config_file=Path(config_file),
        repeat_count_override=repeat_count_override,
        repeat_index=repeat_index,
        repeats_per_seed=repeats_per_seed,
    )
    return run_serial(
        runs,
        replicate_experiment=replicate_experiment,
        pre_data_path=pre_data_path,
        dataloader_workers=dataloader_workers,
        aggregation_device=aggregation_device,
        quick_check=quick_check,
    )


def run_config_folder(
        config_folder: Path,
        replicate_experiment: bool = False,
        pre_data_path=None,
        repeat_count_override: int | None = None,
        repeat_index: int | None = None,
        dataloader_workers: int = 0,
        aggregation_device: str = "auto",
        repeats_per_seed: int = 1,
        quick_check: bool = False,
):
    runs = build_run_plan(
        config_folder=Path(config_folder),
        repeat_count_override=repeat_count_override,
        repeat_index=repeat_index,
        repeats_per_seed=repeats_per_seed,
    )
    return run_serial(
        runs,
        replicate_experiment=replicate_experiment,
        pre_data_path=pre_data_path,
        dataloader_workers=dataloader_workers,
        aggregation_device=aggregation_device,
        quick_check=quick_check,
    )


def run_serial(
        runs: list[ExperimentRun],
        replicate_experiment: bool = False,
        pre_data_path=None,
        dataloader_workers: int = 0,
        aggregation_device: str = "auto",
        dry_run: bool = False,
        quick_check: bool = False,
) -> int:
    if not runs:
        raise ValueError("No experiment runs to execute")
    dataloader_workers = _non_negative_int(
        dataloader_workers,
        "--dataloader-workers",
    )
    repeats_per_seed = runs[0].repeats_per_seed
    quick_summary = (
        f"quick_check=true effective_rounds={QUICK_CHECK_ROUNDS}"
        if quick_check
        else "quick_check=false"
    )
    print(
        f"[runner] runs={len(runs)} dataloader_workers={dataloader_workers} "
        f"repeats_per_seed={repeats_per_seed} "
        f"aggregation_device={aggregation_device} "
        f"{quick_summary}",
        flush=True,
    )

    if dry_run:
        for run in runs:
            print(f"[runner] dry-run {describe_experiment_run(run)}", flush=True)
        return 0

    for run in runs:
        run_experiment_run(
            run,
            replicate_experiment=replicate_experiment,
            pre_data_path=pre_data_path,
            dataloader_workers=dataloader_workers,
            aggregation_device=aggregation_device,
            quick_check=quick_check,
        )
    print(f"[runner] all {len(runs)} runs completed", flush=True)
    return 0


def add_common_cli_arguments(
        parser: argparse.ArgumentParser,
        aggregation_device_default: str,
        include_repeat_index: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config-file",
        default=None,
        help="Run one .yml or .yaml config file. Overrides --config-folder.",
    )
    parser.add_argument(
        "--config-folder",
        default=None,
        help=(
            "Folder containing .yml or .yaml config files. "
            "Defaults to src/config_folder/ normally, or config/ when --workspace is set."
        ),
    )
    parser.add_argument(
        "--replicate-experiment",
        action="store_true",
        help="Load previously prepared datasets instead of creating new splits.",
    )
    parser.add_argument(
        "--pre-data-path",
        default=None,
        help="Dataset path used when --replicate-experiment is enabled.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=None,
        help=(
            "Workspace directory used as the base for relative config paths, "
            "default ./data dataset roots, and log outputs. When omitted, current behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        metavar="GPU",
        help=(
            "GPU ids available to experiments. Examples: --gpus 0, --gpus 0,1, or --gpus 0 1. "
            "Omit to use all visible GPUs."
        ),
    )
    parser.add_argument(
        "--seed-count",
        "--repeat-count",
        dest="seed_count",
        type=int,
        default=None,
        help=(
            "Number of distinct model-initialization seeds per config. "
            "Overrides experiment.repeat_count in YAML; seed index i uses "
            "experiment.seed + i. --repeat-count is retained as an alias."
        ),
    )
    if include_repeat_index:
        parser.add_argument(
            "--repeat-index",
            type=int,
            default=None,
            help=argparse.SUPPRESS,
        )
    parser.add_argument(
        "--repeats-per-seed",
        type=int,
        default=1,
        help="Number of independent experiment runs for each seed. Default: 1.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="DataLoader subprocesses used by each experiment. Default: 0.",
    )
    parser.add_argument(
        "--aggregation-device",
        choices=("auto", "cpu", "cuda"),
        default=aggregation_device_default,
        help=(
            "Device for temporary FL scoring, segment construction, and aggregation tensors. "
            f"Default: {aggregation_device_default}."
        ),
    )
    parser.add_argument(
        "--quick-check",
        "--quick_check",
        dest="quick_check",
        action="store_true",
        default=False,
        help=(
            f"Run every planned experiment for {QUICK_CHECK_ROUNDS} federation rounds "
            "using a temporary config. Omit this flag for the configured number of rounds."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the discovered experiment plan without running simulations.",
    )
    return parser


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run federated learning simulations from YAML config files.")
    add_common_cli_arguments(
        parser,
        aggregation_device_default="auto",
        include_repeat_index=True,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    workspace_path = set_workspace(args.workspace)
    if workspace_path is not None:
        print(f"Workspace: {workspace_path}", flush=True)

    config_folder = resolve_config_folder(
        args.config_folder,
        workspace_path=workspace_path,
    )
    runs = build_run_plan(
        config_file=args.config_file,
        config_folder=config_folder,
        repeat_count_override=args.seed_count,
        repeat_index=args.repeat_index,
        repeats_per_seed=args.repeats_per_seed,
    )
    apply_gpu_selection(parse_gpu_ids(args.gpus))
    return run_serial(
        runs,
        replicate_experiment=args.replicate_experiment,
        pre_data_path=args.pre_data_path,
        dataloader_workers=args.dataloader_workers,
        aggregation_device=args.aggregation_device,
        dry_run=args.dry_run,
        quick_check=args.quick_check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
