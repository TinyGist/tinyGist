# TinyGist Artifact

This repository contains the simulation, ESP32/QEMU emulation, and physical-device workflows used by the TinyGist paper. It also contains the configuration suites and a lightweight notebook for inspecting the raw results produced by those workflows.

## 1. Requirements

- A Linux host. The artifact was tested on Ubuntu 22.04.
- Docker with BuildKit; Docker Compose v2 is needed for the checked-in QEMU binary bank.
- At least 16 GB RAM. Parallel simulations and large-client experiments may require substantially more.
- At least 20 GB free space for a source build. A container image that also embeds all prepared datasets needs substantially more space.
- `/dev/net/tun` for QEMU networking, or USB serial devices for the physical-device experiments.

All times in this document are approximate author estimates. They vary with the CPU/GPU, storage, number of parallel workers, dataset cache state, and other workloads on the host.

## 2. Build the image

```bash
git clone <artifact-repository-url> tinyGist
cd tinyGist
docker build -t tinygist-artifact-mobicom26:latest -f Dockerfile .
```

The image contains four workspaces:

- `/usr/sim`: the simulation framework and a default CPU-only PyTorch environment;
- `/usr/emu`: the two ESP32/QEMU projects;
- `/usr/real`: the two physical-device projects;
- `/usr/binary_bank/emulation`: checked-in QEMU flash images and launch scripts.

ESP-IDF v5.2.0 and Xtensa QEMU are installed in the same image. The firmware projects include the files required for compilation.

The default Python executable is ready for simulation. Run `get_idf` before using `idf.py`; this switches `python` to ESP-IDF's environment. Run `get_sim` to return to the simulation Python environment:

```bash
get_idf
idf.py --version
get_sim
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The supplied image installs CPU-only PyTorch. The Dockerfile retains a commented CUDA installation command for users who build a CUDA-enabled variant.

## 3. Simulation quick start

Start a simulation container:

```bash
docker run -it --name tinygist tinygist-artifact-mobicom26:latest
```

Inside the container, first inspect the planned run, then execute a two-round functional check:

```bash
cd /usr/sim
python framework_runner.py \
  --config-file src/config_folder/01_fcn_mnist.yml \
  --gpus cpu \
  --aggregation-device cpu \
  --dry-run

python framework_runner.py \
  --config-file src/config_folder/01_fcn_mnist.yml \
  --gpus cpu \
  --aggregation-device cpu \
  --quick-check
```

`--dry-run` discovers the YAML files and prints the expanded seed/repeat plan. It does not load the dataset, construct a model, validate the complete runtime configuration, or run FL.

`--quick-check` is a boolean flag and takes no value. When present, every planned experiment is run with `federation.rounds` temporarily set to exactly 2. It does not reduce the client count, local epochs, batches, metrics, or number of seed/repeat runs, so a large suite can still take time.

Successful simulation runs create one subdirectory per run under `/usr/sim/log`. Common files include:

- `used_config.yml` and `run_metadata.yml`;
- `fl_simulation.log`;
- `metrics.xlsx`;
- `communication_packets.csv`;
- `wall_clock.csv`;
- `topology_shape_<topology>.svg`.

`importance_correlation.csv` and other probe-specific files are created only by configurations that enable those probes. Dataset-distribution figures are written under `/usr/sim/dataset`.

### Runner selection and GPU options

`framework_runner.py` runs the plan sequentially in one process. `framework_runner_parallel.py` runs the same plan in separate worker processes; parallel scheduling is their main behavioral difference.

Use a configuration directory directly; there is no need to move or replace `src/config_folder`:

```bash
python framework_runner.py --config-folder src/config_folders/contrast_exp
python framework_runner_parallel.py --config-folder src/config_folders/contrast_exp -j 4
```

`--gpus` controls which CUDA devices are visible:

- `--gpus cpu` forces CPU execution;
- `--gpus 0` exposes GPU 0;
- `--gpus 0,1` and `--gpus 0 1` expose GPUs 0 and 1;
- omitting the option keeps all currently visible GPUs available.

For the sequential runner, a list of GPU IDs does not mean multi-GPU training; one experiment still runs at a time and selects one visible device. The parallel runner assigns each worker to one GPU slot in round-robin order. `-j` controls the maximum concurrent experiments; setting it above the GPU count makes workers share GPUs and may exhaust memory.

Numeric GPU IDs are useful only in a CUDA-enabled image. They do not turn the supplied CPU-only PyTorch build into a CUDA build. For comparable runs, also pass the same `--aggregation-device {auto,cpu,cuda}` value to both runners; the sequential default is `auto`, while the parallel default is `cpu`.

Other useful controls are:

```bash
python framework_runner.py --help
python framework_runner_parallel.py --help

# Two initialization seeds, three runs per seed.
python framework_runner.py \
  --config-file src/config_folder/01_fcn_mnist.yml \
  --seed-count 2 \
  --repeats-per-seed 3
```

### Simulation dataset preparation

Torchvision datasets such as MNIST, Fashion-MNIST, CIFAR, EMNIST, and SVHN are downloaded into `/usr/sim/data` when first used. A source-built image does not embed the raw Muscle-Gesture data or the large prepared Speech Commands/COCO-derived datasets.

The repository contains the four Muscle-Gesture CSV files, but `.dockerignore` intentionally keeps datasets out of a source-built image. Copy them from the host into the running simulation container before a Gesture configuration is executed:

```bash
# Host commands, from the tinyGist repository root.
docker exec tinygist mkdir -p /usr/sim/data/muscle-gesture
docker cp workspace_simulation/data/muscle-gesture/. \
  tinygist:/usr/sim/data/muscle-gesture/
docker exec tinygist sh -c \
  'test "$(find /usr/sim/data/muscle-gesture -maxdepth 1 -type f -name "*.csv" | wc -l)" -eq 4'
```

For the full Figure 10 suite, prepare the additional datasets inside the container:

```bash
cd /usr/sim/src/data_getter

# Google Speech Commands and the KWS subset: approximately 7 minutes.
python speech_command_dataset_preparer.py

# Downloads COCO on first use, then prepares Visual Wake Words.
# Approximate first-time download: 80 minutes; processing: 25 minutes.
python vww_dataset_preparer.py

# Reuses the COCO download. Approximate processing time: 20 minutes.
python fomo_dataset_preparer.py
python coco_dataset_preparer.py
```

The smaller `minimum_val/contrast_exp` suite avoids the two COCO object-detection tasks and is suitable when evaluation time or storage is limited.

## 4. Simulation replications

The same prepared container can run several suites. To prevent their results from being mixed, define the following host-side helpers once from the `tinyGist` directory. `mark_sim_results` records the log directories that already exist; `collect_sim_results` copies only directories created by the subsequent suite; `link_result_runs` creates a checked, staged hard-link view for another claim. Run only one suite in this container between the mark and collect calls. Save the runner's console stream with the claim-specific `tee` command shown below; this preserves errors that occur before the framework creates its first run directory.

```bash
cd <path-to-tinyGist>

mark_sim_results() {
  local claim="$1"
  local before="/tmp/tinygist-${claim}.before"
  local temporary
  rm -f -- "$before" || return 1
  temporary="$(mktemp "${before}.XXXXXX")" || return 1

  if docker exec tinygist bash -lc \
    'if test -d /usr/sim/log; then find /usr/sim/log -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | LC_ALL=C sort; fi' \
    > "$temporary"; then
    mv "$temporary" "$before"
  else
    local status=$?
    rm -f -- "$temporary"
    return "$status"
  fi
}

collect_sim_results() {
  local claim="$1"
  local before="/tmp/tinygist-${claim}.before"
  local after="/tmp/tinygist-${claim}.after"
  local destination="$PWD/result_folder_all/results/${claim}/raw"
  local evidence="$PWD/result_folder_all/results/${claim}/evidence"
  local console_source="/tmp/tinygist-${claim}.console.log"
  local temporary
  local staging
  local -a new_directories=()

  test -f "$before" || {
    echo "Missing $before; run mark_sim_results first." >&2
    return 1
  }

  temporary="$(mktemp "${after}.XXXXXX")" || return 1
  if docker exec tinygist bash -lc \
    'if test -d /usr/sim/log; then find /usr/sim/log -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | LC_ALL=C sort; fi' \
    > "$temporary"; then
    mv "$temporary" "$after"
  else
    local status=$?
    rm -f -- "$temporary"
    return "$status"
  fi

  mkdir -p "$destination" "$evidence" || return 1
  if docker exec tinygist test -f "$console_source"; then
    docker cp "tinygist:${console_source}" \
      "$evidence/console_$(date -u +%Y%m%dT%H%M%SZ).log" || return 1
  else
    echo "Warning: no claim-specific console log found at $console_source" >&2
  fi

  mapfile -t new_directories < <(comm -13 "$before" "$after")
  if (( ${#new_directories[@]} == 0 )); then
    echo "Error: this suite created no new /usr/sim/log directory." >&2
    return 1
  fi

  for run_directory in "${new_directories[@]}"; do
    test ! -e "$destination/$run_directory" || {
      echo "Refusing to overwrite $destination/$run_directory" >&2
      return 1
    }
  done

  staging="$(mktemp -d "$(dirname "$destination")/.${claim}.collect.XXXXXX")" || return 1
  for run_directory in "${new_directories[@]}"; do
    if ! docker cp "tinygist:/usr/sim/log/$run_directory" "$staging/"; then
      rm -rf -- "$staging"
      return 1
    fi
  done
  for run_directory in "${new_directories[@]}"; do
    mv -T "$staging/$run_directory" "$destination/$run_directory" || {
      rm -rf -- "$staging"
      return 1
    }
  done
  rmdir "$staging"
}

link_result_runs() {
  local source="$1"
  local destination="$2"
  local label="${3:-linked result}"
  local manifest
  local staging
  local run_directory
  local run_name
  local -a run_directories=()
  local -a committed=()

  test -d "$source" || {
    echo "Missing source result directory: $source" >&2
    return 1
  }

  manifest="$(mktemp /tmp/tinygist-result-links.XXXXXX)" || return 1
  if ! find "$source" -mindepth 1 -maxdepth 1 -type d -print0 > "$manifest"; then
    rm -f -- "$manifest"
    return 1
  fi
  sort -z -o "$manifest" "$manifest" || {
    rm -f -- "$manifest"
    return 1
  }
  mapfile -d '' -t run_directories < "$manifest"
  rm -f -- "$manifest"

  if (( ${#run_directories[@]} == 0 )); then
    echo "No run directories found under $source" >&2
    return 1
  fi

  mkdir -p "$destination" || return 1
  for run_directory in "${run_directories[@]}"; do
    run_name="$(basename "$run_directory")"
    test ! -e "$destination/$run_name" || {
      echo "Refusing to overwrite existing $label input: $destination/$run_name" >&2
      return 1
    }
  done

  staging="$(mktemp -d "$(dirname "$destination")/.result-links.XXXXXX")" || return 1
  for run_directory in "${run_directories[@]}"; do
    if ! cp -al "$run_directory" "$staging/"; then
      rm -rf -- "$staging"
      return 1
    fi
  done

  for run_directory in "${run_directories[@]}"; do
    run_name="$(basename "$run_directory")"
    if mv -T "$staging/$run_name" "$destination/$run_name"; then
      committed+=("$run_name")
    else
      for run_name in "${committed[@]}"; do
        mv "$destination/$run_name" "$staging/$run_name" || true
      done
      echo "Commit failed; staged links remain at $staging" >&2
      return 1
    fi
  done
  rmdir "$staging"
}
```

These helpers assume the container is named `tinygist` and that commands are invoked from the artifact's `tinyGist` directory. They do not delete or move experiment outputs inside the container. If a suite fails before creating a run directory, `collect_sim_results` returns nonzero but still archives the claim-specific console log under `results/<claim>/evidence/`.

### Estimated time per complete replication

The following are author planning estimates for **one complete replication**. A simulation row refers to the complete YAML suite named in the table. A QEMU or physical-device row refers to one task/method/device-count condition; the final column shows the sequential cost of all conditions needed for that figure. Treat these as experiment-execution estimates and plan additional time for dataset downloads, initial image builds, firmware compilation, result copying, and failed attempts. Multiply the estimate by the number of independent repetitions you plan to retain.

| Claim | Complete workload | Estimate for one replication | Sequential cost for the complete claim | Notes |
|---|---:|---:|---:|---|
| Figure 10 | 75 simulation configurations | 5 days on CPU sequentially; 3 days with CPU parallelism; 2 days on one GPU sequentially; 1 day with suitable GPU parallelism | Same as one complete suite | The 60-configuration reduced suite is partial only. |
| Figure 12 | 60 simulation configurations | About 5 hours with sufficient GPU parallelism | About 5 hours | Reduce `-j` when 100- or 200-client runs exceed available memory. |
| Table 3 | 14 importance-metric configurations | About 1 GPU day | About 1 GPU day | Hutchinson estimators dominate the cost; CPU execution can take much longer. |
| Figure 21 | 2 importance-correlation configurations | About 1 hour on the authors' GPU system | About 1 hour | Repeat runs when reproducing the paper's mean/standard-deviation claim. |
| Figure 22 | 12 DP-SGD configurations | About 1 hour with sufficient GPU parallelism | About 1 hour | The validated CPU two-round quick-check is about 3 minutes and is functional-only. |
| Figure 14 — emulation | 12 QEMU task/method/device-count conditions | About 3 hours per condition | About 36 hours | One container runs one condition at a time; use separate Compose services/containers for intentional parallel execution. |
| Figure 14 — real devices | 12 homogeneous ESP32-S3 conditions | About 5 hours per condition | About 60 hours | The full three-environment path may reuse the Figure 12 simulation results. |
| Figure 16 | 12 heterogeneous physical-device conditions | About 7 hours per condition | About 84 hours | Slow devices and per-board flashing/logging make this the longest physical workflow. |

These values are planning aids, not performance guarantees. Cache state, CPU/GPU resources, worker count, USB/Wi-Fi reliability, and asynchronous device scheduling can materially change wall-clock time.

The commands below use `tee` with `pipefail` so the complete console stream is retained without hiding the runner's exit status. Add `--quick-check` before `2>&1` when performing the two-round functional test instead of a full replication.

### Figure 10: comparison across tasks and methods

The full suite contains 75 configurations. The reduced suite contains 60 configurations.

On the host, mark the start of this replication:

```bash
cd <path-to-tinyGist>
mark_sim_results Figure10
```

```bash
cd /usr/sim

# Inspect only.
python framework_runner.py \
  --config-folder src/config_folders/contrast_exp \
  --gpus cpu \
  --dry-run

# Choose one execution command. Full sequential run in the supplied CPU image:
set -o pipefail
python framework_runner.py \
  --config-folder src/config_folders/contrast_exp \
  --gpus cpu \
  --aggregation-device cpu \
  2>&1 | tee /tmp/tinygist-Figure10.console.log

# Reduced parallel alternative for partial validation only:
set -o pipefail
python framework_runner_parallel.py \
  --config-folder src/config_folders/minimum_val/contrast_exp \
  --gpus cpu \
  --aggregation-device cpu \
  -j 4 \
  2>&1 | tee /tmp/tinygist-Figure10.console.log
```

Author estimate: approximately 5 days sequentially or 3 days with CPU parallelism; approximately 2 days sequentially or 1 day with suitable GPU parallelism. Treat these as rough planning values.

Only the 75-configuration full suite covers the complete Figure 10 grid. The reduced suite is for functional or partial validation and must not be presented as a complete Figure 10, 19, or 20 replication.

Immediately after completion, collect the raw logs on the host:

```bash
cd <path-to-tinyGist>
collect_sim_results Figure10
```

Figures 19 and 20 use the communication records from the same experiments. Keep separate notebook input folders without duplicating file contents by making hard links on the host:

```bash
link_result_runs \
  result_folder_all/results/Figure10/raw \
  result_folder_all/results/Figure19/raw \
  Figure19
link_result_runs \
  result_folder_all/results/Figure10/raw \
  result_folder_all/results/Figure20/raw \
  Figure20
```

The paper's Figure 19 uses panel 7 (MobileNetV1–CIFAR-10). Figure 20 uses the same 15 task letters as Figure 10 and compares TX/RX volume for representative `device_0`; its task panels have independent horizontal scales, as in the paper.

Plotting status: **TO_DO until raw results are available.** Afterwards enable the Figure 10, Figure 19, or Figure 20 flag in the lightweight notebook.

### Figure 12: scalability

The suite contains 60 configurations: two tasks, six client counts, and five methods.

On the host:

```bash
cd <path-to-tinyGist>
mark_sim_results Figure12
```

```bash
cd /usr/sim
python framework_runner.py \
  --config-folder src/config_folders/scl_comp \
  --gpus cpu \
  --dry-run

set -o pipefail
python framework_runner_parallel.py \
  --config-folder src/config_folders/scl_comp \
  --gpus cpu \
  --aggregation-device cpu \
  -j 4 \
  2>&1 | tee /tmp/tinygist-Figure12.console.log
```

Author estimate: approximately 5 hours with sufficient GPU parallelism. The 100- and 200-client configurations need substantially more memory than the small-client runs, so reduce `-j` when necessary.

Collect the logs immediately:

```bash
cd <path-to-tinyGist>
collect_sim_results Figure12
```

Plotting status: **TO_DO until raw results are available.** Then enable `RUN_FIGURE_12` in the notebook.

### Table 3: importance metrics

On the host:

```bash
cd <path-to-tinyGist>
mark_sim_results Table3
```

```bash
cd /usr/sim
set -o pipefail
python framework_runner_parallel.py \
  --config-folder src/config_folders/metrics_comp \
  --gpus cpu \
  --aggregation-device cpu \
  -j 2 \
  2>&1 | tee /tmp/tinygist-Table3.console.log
```

Author estimate: approximately 1 day on a GPU system because the Hutchinson estimators are computationally expensive. CPU execution can take considerably longer.

Collect the logs immediately:

```bash
cd <path-to-tinyGist>
collect_sim_results Table3
```

Data sorting status: **TO_DO until raw results are available.** The notebook's Table 3 section prints an organized summary rather than drawing a table.

### Figure 21: importance-score relationship

On the host:

```bash
cd <path-to-tinyGist>
mark_sim_results Figure21
```

```bash
cd /usr/sim
set -o pipefail
python framework_runner_parallel.py \
  --config-folder src/config_folders/metrics_relation \
  --gpus cpu \
  --aggregation-device cpu \
  -j 2 \
  2>&1 | tee /tmp/tinygist-Figure21.console.log
```

Author estimate: approximately 1 hour on the GPU system used by the authors.

Collect the logs immediately:

```bash
cd <path-to-tinyGist>
collect_sim_results Figure21
```

Plotting status: **TO_DO until raw results are available.** Then enable `RUN_FIGURE_21` in the notebook.

### Figure 22: record-level DP-SGD

The minimum suite contains the 12 main-paper epsilon-sweep experiments. The full folder also contains the additional DP experiments discussed in the paper.

For a functional smoke test, use `--quick-check`. This runs every selected configuration for exactly two rounds and verifies that training, DP accounting, result collection, and notebook plotting work. Its accuracy and two-round accountant epsilon are **not** intended to reproduce the paper values. The notebook uses the configured 200-round target epsilon encoded in each source filename only as the condition label, and exports the effective two-round privacy values as provenance.

On the host:

```bash
cd <path-to-tinyGist>
mark_sim_results Figure22
```

```bash
cd /usr/sim

# Functional quick-check used for artifact validation (CPU-safe and sequential):
set -o pipefail
python framework_runner.py \
  --config-folder src/config_folders/minimum_val/DP_comp \
  --gpus cpu \
  --aggregation-device cpu \
  --dataloader-workers 0 \
  --quick-check \
  2>&1 | tee /tmp/tinygist-Figure22.console.log

# Full main-paper Figure 22 replication:
set -o pipefail
python framework_runner_parallel.py \
  --config-folder src/config_folders/minimum_val/DP_comp \
  --gpus cpu \
  --aggregation-device cpu \
  -j 4 \
  2>&1 | tee /tmp/tinygist-Figure22.console.log

# Optional full suite:
set -o pipefail
python framework_runner_parallel.py \
  --config-folder src/config_folders/DP_comp \
  --gpus cpu \
  --aggregation-device cpu \
  -j 4 \
  2>&1 | tee /tmp/tinygist-Figure22.console.log
```

Author estimate: approximately 1 hour for the minimum suite with sufficient GPU parallelism.

The validated CPU-only quick-check completed the 12 configurations in approximately 3 minutes on the validation host. This is only a smoke-test estimate; full 200-round runtime depends on the hardware and selected parallelism.

Collect the logs immediately:

```bash
cd <path-to-tinyGist>
collect_sim_results Figure22
```

After collecting the raw results, enable `RUN_FIGURE_22` in the notebook. For quick-check data, successful execution of this section proves the raw-result-to-plot path; do not interpret the plotted accuracy or privacy values as a numerical reproduction of Figure 22.

## 5. Generate C datasets for firmware

The emulation and physical-device projects already contain MNIST and Gesture C headers for `device_0` through `device_9`. Use this section only when regenerating or extending those shards.

The generators are Python modules and must be run from `/usr/sim`:

```bash
get_sim
cd /usr/sim

# Downloads MNIST through torchvision when it is not cached.
python -m src.data_getter.mnist_getter

# Requires /usr/sim/data/muscle-gesture/*.csv.
python -m src.data_getter.muscle_gesture_getter
```

For a source-built container, copy the four Muscle-Gesture CSV files into the container first:

```bash
# Host command, from the tinyGist repository root.
docker exec tinygist mkdir -p /usr/sim/data/muscle-gesture
docker cp workspace_simulation/data/muscle-gesture/. \
  tinygist:/usr/sim/data/muscle-gesture/
```

The generators produce 200 device shards at:

```text
/usr/sim/Data/dataset_mnist/C_library/
/usr/sim/dataset/dataset_gesture/C_library/
```

Copy the *contents* of `C_library`, preserving the layout expected by each firmware project:

```bash
# Emulation projects.
cp -a /usr/sim/Data/dataset_mnist/C_library/. \
  /usr/emu/esp-adfo_fcn/main/dataset_mnist/
cp -a /usr/sim/dataset/dataset_gesture/C_library/. \
  /usr/emu/esp-adfo_conv/main/dataset_gesture/

# Physical-device projects.
cp -a /usr/sim/Data/dataset_mnist/C_library/. \
  /usr/real/phy-esp-adfo-fcn-mnist/main/dataset_mnist/
cp -a /usr/sim/dataset/dataset_gesture/C_library/. \
  /usr/real/phy-esp-adfo-conv-gesture/main/dataset_gesture/
```

## 6. Figure 14: fidelity across environments

Figure 14 contains 12 task/method/client conditions: Gesture and MNIST, each with DFA, SDFA, and tinyGist, at both 5 and 10 devices. The artifact notebook supports three explicit paths:

| `FIGURE_14_PATH` | Required raw data | Physical boards |
|---|---|---:|
| `emulation_only` | 12 Emulation conditions | 0 |
| `emulation_simulation` | 12 Emulation + 12 Simulation conditions | 0 |
| `emulation_simulation_real` | 12 Emulation + 12 Simulation + 12 Real conditions | 10 homogeneous ESP32-S3 boards at 240 MHz |

Only `emulation_simulation_real` is the complete three-environment reproduction shown in the paper. The other paths are supported partial comparisons. Every selected environment must provide both tasks, all three methods, and both client counts. The notebook reports rounds 0 through 199; emulation and real firmware may continue beyond that point, but later rounds are not included in this figure.

### Generate Emulation data for all paths

QEMU creates a `172.17.0.0/24` bridge inside its container. Disable Docker networking to avoid a conflict with Docker's default bridge:

```bash
docker run -it --name tinygist-emu \
  --network none \
  --cap-add NET_ADMIN \
  --device /dev/net/tun \
  tinygist-artifact-mobicom26:latest
```

Inside `tinygist-emu`, compile both the 5- and 10-device images for both tasks. `build_flash.sh` compiles DFA, SDFA, and tinyGist in one invocation and uses its sole argument for both the number of `DATASET` shards and `MAXIMUM_NUM_DEVICES`:

```bash
get_idf
idf.py --version   # Must report ESP-IDF v5.2.0.

cd /usr/emu/esp-adfo_fcn
./build_flash.sh 5
./build_flash.sh 10

cd /usr/emu/esp-adfo_conv
./build_flash.sh 5
./build_flash.sh 10
```

Each project now has `bin_dir_{dfa,sdfa,gist}_{5,10}`. Copy the common launcher into all 12 output directories:

```bash
LAUNCHER=/usr/binary_bank/emulation/FCN_MNIST/5/DFA/emulate.sh
for PROJECT in /usr/emu/esp-adfo_fcn /usr/emu/esp-adfo_conv; do
  for COUNT in 5 10; do
    for INTERNAL_METHOD in dfa sdfa gist; do
      install -m 0755 "${LAUNCHER}" \
        "${PROJECT}/bin_dir_${INTERNAL_METHOD}_${COUNT}/emulate.sh"
    done
  done
done
```

### One condition per container; optional parallel Compose runs

One `tinygist-emu` container can execute only **one** `emulate.sh` condition at a time. Every launcher creates the same in-container bridge (`br0`) and TAP names (`tap1` through `tapN`), so starting another condition in that container would collide with the running condition's network and log state. Run the 12 compiled conditions sequentially in this container. For example:

```bash
cd /usr/emu/esp-adfo_fcn/bin_dir_dfa_10
./emulate.sh 10
```

The launcher creates `br0`, `tap1` through `tapN`, starts one QEMU per flash image, and writes `logs_N_devices/flash*.log`. Stop with Ctrl-C only after every device log contains the success-rate record for `[ROUND 199]`. The launcher terminates only the QEMU PIDs it created and removes its bridge and TAP devices before returning.

If every device log continues to report communication errors instead of progressing through rounds, the virtual Ethernet/TAP setup has failed and the communication procedure is not valid. Stop that condition, confirm that its QEMU processes, bridge, and TAP devices were removed, and restart the condition. Do not collect or reuse the failed run's logs.

To run multiple independent conditions concurrently, use the supplied Compose files from the **host machine**. Each Compose service starts a separate container and therefore has its own network namespace, bridge, and TAP devices. This host-orchestrated Compose workflow is the preferred way to launch parallel emulation experiments.

The Compose files mount a host directory at `/work`, so newly compiled images must first be copied out of `tinygist-emu`. Copy the complete output directory---both `flash*.bin` files and `emulate.sh`---to a separate host directory, then update or add the Compose service's bind mount, service name, and device count. For example, to prepare a newly compiled FCN/DFA five-device condition:

```bash
cd <path-to-tinyGist>
HOST_RUNS="$PWD/host_emulation_runs/FCN_MNIST/5/DFA"
mkdir -p "$HOST_RUNS"
docker cp tinygist-emu:/usr/emu/esp-adfo_fcn/bin_dir_dfa_5/. "$HOST_RUNS/"
```

Then set the service volume in the appropriate Compose file to `"${HOST_RUNS}:/work"` (replace the shell variable with its absolute path in YAML), keep `network_mode: "none"`, `NET_ADMIN`, and `/dev/net/tun`, and launch the service from the host:

```bash
docker compose -f docker-compose-fcn-5.yml up fcn-5-dfa
```

Use a separate host output directory and a distinct Compose service/container name for every concurrent condition. Do not run more than one `emulate.sh` process inside the same `tinygist-emu` container, and collect each container's logs into a distinct run ID.

Collect each condition immediately. Use the paper-facing method directory `tinyGist` for the internal `gist` image:

```text
result_folder_all/results/Figure14/raw/emulation/
├── FCN_MNIST/{DFA,SDFA,tinyGist}/{5_devices,10_devices}/<run-id>/flash*.log
└── Conv_Gesture/{DFA,SDFA,tinyGist}/{5_devices,10_devices}/<run-id>/flash*.log
```

The following host-side collection example is parameterized for one condition and fails if the number of nonempty logs, device IDs, or round-199 records is incomplete:

```bash
(
set -euo pipefail
cd <path-to-tinyGist>

TASK=FCN_MNIST
METHOD=DFA
COUNT=10
case "${TASK}" in
  FCN_MNIST) PROJECT=esp-adfo_fcn ;;
  Conv_Gesture) PROJECT=esp-adfo_conv ;;
  *) echo "Unsupported Figure 14 task: ${TASK}" >&2; exit 1 ;;
esac
case "${METHOD}" in
  DFA) INTERNAL_METHOD=dfa ;;
  SDFA) INTERNAL_METHOD=sdfa ;;
  tinyGist) INTERNAL_METHOD=gist ;;
  *) echo "Unsupported Figure 14 method: ${METHOD}" >&2; exit 1 ;;
esac
case "${COUNT}" in
  5|10) ;;
  *) echo "Figure 14 device count must be 5 or 10." >&2; exit 1 ;;
esac
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SOURCE="/usr/emu/${PROJECT}/bin_dir_${INTERNAL_METHOD}_${COUNT}/logs_${COUNT}_devices"
DEST="result_folder_all/results/Figure14/raw/emulation/${TASK}/${METHOD}/${COUNT}_devices/${RUN_ID}"
PARENT="$(dirname "${DEST}")"
STAGING=""
cleanup_staging() {
  if test -n "${STAGING}" && test -d "${STAGING}"; then
    rm -rf -- "${STAGING}"
  fi
}
trap cleanup_staging EXIT

mkdir -p "${PARENT}"
test ! -e "${DEST}"
docker exec tinygist-emu test -d "${SOURCE}"
STAGING="$(mktemp -d "${PARENT}/.${RUN_ID}.collect.XXXXXX")"
docker cp "tinygist-emu:${SOURCE}/." "${STAGING}/"
test "$(find "${STAGING}" -maxdepth 1 -type f -name 'flash*.log' -size +0c | wc -l)" -eq "${COUNT}"
for LOG in "${STAGING}"/flash*.log; do
  grep -Eq "This Device.s ID: [0-9]+" "${LOG}"
  grep -Eq '\[ROUND 199\].*### Success Rate:' "${LOG}"
done
test "$(
  grep -h -m1 -o "This Device.s ID: [0-9]*" "${STAGING}"/flash*.log |
    grep -o '[0-9]*$' | sort -u | wc -l
)" -eq "${COUNT}"
ACTUAL_IDS="$(
  grep -h -m1 -o "This Device.s ID: [0-9]*" "${STAGING}"/flash*.log |
    grep -o '[0-9]*$' | sort -nu
)"
EXPECTED_IDS="$(seq 10 "$((9 + COUNT))")"
test "${ACTUAL_IDS}" = "${EXPECTED_IDS}"
mv -T "${STAGING}" "${DEST}"
STAGING=""
trap - EXIT
)
```

For a ten-minute mechanism smoke test, replace the full run with:

```bash
status=0
timeout --foreground --signal=INT --kill-after=20s 10m ./emulate.sh "${COUNT}" || status=$?
case "${status}" in
  124|130) ;;
  *) echo "Emulation ended unexpectedly with status ${status}." >&2; exit 1 ;;
esac
```

Then verify that no `qemu-system-xtensa`, `br0`, or TAP device remains and collect the logs with a smoke-specific run ID. Reuse the staged collection block, but replace its `[ROUND 199]` check with `grep -Eq '\[ROUND [0-9]+\].*### Success Rate:'`. A smoke test need only contain at least one success-rate record per device; it is not a complete Figure 14 input and must be plotted with `STRICT_INPUT=False`.

The checked-in emulation binary bank contains only the six 5-device smoke conditions. They can also be launched from the host with Compose without copying newly compiled binaries:

```bash
cd <path-to-tinyGist>/binary_bank/emulation/FCN_MNIST
docker compose -f docker-compose-fcn-5.yml up fcn-5-dfa
```

Specify one service for a single-condition run. Omitting the service name starts all three methods concurrently in three separate containers; this is valid only when parallel runs are intentional and their logs are collected independently. The build-only validator verifies the 30 ESP-IDF v5.2.0 `flash*.bin` files and removes its checksum manifest before the binary bank enters the final image.

### Add Simulation data for paths 2 and 3

Only 12 configurations from `scl_comp` are relevant. First, on the host, record the simulation container's existing run directories with a staging claim:

```bash
cd <path-to-tinyGist>
mark_sim_results Figure14-simulation
```

Then run the 12 configurations sequentially inside the prepared simulation container:

```bash
cd /usr/sim
set -euo pipefail
for CONFIG in \
  src/config_folders/scl_comp/{01_cnn_gesture_recognition,02_fcn_mnist}_{dfa,sdfa,tinygist}_{5,10}.yml
do
  python framework_runner.py \
    --config-file "${CONFIG}" \
    --gpus cpu \
    --aggregation-device cpu
done 2>&1 | tee /tmp/tinygist-Figure14-simulation.console.log
```

Immediately afterward, return to the host, collect the staging claim, and create the checked hard-link view at the nested path read by the notebook:

```bash
cd <path-to-tinyGist>
collect_sim_results Figure14-simulation
link_result_runs \
  result_folder_all/results/Figure14-simulation/raw \
  result_folder_all/results/Figure14/raw/simulation \
  Figure14-simulation
```

If Figure 12 has already been collected, skip the new simulation run and link it directly instead; the notebook filters that 60-configuration suite to the 12 Figure 14 conditions:

```bash
link_result_runs \
  result_folder_all/results/Figure12/raw \
  result_folder_all/results/Figure14/raw/simulation \
  Figure14-simulation
```

### Add homogeneous Real data for path 3

This path is distinct from the heterogeneous Figure 16 topology. It requires a pool of **10 ESP32-S3 devices, all configured at 240 MHz**. The 5-device trials reuse devices `0` through `4`; the 10-device trials use devices `0` through `9`. During a 5-device trial, power off or disconnect the other five boards from both serial USB and the experiment LAN. `multi_esp_monitor.py` intentionally resets and records every matching `ttyACM<number>`/`ttyUSB<number>` port it finds.

Start a separate container with serial-device access:

```bash
docker run -it --name tinygist-real \
  --privileged \
  -v /dev:/dev \
  tinygist-artifact-mobicom26:latest
```

The boards and container must share a `192.168.1.0/24` LAN. Set `WIFI_SSID` and `WIFI_PASS` in `main/adfo-com.c` in each project before compiling, and reserve `192.168.1.40` through `.49` so they are not assigned to other hosts. Disconnect every unrelated `ttyACM<number>` or `ttyUSB<number>` device before monitoring.

For each of the two real-device projects, create a dedicated ESP-IDF 5.2 ESP32-S3/240 MHz configuration using the common and high-tier defaults, then verify the board-specific Flash, PSRAM, and console selections in `menuconfig`:

```bash
get_idf
idf.py --version   # Must report ESP-IDF v5.2.0.
cd /usr/real/phy-esp-adfo-fcn-mnist
# cd /usr/real/phy-esp-adfo-conv-gesture

idf.py -B build-figure14-esp32s3-240 \
  -DSDKCONFIG=sdkconfig.figure14.esp32s3-240 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.high" \
  -DDATASET=0 set-target esp32s3
idf.py -B build-figure14-esp32s3-240 \
  -DSDKCONFIG=sdkconfig.figure14.esp32s3-240 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.high" \
  -DDATASET=0 menuconfig
```

In `menuconfig`, verify the complete IDF 5.2 configuration rather than only the CPU frequency: select the module's actual flash size (confirm that the supplied 8 MB default matches it; these partitions require at least 4 MB) and enable header-size update, use the custom `partitions.csv`, performance (`-O2`) optimization, ESP32-S3 PSRAM mode/speed/pins matching the exact board, 240 MHz CPU, the correct UART or USB console, main-task stack size `100000`, **Print registers and halt**, interrupt-watchdog timeout `300 ms`, task watchdog disabled, and power management disabled. Section 7 gives the full menu paths and noninteractive verification command for these settings.

Select one method, deriving the internal numeric ID from the same paper-facing name used for the result directory:

```bash
METHOD=DFA   # DFA, SDFA, or tinyGist
case "${METHOD}" in
  DFA) METHOD_ID=1 ;;
  SDFA) METHOD_ID=2 ;;
  tinyGist) METHOD_ID=3 ;;
  *) echo "Unsupported Figure 14 method: ${METHOD}" >&2; exit 1 ;;
esac
./set_method.sh "${METHOD_ID}"
```

Then build and flash every participating board **one at a time**. Do not use `build_flash.sh` for this path because it cannot make the target/configuration/port mapping explicit:

```bash
DATASET_INDEX=0
PORT=/dev/ttyACM0
idf.py -B build-figure14-esp32s3-240 \
  -DSDKCONFIG=sdkconfig.figure14.esp32s3-240 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.high" \
  -DDATASET="${DATASET_INDEX}" -p "${PORT}" build flash
```

Use contiguous `DATASET` indices `0..N-1`; they produce firmware device IDs and IP suffixes `40..(39+N)`. For every new task, method, or client count, run `set_method.sh` again, rebuild, and reflash every active board; a cluster must never mix firmware methods. After flashing exactly 5 or 10 visible boards, restart and collect them as one barrier:

```bash
cd /usr/real
TINYGIST_LOG_DIR=/usr/real/logs python multi_esp_monitor.py
```

Stop after every device log contains `[ROUND 199] ... ### Success Rate:`. Copy the resulting run directory immediately to:

```text
result_folder_all/results/Figure14/raw/real/
├── FCN_MNIST/{DFA,SDFA,tinyGist}/{5_devices,10_devices}/<run-id>/
└── Conv_Gesture/{DFA,SDFA,tinyGist}/{5_devices,10_devices}/<run-id>/
```

Use a staged host-side copy so an incomplete run cannot merge with an existing result. `device_<monitor_index>.log` is named by serial-port discovery order, not by `DATASET`; the collection code derives the firmware `device_id` from each log:

```bash
(
set -euo pipefail
cd <path-to-tinyGist>

TASK=FCN_MNIST
METHOD=DFA
COUNT=10
case "${TASK}" in
  FCN_MNIST) TASK_LABEL=MNIST ;;
  Conv_Gesture) TASK_LABEL=Gesture ;;
  *) echo "Unsupported Figure 14 task: ${TASK}" >&2; exit 1 ;;
esac
case "${METHOD}" in
  DFA|SDFA|tinyGist) ;;
  *) echo "Unsupported Figure 14 method: ${METHOD}" >&2; exit 1 ;;
esac
case "${COUNT}" in
  5|10) ;;
  *) echo "Figure 14 device count must be 5 or 10." >&2; exit 1 ;;
esac
RUN_ID="20260826_123456_000000"  # Replace with the monitor's directory name.
SOURCE="/usr/real/logs/${RUN_ID}"
DEST="result_folder_all/results/Figure14/raw/real/${TASK}/${METHOD}/${COUNT}_devices/${RUN_ID}"
PARENT="$(dirname "${DEST}")"
STAGING=""
cleanup_staging() {
  if test -n "${STAGING}" && test -d "${STAGING}"; then
    rm -rf -- "${STAGING}"
  fi
}
trap cleanup_staging EXIT

mkdir -p "${PARENT}"
test ! -e "${DEST}"
docker exec tinygist-real test -d "${SOURCE}"
STAGING="$(mktemp -d "${PARENT}/.${RUN_ID}.collect.XXXXXX")"
docker cp "tinygist-real:${SOURCE}/." "${STAGING}/"
test -s "${STAGING}/device_map.json"
test "$(find "${STAGING}" -maxdepth 1 -type f -name 'device_*.log' -size +0c | wc -l)" -eq "${COUNT}"
for LOG in "${STAGING}"/device_*.log; do
  grep -Eq "This Device.s ID: [0-9]+" "${LOG}"
  grep -Eq '\[ROUND 199\].*### Success Rate:' "${LOG}"
done
test "$(
  grep -h -m1 -o "This Device.s ID: [0-9]*" "${STAGING}"/device_*.log |
    grep -o '[0-9]*$' | sort -u | wc -l
)" -eq "${COUNT}"
ACTUAL_IDS="$(
  grep -h -m1 -o "This Device.s ID: [0-9]*" "${STAGING}"/device_*.log |
    grep -o '[0-9]*$' | sort -nu
)"
EXPECTED_IDS="$(seq 40 "$((39 + COUNT))")"
test "${ACTUAL_IDS}" = "${EXPECTED_IDS}"

CLUSTER="${COUNT}H/0M/0L"
printf '%s\n' 'path,task,method,client_count,cluster,device_tier,board,target,cpu_mhz,device_id,idf_version' \
  > "${STAGING}/manifest.csv"
for LOG in "${STAGING}"/device_*.log; do
  DEVICE_ID="$(
    grep -m1 -o "This Device.s ID: [0-9]*" "${LOG}" | grep -o '[0-9]*$'
  )"
  printf '%s,%s,%s,%s,%s,High,ESP32-S3,esp32s3,240,%s,5.2.0\n' \
    "$(basename "${LOG}")" "${TASK_LABEL}" "${METHOD}" "${COUNT}" \
    "${CLUSTER}" "${DEVICE_ID}" >> "${STAGING}/manifest.csv"
done
test "$(wc -l < "${STAGING}/manifest.csv")" -eq "$((COUNT + 1))"
mv -T "${STAGING}" "${DEST}"
STAGING=""
trap - EXIT
)
```

The resulting `manifest.csv` has one row per log:

```csv
path,task,method,client_count,cluster,device_tier,board,target,cpu_mhz,device_id,idf_version
device_0.log,MNIST,DFA,10,10H/0M/0L,High,ESP32-S3,esp32s3,240,40,5.2.0
```

For 5 devices use cluster `5H/0M/0L`; for 10 use `10H/0M/0L`. The notebook validates the exact device count and, with `STRICT_INPUT=True`, requires every Real row to identify ESP32-S3, target `esp32s3`, 240 MHz, and ESP-IDF 5.2.0.

### Select the notebook path

Set the Figure 14 controls near the top of `tinygist_artifact_figure.ipynb`, then run from the top:

```python
RUN_FIGURE_14 = True
STRICT_INPUT = True
FIGURE_14_PATH = 'emulation_only'
# FIGURE_14_PATH = 'emulation_simulation'
# FIGURE_14_PATH = 'emulation_simulation_real'
```

The paths write separate PDF, PNG, summary CSV, and coverage CSV files, so one comparison does not overwrite another. With strict validation, every selected environment must have all 12 conditions, exactly 5 or 10 unique device curves per run, and rounds `0..199` for every device. Use `STRICT_INPUT=False` only for a bounded parser/plot smoke test; it permits a partial condition grid but still rejects a malformed run whose usable device count differs from its declared client count.

Plotting status: **TO_DO until the selected raw results are available.**

## 7. Physical-device replication: Figure 16

Reuse the `tinygist-real` container from Section 6 if it already exists. Otherwise, create a container with access to all host devices:

```bash
docker run -it --name tinygist-real \
  --privileged \
  -v /dev:/dev \
  tinygist-artifact-mobicom26:latest
```

The devices and the container must reach the same `192.168.1.0/24` LAN. Before compiling, fill in `WIFI_SSID` and `WIFI_PASS` in the selected project's `main/adfo-com.c`. Ensure that `192.168.1.40` through `192.168.1.49` are not assigned to other hosts.

Select the FL method before building:

```bash
cd /usr/real/phy-esp-adfo-conv-gesture
# cd /usr/real/phy-esp-adfo-fcn-mnist
./set_method.sh 1   # 1=DFA, 2=SDFA, 3=tinyGist (called Gist_Ada by the script)
```

Use the paper-facing name `tinyGist` in result directories and manifests. `Gist_Ada` is only the legacy internal label used by `set_method.sh` for method 3.

### Configure each hardware tier with `menuconfig`

Figure 16 uses three hardware tiers. The CPU frequency is part of the experiment and must be selected explicitly and verified instead of relying on the default chosen by `idf.py set-target`:

| Tier | Physical board | ESP-IDF target | CPU frequency | 10-device cluster | 5-device cluster |
|---|---|---|---:|---:|---:|
| High (H) | ESP32-S3 | `esp32s3` | 240 MHz | 4 | 2 |
| Mid (M) | M5Stack Core2 | `esp32` | 160 MHz | 3 | 2 |
| Low (L) | ESP32-WROVER | `esp32` | 80 MHz | 3 | 1 |

All Figure 16 configurations must be created, edited, built, and flashed with the container's **ESP-IDF 5.2.0** environment. Run the following configuration procedure separately inside both the Gesture and MNIST project directories. Each project supplies an IDF 5.2 target-neutral common defaults file and three CPU-tier overlays; they establish the framework settings while leaving Flash, PSRAM electrical details, and the console transport for `menuconfig`. `idf.py set-target` regenerates the project configuration. Keep a separate `sdkconfig` and build directory for every tier, including the two tiers that both use the `esp32` target. The project requires a `DATASET` value during CMake configuration, so the `set-target` and `menuconfig` commands below use `0` only as a configuration-time placeholder:

```bash
get_idf
idf.py --version   # Must report ESP-IDF v5.2.0; stop here if it does not.

# High tier: ESP32-S3 at 240 MHz.
idf.py -B build-idf52-esp32s3-240 \
  -DSDKCONFIG=sdkconfig.idf52.esp32s3-240 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.high" \
  -DDATASET=0 set-target esp32s3
idf.py -B build-idf52-esp32s3-240 \
  -DSDKCONFIG=sdkconfig.idf52.esp32s3-240 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.high" \
  -DDATASET=0 menuconfig

# Mid tier: M5Stack Core2 at 160 MHz.
idf.py -B build-idf52-esp32-160 \
  -DSDKCONFIG=sdkconfig.idf52.esp32-160 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.mid" \
  -DDATASET=0 set-target esp32
idf.py -B build-idf52-esp32-160 \
  -DSDKCONFIG=sdkconfig.idf52.esp32-160 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.mid" \
  -DDATASET=0 menuconfig

# Low tier: ESP32-WROVER at 80 MHz.
idf.py -B build-idf52-esp32-80 \
  -DSDKCONFIG=sdkconfig.idf52.esp32-80 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.low" \
  -DDATASET=0 set-target esp32
idf.py -B build-idf52-esp32-80 \
  -DSDKCONFIG=sdkconfig.idf52.esp32-80 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.low" \
  -DDATASET=0 menuconfig
```

Run `set-target` only when creating a tier configuration for the first time. Later builds must reuse that tier's saved `sdkconfig`; running `set-target` again can replace it and save the old file as a `.old` backup.

For each of the three configurations, review these settings before building:

- **Partition Table → Partition Table:** select **Custom partition table CSV**, set the filename to `partitions.csv`, and keep the offset at `0x8000`. The supplied Gesture project uses a 3 MiB factory partition; the supplied MNIST project uses `0x340000`. Do not use the default 1 MiB single-app partition—the MNIST firmware does not fit in it.
- **Serial flasher config:** enable **Detect flash size when flashing bootloader**, then select the flash size, SPI mode, and SPI frequency supported by the exact board module. The selected size must not exceed the physically installed flash and must cover the end of the supplied factory partition (`0x310000` for Gesture and `0x350000` for MNIST); a 4 MiB or larger physical flash satisfies those partition layouts. Do not write an 8 MiB image header for a board that only has 4 MiB of flash.
- **Compiler options → Optimization Level:** select **Optimize for performance**.
- **Component config → ESP System Settings → CPU frequency:** select 240 MHz for ESP32-S3, 160 MHz for M5Stack Core2, and 80 MHz for ESP32-WROVER, exactly as listed above. The physical board type, ESP-IDF target, and resulting `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ` value together define the H/M/L tier.
- **Component config → Power Management:** leave dynamic power management disabled (`CONFIG_PM_ENABLE` unset), so the CPU remains at the configured tier frequency during the experiment.
- **Component config → ESP System Settings → Main task stack size:** set it to `100000` bytes.
- **Component config → ESP PSRAM:** enable external RAM, initialize it at boot, make it available through `malloc()`, keep the memory test enabled, set “always allocate internally” to `16384`, and reserve `32768` bytes of internal memory. Select the Quad/Octal mode, type, and clock that match the exact board module; these electrical settings are board-specific and must not be copied blindly between ESP32-S3, Core2, and WROVER.
- **Component config → ESP System Settings:** select the console transport actually connected to the host. Use UART0 at 115200 baud for a board connected through an external USB-UART bridge; use the matching USB Serial/JTAG console option for a board connected through native ESP32-S3 USB. Select **Print registers and halt** for panic handling, enable the interrupt watchdog with a 300 ms timeout, and disable the task watchdog because a local training step can be long.

After leaving `menuconfig`, inspect all three tier files before flashing. Confirm the target and CPU choice as well as the hardware-specific Flash, PSRAM, and console selections:

```bash
for CONFIG_FILE in \
  sdkconfig.idf52.esp32s3-240 \
  sdkconfig.idf52.esp32-160 \
  sdkconfig.idf52.esp32-80
do
  printf '\n%s\n' "${CONFIG_FILE}"
  grep -E '^(CONFIG_IDF_TARGET=|CONFIG_ESPTOOLPY_(FLASH|HEADER_FLASHSIZE_UPDATE=)|CONFIG_PARTITION_TABLE_|CONFIG_COMPILER_OPTIMIZATION_PERF=|CONFIG_SPIRAM|CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ|CONFIG_ESP_SYSTEM_PANIC_PRINT_HALT=|CONFIG_ESP_MAIN_TASK_STACK_SIZE=|CONFIG_ESP_INT_WDT_TIMEOUT_MS=|CONFIG_ESP_CONSOLE_)|^# CONFIG_(PM_ENABLE|ESP_TASK_WDT_EN) is not set' \
    "${CONFIG_FILE}"
done
```

### Build and flash one device at a time

Assign every physical board a unique `DATASET` index. It selects `device_<index>`, sets the firmware device ID to `40 + index`, and assigns the last IP octet to the same value. Reuse the matching tier configuration but pass the board's real `DATASET` index and serial port to `build flash`:

```bash
# Example high-tier ESP32-S3 using device_0.
idf.py -B build-idf52-esp32s3-240 \
  -DSDKCONFIG=sdkconfig.idf52.esp32s3-240 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.high" \
  -DDATASET=0 -p /dev/ttyACM0 build flash

# Example mid-tier Core2 using device_4.
idf.py -B build-idf52-esp32-160 \
  -DSDKCONFIG=sdkconfig.idf52.esp32-160 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.mid" \
  -DDATASET=4 -p /dev/ttyUSB0 build flash

# Example low-tier WROVER using device_7.
idf.py -B build-idf52-esp32-80 \
  -DSDKCONFIG=sdkconfig.idf52.esp32-80 \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.idf52.common;sdkconfig.defaults.idf52.low" \
  -DDATASET=7 -p /dev/ttyUSB1 build flash
```

The indices above only illustrate the command format; choose and record an explicit mapping for the connected boards. Repeat `idf.py ... build flash` for every board, one device at a time, and keep a manifest of `port,board,target,cpu_mhz,DATASET,task,method,idf_version`, with `idf_version` set to `5.2.0`. Before starting a different method, run `set_method.sh` again and rebuild and reflash every board so that one cluster never contains mixed-method firmware. Do **not** use `build_flash.sh` for the heterogeneous Figure 16 topology: it cannot identify each board's target or CPU tier and cannot preserve an explicit mixed-device mapping.

For a separate homogeneous maintenance batch, `build_flash.sh` is guarded against implicit configuration fallback and requires both `TINYGIST_SDKCONFIG=<tier-specific-sdkconfig>` and `TINYGIST_BUILD_DIR=<matching-build-directory>`. This helper is not the Figure 16 replication path.

### Synchronized restart and logging

After all devices have been flashed, start the monitor from `/usr/real`:

```bash
cd /usr/real
get_idf
python multi_esp_monitor.py
```

The monitor automatically discovers `ttyACM<number>` and `ttyUSB<number>` ports, opens all of them, holds every connected ESP in reset through DTR/RTS, releases them as one barrier, and only then starts one log stream per port. It writes a timestamped run directory under `/usr/real/logs` containing `device_map.json` and `device_<monitor_index>.log`. Disconnect unrelated USB serial devices before starting it. Stop logging with Ctrl-C.

Collect each completed task/method run immediately on the host:

```bash
cd <path-to-tinyGist>
RUN_ID="<run-id>"
DEST="result_folder_all/results/Figure16/raw/real/Conv_Gesture/tinyGist/4H_3M_3L/${RUN_ID}"
mkdir -p "${DEST}"
docker cp "tinygist-real:/usr/real/logs/${RUN_ID}/." "${DEST}/"
```

Keep the `<run-id>` directory: it is the replication identity and prevents a later run from overwriting earlier logs. Place a `manifest.csv` in `${DEST}` next to the log files, with one row per device log. The minimum columns are `path,task,method,client_count,cluster,device_tier,board,target,cpu_mhz,device_id,idf_version`. For example:

```csv
path,task,method,client_count,cluster,device_tier,board,target,cpu_mhz,device_id,idf_version
device_0.log,Gesture,tinyGist,10,4H/3M/3L,High,ESP32-S3,esp32s3,240,40,5.2.0
```

The log itself also reports each firmware device ID, so the notebook does not assume that the monitor index equals the manually assigned `DATASET` index.

Plotting status: **TO_DO until raw results are available.** Then enable `RUN_FIGURE_16` in the notebook.

## 8. Lightweight result notebook

The notebook [tinygist_artifact_figure.ipynb](result_folder_all/tinygist_artifact_figure.ipynb) reads the raw experiment outputs directly; it does not require preprocessed NumPy arrays. Each replication has an independent Boolean flag near the top of the notebook. All flags default to `False`, so the notebook can be opened and executed before results are collected.

Raw inputs are kept in individual directories:

```text
result_folder_all/results/Figure10/raw/
result_folder_all/results/Figure12/raw/
result_folder_all/results/Table3/raw/
result_folder_all/results/Figure19/raw/
result_folder_all/results/Figure20/raw/
result_folder_all/results/Figure21/raw/
result_folder_all/results/Figure22/raw/
result_folder_all/results/Figure14/raw/
result_folder_all/results/Figure16/raw/
```

Create a small notebook environment on the host:

```bash
cd <path-to-tinyGist>/result_folder_all
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook tinygist_artifact_figure.ipynb
```

Use the `requirements.txt` in `result_folder_all` for this environment. It explicitly installs `openpyxl`, which pandas uses to read the simulation `metrics.xlsx` files. The separate `workspace_simulation/requirements.txt` describes the simulation runtime: that runtime writes workbooks with `XlsxWriter` and therefore does not require `openpyxl` unless you also intend to run notebook-reading or validation code inside the simulation container.

Set only the flag for the result set you have collected, then run the notebook from the top. Figure 14 additionally uses `FIGURE_14_PATH` to select Emulation only, Emulation versus Simulation, or the full Emulation/Simulation/Real comparison described in Section 6. Each enabled figure is displayed inline even with the notebook's headless `Agg` backend and is also saved under the corresponding `output/` directory. The Table 3 section prints a grouped, organized summary.
