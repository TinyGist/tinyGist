# Workspace Simulation

PyTorch simulation framework for decentralized and distributed federated-learning experiments, including kernel/channel/layer model partitioning, importance-guided communication, Fisher-weighted aggregation, and non-IID client data.

Experiments use strict YAML schema version 2. The supported configuration contract is the current schema-v2 content under `src/config_folder/` and `src/config_folders/`; archived configs and legacy key/value aliases are not compatibility targets.

## Quick Start

Inside the artifact container, select the default simulation environment:

```bash
get_sim
```

For a standalone source checkout, create a local virtual environment instead:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate the commented full example without starting training:

```bash
python -c "from src.sim_tools.simulation_config import load_simulation_config; print(load_simulation_config('src/config_folder/01_fcn_mnist.yml').fl_method)"
```

Run one config:

```bash
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml
```

Preview or run every YAML file recursively under the default `src/config_folder/`:

```bash
python framework_runner.py --dry-run
python framework_runner.py
python framework_runner_parallel.py --dry-run
python framework_runner_parallel.py
```

Run every discovered configuration as a two-round FL smoke test without
modifying the source YAML files:

```bash
python framework_runner.py --quick-check
python framework_runner_parallel.py --quick-check
```

`--quick_check` is accepted as an alias. When the flag is omitted, each YAML
file's configured `federation.rounds` is used. Quick-check creates a
temporary per-run config with `federation.rounds: 2`; seed and same-seed repeat
expansion are otherwise unchanged. If `--dry-run` and quick-check are both
enabled, dry-run takes precedence and only prints the planned experiments.

Select GPUs explicitly, expose all visible GPUs, or force CPU:

```bash
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml --gpus 0
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml --gpus 0,1
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml --gpus cpu
```

With multiple GPU IDs, the sequential runner exposes the full list and one
experiment selects one visible device. The parallel runner assigns one listed
GPU to each worker slot; neither runner performs multi-GPU training for a
single experiment.

The simulation selects CUDA automatically when it is available. Training,
validation, testing, and gradient recording stay on the model device. Temporary
federated-learning tensors can be placed independently with
`--aggregation-device auto|cpu|cuda`. This covers score post-processing, block
reduction, segment construction, communication simulation, and aggregation;
it does not move the models or optimizers.

## Runner Behavior

Both runners recursively discover `.yml` and `.yaml` files in a requested folder.

`experiment.repeat_count` defaults to `1` and denotes the number of distinct
model-initialization seeds. Zero-based seed index `i` uses `seed + i`, so every
sweep includes the configured base seed. Both runners accept `--seed-count`;
the existing `--repeat-count` spelling remains an alias. Either option overrides
YAML. The model seed is isolated to CPU parameter construction: it does not
seed Python, NumPy, CUDA runtime streams, dataset partitioning, DataLoader
order, augmentation, or other runtime random streams.

Both runners support `--repeats-per-seed`, creating independent experiment runs
with the same model-initialization seed and unique log labels. The sequential
runner executes them one at a time, while the parallel runner schedules them as
separate processes:

```bash
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml --seed-count 3 --repeats-per-seed 3
python framework_runner_parallel.py --max-workers 9 --repeats-per-seed 3 --seed-count 3 --dataloader-workers 0
python framework_runner_parallel.py --max-workers 9 --repeats-per-seed 3 --seed-count 1 --dataloader-workers 0
```

The sequential runner uses `DataLoader(num_workers=0)` by default. The parallel
runner can forward an independent per-experiment value; this does not limit the
number of experiment processes assigned to one GPU. For these lightweight,
short local epochs, keep it at `0` unless a benchmark shows that subprocess
loading helps:

```bash
python framework_runner_parallel.py --max-workers 8 --gpus 0,1 --dataloader-workers 0
```

The sequential runner defaults to `--aggregation-device auto`, which uses the
model device and preserves the single-experiment behavior. The parallel runner
defaults to `--aggregation-device cpu`, so several experiments sharing one GPU
do not also compete for thousands of small block-scoring and aggregation
kernels. Use CUDA explicitly only when an experiment has a GPU to itself or a
benchmark shows a benefit:

```bash
python framework_runner_parallel.py --max-workers 8 --gpus 0,1 --aggregation-device cpu
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml --aggregation-device cuda
```

Block methods compile the primitive-block layout once per experiment. Later
rounds pack model and score tensors into one block-ordered vector, perform
block reductions and aggregation with batched tensor operations, and unpack
once per aggregated model. This avoids rebuilding or scanning hundreds of
thousands of Python block objects every round. CPU placement transfers each
canonical parameter/score vector once rather than transferring refined blocks
one by one. The requested and resolved placement are recorded in
`run_metadata.yml`.

Run the complete test suite from the repository root with `pytest`. Test
discovery is restricted to `tests/`, so local datasets and experiment logs are
not recursively scanned.

Both runners support `--workspace`. After switching to that directory, relative dataset, config, and output paths resolve from the workspace. Without an explicit config path, workspace mode reads `config/`.

## Project Layout

| Path | Purpose |
| --- | --- |
| `framework_runner.py` | Sequential runner for one config or a recursive config folder. |
| `framework_runner_parallel.py` | Multi-process experiment scheduler. |
| `src/tools/` | Standalone config analysis, privacy-budget, and clipping calibration tools. |
| `src/config_folder/` | Default recursively discovered schema-v2 configs. |
| `src/config_folders/` | Grouped paper-experiment config suites. |
| `src/sim_tools/simulation_config.py` | Strict schema parser and typed runtime config. |
| `src/sim_tools/simulation_manager_tool.py` | Dataset/model lifecycle and round orchestration. |
| `src/fl_methods/` | Centralized, SegmentPulling, DFA, SDFA, and Gist methods. |
| `src/scoring/` | Metric registry and GPU-aware score buffers. |
| `src/differential_privacy/` | Standalone model-update DP, local DP-SGD, Poisson sampling, RDP accounting, and privacy CSV output. |
| `src/models/parameter_vector.py` | Canonical scope-aware vector, BatchNorm, and block model interface. |
| `src/utils/early_stop.py` | Reusable local-training early-stop controller. |
| `src/experiment_probes/` | Optional importance-metric correlation diagnostics. |
| `log/` | Config snapshots, metadata, metrics, timings, and probe outputs. |

## Schema Version 2

Every config starts with:

```yaml
schema_version: 2
```

Unknown keys fail immediately with their section path. There is no fallback to the former `training/device_count`, `method.base_unit.score`, `score_buffer`, `previous`, or multi-stage score-source layout.

Core model and scoring code follows the same strict-interface rule: scope is
selected through the canonical vector/block methods, and post-training score
pipelines expose only the batch-aware entry points used by the simulator.

The top-level structure is:

```yaml
schema_version: 2
experiment: {}
federation: {}
dataset: {}
model: {}
training: {}
method: {}
differential_privacy: null  # optional; null/omission disables it
probes: {}   # optional
output: {}   # optional
```

### Experiment and Federation

```yaml
experiment:
  task: classification
  repeat_count: 1
  seed: 42

federation:
  rounds: 200
  clients:
    count: 10
    availability:
      strategy: probabilistic
      distribution: gaussian
      gaussian_mean: 0.8
      gaussian_std: 0.1
      chi_square_k: 2
      uniform_multiplier: 2
      minimum: 1.0
      maximum: 1.0
  network:
    topology: mesh_full
    reshuffle_each_round: false
    reliability:
      mean: 0.8
      std: 0.1
      minimum: 1.0
      maximum: 1.0
```

Client ids are generated internally as `device_0`, `device_1`, and so on. There is no configurable id prefix.

### Dataset, Model, and Training

Dataset settings separate non-IID partitioning, evaluation sample counts, and dataloader batch sizes. Model initialization uses `experiment.seed`; other random processes intentionally retain their normal runtime state. CIFAR100 validation uses the training-split indices with deterministic resize/normalization, while training retains random crop/flip augmentation.

Dataset class/count checks are applied once by the shared splitter after a
dataset is downloaded. A separate validation-transform view must preserve the
training split's length and indices. Label-distribution plots read labels
directly through nested `Subset`/`ConcatDataset` indices, so generating the plot
does not decode images, execute augmentation, consume DataLoader shuffles, or
repeat a full CIFAR100 transform pass. Dirichlet and uniform client allocations
use integer apportionment, so every client receives exactly
`samples_per_client` samples even when the proportions do not divide evenly.
Test and validation allocations likewise preserve their requested totals when
every class has enough source samples.

```yaml
dataset:
  name: cifar10
  partition:
    samples_per_client: 610
    labels_per_client: 5
    label_assignment: {strategy: loop, loop_step: 3}
    sample_assignment: {strategy: dirichlet, dirichlet_alpha: 100}
  evaluation: {test_samples: 2010, validation_samples: 2010}
  batches: {train_size: 50, test_size: 100, validation_size: 100}

model:
  name: MobileNetV1Small
  input_size: 3
  num_classes: 10

training:
  optimizer:
    name: adam_cross_round
    learning_rate: 0.01
    weight_decay: 0.0005
  local:
    epochs_per_round: 3
    max_batches_per_epoch: 3
  loss: cross_entropy
```

### Normalization model variants

Model names without a normalization suffix preserve their original BatchNorm
behavior. Every registered model that uses BatchNorm now has matching
GroupNorm and LayerNorm variants:

| BatchNorm model | GroupNorm model | LayerNorm model |
| --- | --- | --- |
| `MobileNetV1Small` | `MobileNetV1SmallGroupNorm` | `MobileNetV1SmallLayerNorm` |
| `MobileNetV2Small` | `MobileNetV2SmallGroupNorm` | `MobileNetV2SmallLayerNorm` |
| `MobileNetV2Baseline` | `MobileNetV2BaselineGroupNorm` | `MobileNetV2BaselineLayerNorm` |
| `MobileNetV2Alpha035` | `MobileNetV2Alpha035GroupNorm` | `MobileNetV2Alpha035LayerNorm` |
| `MobileNetV4Small` | `MobileNetV4SmallGroupNorm` | `MobileNetV4SmallLayerNorm` |
| `FOMOMNv2Baseline` | `FOMOMNv2BaselineGroupNorm` | `FOMOMNv2BaselineLayerNorm` |
| `FOMOMNv2Alpha035` | `FOMOMNv2Alpha035GroupNorm` | `FOMOMNv2Alpha035LayerNorm` |
| `uYOLO` | `uYOLOGroupNorm` | `uYOLOLayerNorm` |

GroupNorm uses the project's fixed eight groups. LayerNorm normalizes the
channel vector independently at each spatial position; the dense uYOLO head
uses standard feature-wise LayerNorm. These variants retain affine parameters
but have no running mean or variance. Consequently, BatchNorm sharing and
refresh settings affect only the unsuffixed BatchNorm models. The existing
`MobileNetV1SmallNoBN` remains the only explicit normalization-free variant.

The optimizer name defines its global-round state lifetime. `adam_cross_round`
keeps Adam's `step`, first moment, and second moment across rounds. `adam_round`
clears those values before each active client's local training while preserving
the optimizer's current learning rate and parameter-group settings. `sgd` is
unchanged. The former `adam` name is intentionally unsupported.

Learning-rate adaptation has no `enabled` flag. Omit it or use an empty list to disable it:

```yaml
training:
  learning_rate_adaptation:
    strategies: [loss, staircase]
```

Local early stop is another optional mapping. Omit it or set it to `null` to disable it. `ceiling` and `plateau` are individually optional, but an enabled mapping must define at least one of them.

```yaml
training:
  local:
    early_stop:
      min_epochs: 1
      ceiling: {metric: accuracy, threshold: 0.98}
      plateau:
        metric: accuracy
        patience: 3
        min_delta: 0.001
        near_best_ratio: 0.995
      record: {scope: round, decay: 1.0}
```

The controller implementation is a reusable utility in `src/utils/early_stop.py`.

## Method Configuration

The method configuration has four distinct concepts:

1. `partition` defines primitive parameter units.
2. Optional `grouping.criterion` only decides which primitive units form a group.
3. `segment_importance` is one continuous semantic quantity propagated primitive unit 鈫?group 鈫?segment.
4. `aggregation.weight` independently weights received/local model values.

No stage can switch to a different importance metric midway through that hierarchy.

```yaml
method:
  name: Gist
  activation_schedule:
    segment_importance_start_round: 1
    aggregation_weight_start_round: 1
    local_update_l2_start_round: 1
  parameters:
    scope: all
    batch_norm:
      include: affine_and_running_stats
      distribution: once_per_recipient
      aggregation_weight: uniform
  partition:
    unit: kernel
    channel_size: 0
    bias: separate
    refinement:
      targets: [linear, pointwise]
      linear_chunk_size: 8
      pointwise_chunk_size: 8
  grouping:
    strategy: within_layer_by_criterion
    arrangement: sensitivity_aligned
    size: 3
    criterion:
      metric: gradient_magnitude.round_step_ema
      parameter_to_unit: l2
  segment_importance:
    metric: fisher_taylor_current.cross_round_step_ema
    reductions:
      parameter_to_unit: l2
      unit_to_group: l2
      group_to_segment: l2
```

`activation_schedule` accepts either positive whole, 1-based global-round values
or decimals strictly between `0` and `1`. A decimal ratio is resolved as
`ceil(training.rounds * value)`. Values such as `1` and `1.0` both mean absolute
round 1; `0`/`0.0` and non-whole values greater than 1 are invalid. The block and
every field are optional and default to round `1`, preserving immediate use.
`segment_importance_start_round` delays the influence of segment-importance
scores: before that round, importance construction becomes random-per-device
construction and probabilistic selection becomes uniform selection. The underlying cross-round
score buffers continue to update, so their first active round uses warmed state.
`aggregation_weight_start_round` uses uniform aggregation before its start and
skips validation-accuracy evaluation plus Fisher projection/transmission when they
are not yet needed. `local_update_l2_start_round` skips both the pre-training
snapshot and post-training primitive-unit projection before its start. These
settings only delay configured functionality; use the existing metric/strategy/
mode fields to disable it. A start round greater than `training.rounds` means the
feature never affects that run. Non-default values are included in the method
abbreviation and `run_metadata.yml`.

`partition.refinement` and `grouping` are optional mappings, not features with separate `enabled` flags. Omit either mapping or set it to `null` to disable it.

`partition.unit` accepts `parameter`, `kernel`, `channel`, or `layer`.
With `layer`, every module that directly owns trainable parameters is one
indivisible primitive unit. A following BatchNorm, GroupNorm, or LayerNorm
module is attached to the previous parameter-owning layer instead of becoming
an independent unit. Parameterless activation, pooling, and dropout modules do
not create units.

For `kernel` and `channel`, each GroupNorm/LayerNorm module contributes exactly
one affine parameter block containing its complete `weight` and `bias`; it is
not split or duplicated per convolution block. GroupNorm computational groups
do not define communication blocks. For `parameter`, these affine values remain
ordinary trainable parameters.

For an attached BatchNorm, layer importance always uses trainable tensors only:
the preceding layer parameters plus BN affine `weight`/`bias`. BN
`running_mean` and `running_var` never enter magnitude, gradient, Fisher,
Lipschitz, or EMA score reductions. When
`batch_norm.distribution: as_base_unit` is used, the selected layer payload
still follows `batch_norm.include`: `affine` sends only affine parameters,
while `affine_and_running_stats` also sends the running statistics.
`channel_size`, `bias`, `refinement`, and within-layer `grouping` are
invalid for `partition.unit: layer`; a layer unit is never split implicitly.

`grouping.arrangement` defaults to `sensitivity_aligned`. Within each layer,
`sensitivity_aligned` sorts primitive units by the configured criterion and
groups adjacent values. `sensitivity_diverse` uses the same criterion but
alternates the lowest and highest remaining values before forming fixed-size
groups. Both are deterministic for equal scores because the primitive-unit
index is the tie breaker. Run-directory names include `sensalign` or `sensdiv`,
so otherwise identical grouping experiments do not share a log directory.

The configuration keeps the parameter-to-unit, unit-to-group, and
group-to-segment reductions explicit, while the runtime evaluates large group
sets with a cached block-to-parameter layout, packed tensor reductions, and
payload-prefix searches. A
criterion score is reused as segment-importance unit scores only when the
metric, parameter-to-unit reduction, and Lipschitz input mode are all identical;
different grouping and importance signals remain independent.

### Metric Lifetime Names

Temporal behavior belongs to the metric name:

- `round_step_ema`: update after optimizer steps and clear at the beginning of every federation round.
- `cross_round_step_ema`: update after optimizer steps and retain state across rounds.
- `round_sample_mean`: compute a sample-weighted mean over the current round's complete local training data.
- `round_estimate`: compute one current-round estimator after local training.
- `current`: use current model values without a temporal buffer.

Consequently there is no public score-buffer beta, reset flag, or 鈥淓MA unit鈥?field.
Grouping and segment importance may independently select the round or cross-round
gradient-magnitude EMA; the runtime keeps separate buffers when both lifetimes are
used in one experiment. Other same-quantity EMA metrics must still select the same
lifetime until they have independent state buffers; the parser rejects unsupported
combinations instead of silently sharing an incorrectly reset buffer.

Supported parameter-derived segment/group metrics are:

- `parameter_magnitude.current`
- `parameter_magnitude.{round_step_ema,cross_round_step_ema}`
- `gradient_magnitude.{round_step_ema,cross_round_step_ema,round_sample_mean}`
- `fisher_diagonal.{round_step_ema,cross_round_step_ema}`
- `gradient_weight_magnitude.round_sample_mean`
- `gradient_signal_preservation.round`
- `empirical_fisher_diagonal.round_sample_mean`
- `empirical_fisher_weighted.round_sample_mean`
- `hutchinson_diagonal.round_estimate`
- `hutchinson_weighted.round_estimate`
- `taylor_first.{round_step_ema,cross_round_step_ema}`
- `taylor_first_current.{round_step_ema,cross_round_step_ema}`
- `taylor_second.{round_step_ema,cross_round_step_ema}`
- `fisher_taylor_current.{round_step_ema,cross_round_step_ema}`
- `hessian_taylor_exact.round`
- `hessian_taylor.{round_step_ema,cross_round_step_ema}`
- `hessian_taylor_current.{round_step_ema,cross_round_step_ema}`

Supported direct block metrics are:

- `block_magnitude.mean_absolute`
- `block_magnitude.root_mean_square`
- `block_lipschitz.current`
- `block_lipschitz.cross_round_ema`
- `fisher_lipschitz_interaction.{round_step_ema,cross_round_step_ema}`

`parameter_to_unit` supports `mean_abs`, `l2`, and `rms`. `unit_to_group` and `group_to_segment` support `mean`, `sum`, `max`, `l2`, and `rms`.

### Segments and Exchange

```yaml
method:
  segments:
    count: 5
    construction: importance_balanced_by_payload
    selection:
      strategy: importance_weighted
      exponential_normalization: true
      exponential_base: 10.0
  exchange:
    sends_per_client: 10
    recipient_selection:
      strategy: balanced_unique_probabilistic
      balance_strength: 5.0
    receive_queue:
      aggregation_threshold: 1
      capacity: 50
```

Segment construction choices are `fixed`, `reshuffle_once`, `reshuffle_each_round`, and `importance_balanced_by_payload`. Selection choices are `random_uniform` and `importance_weighted`.

For importance-weighted selection, let the normalized importance probability
be `p_i`. With `exponential_normalization: true`, the final probability is
proportional to `exponential_base ** p_i`. `exponential_base` must be greater
than `1` and defaults to `e`, preserving the previous behavior when omitted.
Larger bases concentrate more probability on high-importance segments; setting
`exponential_normalization: false` uses `p_i` directly. Enabled exponential
normalization and its base are included in the run-directory name.

Gist requires importance-balanced construction and importance-weighted selection. DFA requires one fixed segment. SegmentPulling uses `pulls_per_segment` instead of `sends_per_client` and supports random-uniform selection.

Recipient strategies for sender-push methods are `random_with_replacement`,
`random_without_replacement`, `balanced_probabilistic`,
`balanced_unique_probabilistic`, and `balanced_round_robin`.
`balanced_unique_probabilistic` keeps the same cross-round historical-count
weighting as `balanced_probabilistic`, but maintains a separate candidate pool
for every sender/segment within one selection call. A recipient cannot be used
again for that sender/segment until every candidate has been used once; if the
segment occurs more times than the candidate count, a new pool cycle begins.
Recipient strategy and probabilistic balance strength are included in the
run-directory name.

### Aggregation Weight

```yaml
method:
  aggregation:
    weight:
      metric: validation_accuracy
      refresh_batch_norm_from_validation: false
```

Supported aggregation metrics are:

- `uniform`
- `validation_accuracy`
- `fisher_diagonal.{round_step_ema,cross_round_step_ema}`
- `gradient_magnitude.{round_step_ema,cross_round_step_ema}`
- `fisher_lipschitz_interaction.{round_step_ema,cross_round_step_ema,cross_round_ema}`

`refresh_batch_norm_from_validation` only has an effect with
`validation_accuracy` and defaults to `true` for that metric to preserve
historical experiments. It does not modify BN state while calculating
validation-accuracy aggregation weights. After aggregation and immediately
before test-set evaluation, the active model is recalibrated with the shared
validation loader when this option is enabled. For every other aggregation
metric, an explicitly supplied value is accepted but normalized to `false`, and
the refresh path does not access a validation loader.
The run-directory abbreviation includes `val_bn_stats_update_on` or
`val_bn_stats_update_off`, so the two experiment variants cannot be confused by
their log names.
`granularity` and `block_reduction` can be omitted for `uniform` and
`validation_accuracy`. With `granularity: block`, the aggregation score unit is
always the primitive unit produced by `method.partition.unit`: parameter
partitioning sends one score per selected parameter, while kernel, channel, and
layer partitioning reduce and send one score per selected primitive block.
There is no independent aggregation `block_unit`. Block Fisher-Lipschitz
aggregation requires kernel, channel, or layer partitioning.

Fisher block aggregation optionally supports per-device, per-layer L2
normalization:

```yaml
method:
  partition:
    unit: kernel
  aggregation:
    weight:
      metric: fisher_diagonal.cross_round_step_ema
      granularity: block
      block_reduction: l2
      normalization: device_layer_l2
```

For device `i`, layer `l`, and unit `b`, the transmitted aggregation weight is
`||F_i,l,b||_2 / (||F_i,l||_2 + eps)`. The denominator uses every primitive
unit in the complete layer before segment selection, so it does not vary with
the transmitted subset. This removes device-layer scale differences while
preserving all within-device unit ratios. A layer whose Fisher scores are all
zero falls back to the unit-L2-normalized uniform vector `1 / sqrt(unit_count)`.
`device_layer_l2` is accepted only for `fisher_diagonal` metrics with block
granularity, kernel/channel partitioning, and `block_reduction: l2`. Omit
`normalization` or use `none` for the original raw
Fisher behavior. Non-default normalization is included in the run-directory
name.

### Aggregation-Weight Exponential Reprocessing

Aggregation weights can use the same exponential probability reprocessing as
importance-weighted segment selection:

~~~yaml
method:
  aggregation:
    weight:
      metric: fisher_diagonal.cross_round_step_ema
      exponential_normalization: true
      exponential_base: 10.0
~~~

For the contributors that actually provide a given parameter or primitive
unit, raw nonnegative weights a_i are first normalized to p_i. The effective
weights are q_i = base ** p_i / sum(base ** p_j), and q_i are used for the
weighted parameter average. The transformation is performed independently for
every parameter's actual contributor set, so sparse segment delivery is
handled correctly. Scalar validation-accuracy weights, parameter Fisher
weights, block Fisher weights, and Centralized aggregation use the same
implementation.

The stable implementation subtracts the largest normalized weight before
exponentiation, preventing overflow for large finite bases. Negative and
non-finite weights are rejected. Only an exactly zero weight sum triggers the
fallback; arbitrarily small positive weights retain their ratios. When every
raw weight for a parameter is zero, enabled exponential processing gives each
present contributor equal weight; when disabled, the existing aggregation
fallback remains unchanged. Uniform input weights remain uniform. Centralized
validation accuracy may use one scalar per model, while Centralized Fisher
weights must exactly match the aggregated parameter-vector length. Centralized
uniform parameters, separately averaged BN state, and aggregation results reject
non-finite values instead of propagating them to every model.

The base must be finite and greater than 1. The setting is independent of the
segment-selection exponential base and adds
`agg_weight_expb<base>` to the run-directory name.
### Post-Local-Training Primitive-Unit Update L2 Bound

Local model drift can optionally be limited after every active client's local
training and before post-training scores, privacy preparation, communication,
and aggregation:

~~~yaml
method:
  partition:
    unit: kernel
    bias: separate
  local_update:
    unit_l2:
      mode: bounded
      multiplier: 0.1
~~~

For active client i and primitive unit b, let w_pre be the value snapshotted
immediately before local training and w_local be the raw locally trained value.
The constrained update is:

    delta = w_local - w_pre
    if ||w_pre||_2 < 1e-5:
        w_out = w_local
    else:
        upper = multiplier * ||w_pre||_2
        scale = min(1, upper / (||delta||_2 + eps))
        w_out = w_pre + scale * delta

This is an upper-bound projection: it only shortens an update that is too
large, never enlarges a smaller update. The multiplier must be finite and
strictly positive, so values below 1 such as 0.1 are valid. The run-directory
name includes 'local_unit_l2_bound_<multiplier>' when enabled. Its use can be
delayed independently with
`method.activation_schedule.local_update_l2_start_round`.

Norms and scaling include only trainable, non-bias parameter values. Bias,
frozen parameters, and buffers retain their raw post-training values. A
primitive unit whose constrained pre-training L2 norm is strictly below the
fixed `1e-5` threshold is left unchanged; a unit exactly at the threshold
remains eligible for clipping. The threshold is applied independently to each
primitive unit and is not a configuration field. This avoids suppressing
updates when a relative bound has no meaningful parameter scale. Pre-training
values, post-training values, updates, bounds, scale factors, and outputs are
checked for non-finite values.

The feature supports kernel, channel, and layer primitive units in segmented
methods. Parameter partitioning and 'Centralized' are rejected. It is
independent of 'method.aggregation.unit_l2': local-update clipping acts first,
while the aggregation constraint acts later on the received aggregate.
### Post-Aggregation Primitive-Unit Update L2 Bound

The displacement caused specifically by aggregation can be bounded separately
from local training and from the final parameter norm:

~~~yaml
method:
  aggregation:
    update_unit_l2:
      mode: bounded
      multiplier: 2.0
~~~

For receiver i and primitive unit b, let w_pre be its constrained local value
immediately before aggregation and w_raw be the raw weighted aggregate:

    delta_agg = w_raw - w_pre
    upper = multiplier * ||w_pre||_2
    scale = min(1, upper / (||delta_agg||_2 + eps))
    w_post = w_pre + scale * delta_agg

Only updates above the bound are shortened. Trainable, non-bias values define
the norms and are scaled together inside each primitive unit. Bias, frozen
parameters, and buffers retain their raw aggregation result. A zero-norm
reference unit is left unchanged and produces a warning. Inputs, updates,
bounds, scales, and outputs are checked for non-finite values.

The mode supports segmented kernel, channel, and layer units; parameter
partitioning and Centralized are rejected. It adds the configured value as
`agg_update_unit_l2_bound_<multiplier>` to the run-directory name.

`aggregation.update_unit_l2` and `aggregation.unit_l2` constrain different
quantities and never alter each other's multipliers. The former bounds
`||w_post - w_pre||_2`; the latter bounds or restores `||w_post||_2`.
For equal-norm parameters, an update multiplier of 2.0 therefore permits a
complete direction reversal while the parameter norm remains unchanged.

When both controls are enabled, the update bound is applied first and the final
norm constraint second. The result is then checked again against the update
bound. If the norm projection caused a primitive unit to exceed that bound,
only its constrained trainable, non-bias values fall back to their
pre-aggregation values. This conflict resolution guarantees both constraints
without coupling their configured multipliers; bias and frozen values retain
their raw aggregation result.

### Post-Aggregation Primitive-Unit L2 Constraints

Aggregation can optionally constrain every primitive unit relative to the
receiving device's own pre-aggregation L2 norm:

~~~yaml
method:
  partition:
    unit: kernel
    bias: separate
  aggregation:
    unit_l2:
      mode: bounded
      multiplier: 1.2
    weight:
      metric: validation_accuracy
~~~

The supported modes are:

- `none`: keep the raw aggregate.
- `exact`: restore the original pre-aggregation norm.
- `bounded`: keep the raw norm when it is inside a multiplicative interval
  and rescale it to the nearest boundary otherwise.

For receiving device i and primitive unit b, let x_pre be the trainable,
non-bias part of the locally outgoing unit recorded before aggregation, x_raw
be the corresponding raw aggregate, r = ||x_pre||_2, and a = ||x_raw||_2.
Exact mode uses target norm t = r. Bounded mode with multiplier m >= 1 uses:

    lower = r / m
    upper = r * m
    t = clamp(a, lower, upper)
    x_post = x_raw * (t / a)

Thus bounded mode leaves an aggregate unchanged while its norm is in range.
A multiplier of 1 is equivalent to exact mode. If the raw aggregate norm is
zero but the target norm is nonzero, its direction is undefined and the local
pre-aggregation constrained values are restored. A zero reference norm
produces a zero target.

Bias values, frozen parameters, and non-parameter buffers are excluded from
both the norm and rescaling, so they retain their ordinary raw aggregation
result. This includes BatchNorm bias and running statistics. Separately
transmitted BatchNorm payloads are not rescaled; when BatchNorm is configured
as a base unit, only a trainable BatchNorm weight participates.

Non-none modes are supported by segmented methods with kernel, channel, or
layer primitive units. Parameter partitioning is rejected because exact mode
would fix every scalar magnitude, and Centralized is rejected because
client-specific reference norms would break its shared-model semantics. All
kernel/channel bias layouts remain supported because bias values do not
participate.

The bounded multiplier must be finite and at least 1. Packed inputs, reference
norms, bounds, target norms, fallback values, scale factors, and restored
outputs are checked for non-finite values. Exact mode adds
`agg_unit_l2_preserve` to the run-directory name; bounded mode adds
`agg_unit_l2_bound_<multiplier>`.

The legacy `preserve_unit_l2: true|false` key remains readable for existing
external configurations and maps to `exact|none`, but it cannot be combined
with the new `unit_l2` section.

### Optional Estimators and Probes

Hutchinson controls are present only when needed:

```yaml
method:
  estimators:
    hutchinson:
      probe_vectors: 5
      batch_limit: 0
```

`batch_limit: 0` uses all local training batches.

Exact diagonal-Hessian methods perform one second-order backward per trainable
parameter and are rejected above 100,000 trainable parameters. Use a Hutchinson
diagonal metric for larger models. If vectorized per-sample gradients are not
supported by an operation, the fallback sample loop is logged once.

### Optional Differential Privacy

The standalone DP package supports communication-level `model_update` and
local sample-level `local_dp_sgd` modes:

```yaml
differential_privacy:
  mode: model_update
  clipping_norm: 1.0
  noise_multiplier: 1.2
  delta: 1.0e-5
```

Omit the section or set it to `null` to disable DP. There is no separate
`enabled` switch. All four fields are required when the mapping is present.
`clipping_norm` is C. `model_update` uses replace-one device-update
adjacency, so its L2 sensitivity is `2C` and its Gaussian standard deviation
is `2 * C * noise_multiplier`. `local_dp_sgd` uses add/remove sample
adjacency within each device, so its per-step sensitivity is C and noise is
added to the clipped gradient sum with standard deviation
`C * noise_multiplier` before division by the expected batch size.

At the start of a participating client's round, the controller snapshots all
trainable named parameters as one model-wide vector. After local training it
computes one update, clips its whole-model L2 norm to C, adds independent
Gaussian noise to every trainable parameter, and creates one private outgoing
model. That same release is reused for all packets and recipients in the round;
splitting it into segments does not add accounting steps. The client's own
aggregation contribution also uses this private view, and the aggregation
result is written back to its training model. With DP disabled, the outgoing
view is the original model, so existing FL behavior is unchanged.

The first-round update baseline is the data-independent initialized model.
After that, each baseline is the previous private aggregation result that was
written back to the client's model; an unprotected locally trained model is
never used as the next round's baseline.

With `mode: local_dp_sgd`, every local optimizer step computes per-sample
gradients, clips each sample using one L2 norm across all trainable model
parameters, sums the clipped gradients, adds Gaussian noise, divides by the
configured expected batch size, and executes SGD. It reuses
`training.local.epochs_per_round`, `training.local.max_batches_per_epoch`, and
`dataset.batches.train_size`; no DP-specific schedule setting exists. For
device d with Nd local samples and configured batch size B, Poisson sampling
uses `q_d = min(B, N_d) / N_d`. Each epoch draws exactly
`max_batches_per_epoch` fresh, independent Poisson batches, so an active round
executes exactly `epochs_per_round * max_batches_per_epoch` private optimizer
steps. An empty Poisson batch is a noise-only step and is still composed.

DP-SGD keeps the model in training mode but places BatchNorm modules in eval
mode during private local optimization. BN affine weight and bias remain
trainable and pass through the same per-sample clipping and noise mechanism.
Running mean and variance are not updated from private local batches, and
validation scoring does not refresh them. After aggregation, enabling
`refresh_batch_norm_from_validation` recalibrates the active model's buffers
from the shared public validation set immediately before test-set evaluation;
when disabled, testing uses the running state already present after aggregation.

`MobileNetV1SmallNoBN` is registered for experiments that must exclude BatchNorm
entirely. It preserves the MNv1 topology, replaces all BatchNorm operations with
identities, and enables the preceding convolution biases.
`MobileNetV1SmallGroupNorm` instead uses eight-group GroupNorm, while
`MobileNetV1SmallLayerNorm` applies channel-wise LayerNorm to NCHW feature maps.
Both variants keep affine parameters, have no running statistics, and therefore
use the same normalization behavior in training and evaluation modes.

`local_dp_sgd` currently additionally requires classification, the SGD
optimizer, disabled local early stop, and an empty
`training.learning_rate_adaptation.strategies` list. Data-dependent early
stopping or LR adaptation is rejected because it would make the number or
parameters of private mechanisms depend on unprotected local metrics.
Fisher block-statistics metrics are also rejected in this mode because the
private optimizer does not populate the ordinary `GradientBuffer`; model-only
`mean_abs`, `rms`, and `lipschitz` probe metrics remain available.

Both modes deliberately accept only the communication combinations covered by
their release:

- `method.parameters.scope: all`;
- `method.partition.unit: parameter`;
- `method.parameters.batch_norm.include: affine`;
- `method.aggregation.weight.metric: validation_accuracy`;
- importance construction, when used, must use
  `parameter_magnitude.current`.

For `model_update`, validation accuracy is computed from the private outgoing
model on the shared public validation set, so it is post-processing of the same
release and does not add privacy cost. BN `running_mean`, `running_var`, and
`num_batches_tracked` are neither copied from private local training into the
outgoing validation model nor transmitted. The private validation copy keeps
its last safe buffers while its score is calculated; trainable BN affine
parameters remain part of the clipped/noised model update. Only `model_update`
uses this separate private validation model; non-DP and `local_dp_sgd`
validation, plus every post-aggregation test, evaluate the active model
directly. After aggregation, `refresh_batch_norm_from_validation: true`
recalibrates the active model from public validation data immediately before
test-set evaluation.

The hard-coded RDP orders live in `src/differential_privacy/accountant.py`.
`model_update` accounts one replace-one Gaussian release per participating
device and round using `RDP(alpha) = T * alpha / (2 * sigma^2)`: both its
sensitivity and actual noise standard deviation contain the same `2C` factor,
so sigma remains the mechanism's noise multiplier. `local_dp_sgd` uses
integer-order Poisson-sampled Gaussian RDP under add/remove sample adjacency
and composes every optimizer step independently for every device. Reusing one
resulting model for validation, segmentation, packets, and multiple recipients
is post-processing and does not add steps.
Both accountants report the minimum over orders of
`RDP(alpha) + log(1 / delta) / (alpha - 1)`.

For `local_dp_sgd`, the integer-order sampled-Gaussian term used by the code is:

\[
A_\alpha=\sum_{j=0}^{\alpha}{\alpha\choose j}
(1-q)^{\alpha-j}q^j
\exp\left(\frac{j(j-1)}{2\sigma^2}\right),
\qquad
R_\alpha^{step}=\frac{\log A_\alpha}{\alpha-1}.
\]

For T private optimizer steps, `RDP(alpha) = T * R_alpha_step`. The
reported epsilon is
`min_alpha(RDP(alpha) + log(1 / delta) / (alpha - 1))`, and
`optimal_alpha` is the minimizing order. Consequently, `rdp_alpha_*`
columns contain RDP values, not per-order epsilon values.

Every DP run logs a planned per-device privacy upper bound at startup,
including q, mechanisms per participation, total configured steps/releases,
epsilon, delta, and optimal alpha. The plan assumes that the device
participates in every configured round. `run_metadata.yml` stores the same
fields under `differential_privacy.planned_privacy`. At shutdown, the log and
`differential_privacy.actual_privacy` report the range of actually composed
per-device epsilon values and the worst device; random availability can make
this smaller than the plan.
Prepared experiment sweeps are separated by purpose: `differential_privacy_epsilon/` varies epsilon at fixed delta, while `differential_privacy_delta/` contains epsilon 2/4 crossed with delta `1e-5`, `1e-4`, `1e-3`, and `1e-2` for both datasets and both DP modes. The matching sweeps are split into `differential_privacy_mnv1_no_bn_sgd_dp_epsilon/` and `differential_privacy_mnv1_no_bn_sgd_dp_delta/`. Their `local_dp_sgd` configurations cover CIFAR-10 `MobileNetV1SmallNoBN`, `MobileNetV1SmallGroupNorm`, and `MobileNetV1SmallLayerNorm` plus FashionMNIST `LeNet5`; the independently calibrated per-sample-gradient P50 clipping norms are respectively `1.6040298342704773`, `11.136213302612305`, `12.319522857666016`, and `2.1551175117492676`. The same two folders also contain 50-round `model_update` sweeps for MNIST `FCN` and FashionMNIST `LeNet5`. Their whole-client-update P50 clipping norms are `0.6816111505031586` and `2.115228295326233`, respectively, and every noise multiplier is recalibrated for 50 private releases rather than copied from the 200-round sweeps. Per-sample-gradient and whole-client-update clipping norms measure different objects and should not be compared directly. The inert affine BN mode in models without BatchNorm is retained only to satisfy the existing strict DP configuration contract.

Standalone command-line tools live in `src/tools`. Preview the final per-device
optimal alpha and epsilon without running training:

```bash
python -m src.tools.privacy_cost_estimator --config-file src/config_folders/DP_comp/05_fcn_mnist_dp_sgd_8_1e-5.yml
```

By default the estimator assumes that one device participates in every
configured global round. This is a per-run upper bound when client availability
is random. If the active-round count is already known, use `--participations`.
Given a target epsilon and delta, solve the smallest compatible noise multiplier:

```bash
python -m src.tools.privacy_cost_estimator --config-file CONFIG.yml --participations 120
python -m src.tools.privacy_cost_estimator --config-file CONFIG.yml --target-epsilon 8 --delta 1e-5
```

For `model_update`, the output includes both the optimal alpha from the exact
hard-coded order set used by the runtime accountant and the continuous analytic
Gaussian optimum. For `local_dp_sgd`, it derives the sample rate and total step
count from the existing dataset/training settings and selects the optimal supported
integer alpha numerically. Estimates are for one experiment run and do not
compose `experiment.repeat_count` runs.

Clipping norm C cannot be inferred from epsilon and delta. The automatic
calibrator loads the configured model and non-IID dataset, performs short
non-private local training, measures the exact whole-trainable-model norm used
by the selected DP mode, and writes both raw samples and a YAML summary:

```bash
python -m src.tools.clipping_norm_calibrator --config-file CONFIG.yml
```

By default it calibrates all configured devices for one local round. For
`local_dp_sgd`, it measures up to eight samples from each full training
batch while still using the full batch for the ordinary warm-up update.
`--calibration-rounds`, `--samples-per-batch`, `--max-devices`,
`--device`, and `--pre-data-path` control an explicit calibration run
without adding experiment-YAML settings. Progress is printed after each
device round. The default 1800-second limit can be changed with
`--time-limit-seconds`; an interrupted run writes collected samples with
`status: partial`.

The output directory contains `clipping_norm_samples.csv`,
`clipping_norm_summary.yml`, a config snapshot, and the generated allocation
visualization. `--pre-data-path` can reuse an externally stored complete
split. The summary reports pooled and
per-device distributions, P50/P75/P80/P90/P95 candidates, clipping fractions,
mean retained ratios, clipping residuals, mechanism noise, post-average noise,
and expected full-parameter noise L2. P80 is labeled as a starting candidate,
not a mathematically optimal C.

For already collected norms, the lighter selector accepts a CSV containing an
`l2_norm` column; device and round columns are optional:

```csv
device_id,round,l2_norm
device_0,1,0.183
device_1,1,0.427
```

For `model_update`, each row must be the whole-trainable-model L2 norm of one
client update before clipping. Use the same model, optimizer-state semantics,
learning rate, local epochs, batches, and non-IID partition pattern as the
target experiment. Collect at least 100 client-round norms across early, middle,
and late training states rather than only at initialization.

For `local_dp_sgd`, each row must instead be one sample's whole-model gradient
L2 norm before clipping. Collect representative samples across clients, labels,
and multiple model checkpoints. A batch-average gradient norm is not a valid
replacement for a per-sample norm.

```bash
python -m src.tools.clipping_norm_selector --config-file CONFIG.yml --norm-file calibration_norms.csv
```

The selector reports the same P50/P75/P80/P90/P95 candidates, observed
clipping fractions, retained ratios, clipping residuals, and mechanism noise.
P80 is only a consistent starting point; final utility experiments should
compare nearby candidates.
Prefer a public or independent calibration dataset from the same task,
partitioned to reproduce the intended non-IID structure. The shared IID
validation set alone is usually not representative of private local updates.
If actual private training data is used, that data-dependent choice of C is not
covered by the current accountant and must be reported as non-private calibration.


`privacy_accounting.csv` contains mode, adjacency relation, L2 sensitivity,
total release/step count, steps in the current round, sample rate, expected
batch size, local dataset size, public mechanism parameters, every RDP value,
epsilon, and `optimal_alpha`. Every candidate remains available in its own
`rdp_alpha_*` column. The sampling fields are blank for
`model_update`. The file intentionally does not contain raw updates,
pre-clipping norms, clip factors, or RNG seeds.

This is a research simulator, not a production DP system: PyTorch's generator
is used for Gaussian sampling and is independently seeded, but is not a
cryptographic random-number generator. The privacy claim covers outgoing
trainable model communication under the configured adjacency: replace-one
device-update adjacency for `model_update` and add/remove sample adjacency
within each device for `local_dp_sgd`. Raw local training metrics and optional
diagnostic/probe files remain private
research outputs and are not made DP by this feature.

The optional importance-correlation probe evaluates how closely reference scores
calculated at the configured stage reproduce a parameter-level comparison baseline. It requires
`method.partition.unit: parameter`; `null` or omission disables it.

```yaml
probes:
  importance_correlation:
    stage: after_training
    comparison_baseline: method
    evaluation_rounds: all
    reference_scores:
      gradient_magnitude.round_sample_mean: true
      gradient_weight_magnitude.round_sample_mean: true
      empirical_fisher_diagonal.round_sample_mean: true
      empirical_fisher_weighted.round_sample_mean: true
      hutchinson_diagonal.round_estimate: true
      hutchinson_weighted.round_estimate: true
    measurements:
      spearman: true
      top_k_overlap: true
      top_k_jaccard: true
      top_k: [0.05, 0.1, 0.2, 0.3]
    hutchinson:
      probe_vectors: 5
      batch_limit: 0
```

`stage` selects whether the probe calculates its scores immediately after local
training or immediately after aggregation. `comparison_baseline: method`
resolves to the canonical metric configured at
`method.segment_importance.metric`; an explicit canonical parameter metric is
also accepted. `evaluation_rounds` accepts `all` or a list of 1-based round
numbers. Reference scores and measurements are individually selectable.
Absolute-gradient and empirical-Fisher values share one per-sample gradient
pass. The probe calculates each measurement over the complete selected
parameter vector and writes one CSV row per device/reference pair. Each selected
measurement has its own column; top-k fractions are encoded in names such as
`top_k_overlap_0.1`. It never writes raw
scores, ranks, or coordinate sets, and its scores are not passed to segment
selection or aggregation.

## Outputs

Each run writes a directory under `log/`. The directory name keeps the leading config identity, then adds compact sweep fields (`ll2`, `al2`, `eps`, and `delta`), the FL method, `vbn/novbn` state, `dpupd/nodp` state, optimizer lifetime, and a hash of the source config filename, complete parallel run label, and complete canonical config. Including the original filename prevents differently named experiment files with identical YAML content from sharing a directory name, including when the parallel runner loads a temporary rewritten config. Including the full run label keeps same-seed repeats distinct even if shortening removes their readable repeat index. Parallel labels such as `seed_index000_repeat002_seed44` are shortened to `si000_r002_s44`. Remaining overlength names are shortened at underscore-delimited boundaries; full names and settings stay in metadata. The name is capped at 96 UTF-8 bytes for Windows path safety; with the repository's current 77-character absolute `log` root and a 49-character statistics filename, this keeps the complete path near 224 characters. Full method details remain in `run_metadata.yml` and `used_config.yml`.

`segment_count` is an upper bound: when fewer non-empty blocks or groups exist, the runtime creates only the available non-empty segments. CUDA-backed DataLoaders automatically use pinned host memory while the runner-controlled worker count is preserved.

Common files include:

| File | Purpose |
| --- | --- |
| `used_config.yml` | Exact schema-v2 YAML snapshot used for the run. |
| `run_metadata.yml` | Source path, model-initialization seed, DataLoader workers, aggregation-device policy, validation-BN behavior, directory-name hash, and timing summary. |
| `wall_clock.csv` | Per-round stage timings. |
| `fl_simulation.log` | Runtime log. |
| `topology_shape_<topology>.svg` | Generated network topology. |
| `metrics.xlsx` | Training and evaluation metrics; CSV sheets are used as a fallback when XlsxWriter is unavailable. |
| `communication_packets.csv` | Default per-packet ledger with source, destination, delivery status, contents, and payload size. |
| `privacy_accounting.csv` | Present only with DP enabled; per-device release/step counts, sampling fields, every `rdp_alpha_*` value, epsilon, delta, and `optimal_alpha`. |
| `importance_correlation.csv` | Present only when the importance-correlation probe is enabled; contains measurement values only. |

Pre-aggregation validation sheets use names such as
`validation_accuracy_pre_agg`; post-aggregation test sheets use names such as
`test_accuracy_post_agg`. Validation values are not labeled as test metrics.

The metrics workbook is checkpointed every 10 rounds and once more at normal
completion when the final round is not a checkpoint. A best-effort snapshot is
also written after an error. Per-round results remain in memory between
checkpoints, avoiding a complete workbook rewrite after every round.

For newly generated splits, `data_dis.svg` is written beside the generated
dataset split directory under `dataset/`, rather than inside the run log
directory. Loading a prepared split is transactional: a missing/corrupt client,
test, or validation file raises an error without leaving a partially populated
splitter state.

Communication recording is always enabled and requires no config switch. Each
selected transfer is one row. Dropped packets count toward sender cost but not
receiver-delivered cost. Push transfers use sender 鈫?selected recipient;
SegmentPulling records selected source 鈫?requesting device; Centralized uses the
virtual endpoint `coordinator` for separate uploads and downloads.
Segmented push and pull share the same transfer-accounting path, so delivery,
BN retry, and payload-size rules stay identical in both directions.

Packet events are recorded when the communication decision is made. For
`once_per_recipient` BN, a dropped attempt retains the BN payload and subsequent
attempts retry it until the first successful delivery; later packets for that
source/recipient pair do not repeat it.

Payloads are classified by transport role as `model_parameters`,
`aggregation_weight`, `batch_norm`, and `bitmap`. BN distributed
`as_base_unit` is already part of `model_parameters`; only independently
distributed BN is counted as `batch_norm`. Validation metrics, Fisher weights,
and Fisher-Lipschitz weights are all `aggregation_weight`, with the exact metric
retained in `aggregation_weight_metric`. Every numeric element is costed as
float32 (4 bytes). The CSV stores the logical `bitmap_bits` and its transmitted
storage as `bitmap_bytes = ceil(bitmap_bits / 8)`. Packet identifiers, device
identifiers, round numbers, and status fields are ledger metadata and are not
included in `total_bytes`.

Network reliability and probabilistic client availability both accept the
closed interval `[0, 1]`; a zero minimum is valid and represents a link/client
that may never participate. Invalid utility boundaries raise `ValueError`
consistently, including when Python is run with optimizations.

## Validation

Use the active simulation environment described in the Quick Start:

```bash
python framework_runner.py --config-file src/config_folder/01_fcn_mnist.yml --gpus cpu --dry-run
python framework_runner_parallel.py --config-file src/config_folder/01_fcn_mnist.yml --gpus cpu --dry-run
```

Runner dry-runs recursively load the selected config folder without starting training. Full training still requires the corresponding local datasets and available runtime resources.

## Development Notes

- Keep config parsing centralized in `src/sim_tools/simulation_config.py`.
- Keep runnable experiment configs schema-valid and place grouped suites under `src/config_folders/`.
- Preserve CUDA device placement for tensor math and transfer to CPU only for inherently CPU-side metadata or serialization.
- Update this README and the example configs whenever runtime configuration behavior changes.
- Keep `simulation_manager_tool.py`, `data_splitter.py`, `fl_methods/base.py`,
  and `gradient_buffer_tool.py` as orchestration/facade modules. Pure segment
  planning, data allocation, DataLoader construction, per-sample score kernels,
  run-path naming, and wall-clock measurement live in their focused sibling
  modules and should be tested there.
