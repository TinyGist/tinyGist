# `src/utils`

This package contains small simulation helper tools. Runtime modules such as the
simulation manager, config parser, metrics recorder, device helpers, and logging
helpers live in `src/sim_tools`.

## Main Files

| File | Purpose |
| --- | --- |
| `communication_simulator_tool.py` | Builds and samples communication topology/stability. |
| `communication_recorder.py` | Writes the default per-packet communication ledger and float32 payload sizes. |
| `stale_training_tool.py` | Selects trainable devices per round. |
| `adaptive_loss_stair_tool.py` | Adaptive learning-rate helper. |
| `early_stop.py` | Local-training ceiling/plateau early-stop state machine. |
| `object_detection_tools.py` | Object-detection task and metric factory helpers. |
| `__init__.py` | Package-level lazy access for selected `src/sim_tools` runtime objects. |

## Runtime Path

The normal flow is:

1. `framework_runner.py` or `framework_runner_parallel.py` loads a config.
2. `src/sim_tools/simulation_config.py` parses and canonicalizes it.
3. `FederatedLearningSim` in `src/sim_tools/simulation_manager_tool.py` builds runtime objects.
4. Per-round training records optional score buffers from `src/scoring/`.
5. The selected FL method performs communication and aggregation.
6. Metrics and logs are written under `log/`.

Keep config parsing deterministic and centralized in `src/sim_tools/simulation_config.py`; avoid adding ad-hoc config reads in method implementations.
