# `src/sim_tools`

This package contains the core simulation runtime and configuration modules.

## Main Files

| File | Purpose |
| --- | --- |
| `simulation_manager_tool.py` | Builds datasets, models, methods, optimizers, buffers, logs, and runs rounds. |
| `simulation_config.py` | Strictly parses schema-v2 YAML into typed runtime config. |
| `simulation_metrics.py` | Metrics collection and export. |
| `definitions.py` | Shared simulation aliases, canonicalization helpers, slugs, and lazy import registry. |
| `device.py` | Device and CUDA diagnostics. |
| `runtime_env.py` | Runtime cache-directory setup. |
| `logging_utils.py` | Round-aware logging setup. |
| `config_abbreviation.py` | Short labels used in generated run directory names. |

Reusable helper tools for early stop, adaptive LR, communication topology, stale
training, and object-detection metric factories live in `src/utils`. New code should import
core runtime modules from `src.sim_tools`; `from src.utils import ...` remains
available for selected runtime objects through package-level lazy loading.
