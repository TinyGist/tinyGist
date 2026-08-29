# `src/data_getter`

This package owns dataset loading and per-device split generation.

## Responsibilities

- Load raw or prepared datasets from the configured data root.
- Split training data across simulated devices.
- Build train/test/validation dataloaders used by `FederatedLearningSim`.
- Store generated split artifacts under `dataset/` unless a replicated split is used.

## Main Files

| File | Purpose |
| --- | --- |
| `base_getter.py` | Shared dataset-getter behavior. |
| `data_splitter.py` | Label/data allocation helpers. |
| `*_getter.py` | Dataset-specific getter implementations. |
| `*_dataset_preparer.py` | Dataset preparation helpers for larger/raw datasets. |
| `__init__.py` | Registers dataset names in `DATASETS`. |

## Adding a Dataset

1. Implement a getter class in this package.
2. Return per-device training dataloaders plus shared test/validation dataloaders.
3. Register the class in `src/data_getter/__init__.py`.
4. Use the registered key in YAML as `dataset.name`.

Keep default data roots relative to `./data` so `--workspace` mode works without extra path rewriting.
