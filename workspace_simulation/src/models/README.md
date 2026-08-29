# Adding a Model

New trainable models should inherit `FederatedModelMixin` and `torch.nn.Module`.

Minimal pattern:

```python
import torch
from torch import nn
from src.models.parameter_vector import FederatedModelMixin


class MyModel(FederatedModelMixin, nn.Module):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=10):
        super().__init__()
        # Seed CPU parameter construction without changing CUDA runtime RNGs.
        torch.random.default_generator.manual_seed(random_seed + device_num)

        self.features = nn.Sequential(...)
        self.classifier = nn.Linear(..., num_class)
        self.blocks = [self.features, self.classifier]

        self.all_modules = self.blocks
        self.conv_modules = [self.features]
        self.fc_modules = [self.classifier]
        self.custom_modules = []
        self.finalize_model_setup()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

Then register it in `src/models/__init__.py`:

```python
from .my_model import MyModel
MODEL_REGISTRY.register("MyModel", MyModel)
```

The simulation code expects this public interface:

- `get_parameter_vector(parameter_scope, bn_mode)` / `load_parameter_vector(..., parameter_scope, bn_mode)`
- `get_batchnorm_vector(parameter_scope, bn_mode)` / `load_batchnorm_vector(..., parameter_scope, bn_mode)`
- `get_parameter_blocks(parameter_scope, conv_mode, channel_length, include_other_blocks, bn_mode, block_refinement, trainable_only)` / `load_parameter_blocks(...)`
- `model_total_params_num`, `model_fc_params_num`, `model_conv_params_num`, `model_custom_params_num`

Use `parameter_scope` on these canonical methods instead of scope-specific
wrapper names. This keeps vector and block extraction on one validated path.
`conv_mode: layer` returns one flat block for each parameter-owning module and
attaches a following BatchNorm, GroupNorm, or LayerNorm to the preceding layer.
`trainable_only: true` retains the same layer boundaries but removes BN running
statistics and other non-trainable tensors; score paths use this form, while
communication paths retain the configured `bn_mode` payload. GroupNorm and
LayerNorm have affine parameters but no running-state payload.

Supported `parameter_scope` values:

- `all`: all trainable model parameters from `all_modules`
- `conv`: feature/backbone parameters from `conv_modules`
- `fc`: classifier/head parameters from `fc_modules`
- `custom`: model-defined parameters from `custom_modules` or `custom_parameter_modules()`

Supported `bn_mode` values:

- `none`: exclude BatchNorm affine parameters and running stats
- `affine`: include BatchNorm trainable affine parameters (`weight`/`bias`)
- `affine_running_stats`: include affine parameters plus `running_mean`/`running_var`

Block extraction returns `ParameterBlockSet` from `src/models/parameter_vector.py`.

- `conv_blocks`: Conv2d blocks, split by `conv_mode`
- `linear_blocks`: Linear output-neuron blocks
- `bn_blocks`: one flattened block per BatchNorm module, containing tensors selected by `bn_mode`
- `other_blocks`: one flattened affine block for each GroupNorm/LayerNorm module,
  plus direct parameters from other modules when `include_other_blocks=True`

The canonical block payload order is `conv_blocks + linear_blocks + bn_blocks + other_blocks`. Block loading expects a `ParameterBlockSet`; old two-list block pairs are no longer supported.
