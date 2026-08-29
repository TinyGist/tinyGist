# `src/fl_methods`

This package implements federated learning aggregation methods.

## Responsibilities

- Build method instances from parsed config through `method_factory.py`.
- Create parameter or block segments.
- Simulate communication over the current topology.
- Dispose received segments into aggregation payloads.
- Aggregate parameters back into local models.

## Main Files

| File | Purpose |
| --- | --- |
| `base.py` | Shared parameter/vector/block segmentation, cached packed-block layout, and aggregation logic. |
| `centralized_method.py` | Centralized full-model aggregation. |
| `segment_pulling_method.py` | SegmentPulling implementation. |
| `dfa_family_base.py` | Shared DFA/SDFA/Gist communication behavior. |
| `dfa_method.py` | DFA specialization. |
| `sdfa_method.py` | SDFA specialization. |
| `gist_method.py` | Gist specialization. |
| `segment_ops.py` | Tensor and block segment helpers. |
| `method_factory.py` | Builds method classes from `FLMethodConfig`. |

## Extension Notes

New methods should inherit `FLMethods` or `SegmentedMethodBase`, implement communication/disposal behavior, register the class in `src/fl_methods/__init__.py`, and add a builder in `method_factory.py`.

For block-based methods, `method.partition.unit: kernel` and `method.partition.unit: channel` rely on the model block interface from `FederatedModelMixin`.
The runtime preserves that public block interface for probes, but compiles it
to a block-ordered tensor mapping once and uses packed tensor operations during
simulation rounds.
