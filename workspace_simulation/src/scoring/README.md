# `src/scoring`

This package contains scoring helpers and EMA buffers used by segment creation
and score-weighted aggregation.

| File | Purpose |
| --- | --- |
| `definitions.py` | Canonical names and aliases for parameter-wise score methods. |
| `block_grouping.py` | Layer-local gradient-sorted block grouping helpers for kernel/channel segment creation. |
| `parameter_scores.py` | Parameter score vector and threshold helpers for parameter-wise importance segmentation. |
| `gradient_buffer_tool.py` | Buffers for Fisher aggregation, online parameter-wise gradient/Fisher/Taylor/Hessian scores, post-training gradient, Hessian, Hutchinson, and empirical Fisher scores, and block-level interaction scores. |
| `block_score_ema_buffer.py` | EMA buffer for kernel/channel block scores, currently used for Lipschitz selection scores and Fisher-Lipschitz aggregation scores. |
