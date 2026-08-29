# `src/experiment_probes`

This package contains optional measurement probes. Probes can request shadow
score calculations and write diagnostics, but their scores are not passed to
the federated-learning method.

## Importance correlation probe

The probe compares the method's parameter-level ranking metric (or another
explicit canonical metric) with independently selectable post-training scores.
It currently requires `method.partition.unit: parameter`.

Enable it with the mapping below. Omit it or set
`importance_correlation: null` to disable it.

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

`stage` accepts `after_training` or `after_aggregation` and defaults to
`after_training`. `comparison_baseline` defaults to `method`, which resolves to
`method.segment_importance.metric`. It may instead be any canonical
parameter-level importance metric. `evaluation_rounds` accepts `all` or a
non-empty list of 1-based round numbers.

The six reference-score keys are their complete public metric names. Missing
keys in an explicit `reference_scores` mapping are disabled; omitting the
whole mapping enables all six. Every reference is calculated at the configured
stage. Absolute-gradient and empirical-Fisher values share one per-sample
gradient pass, while weighted variants reuse the corresponding base estimate
and the parameters at that stage.

The three measurements are independently selectable. `top_k` values are
fractions in `(0, 1]`; the full vector uses `ceil(fraction * parameter_count)`
coordinates. Equal scores are resolved deterministically by coordinate index.

## Output

The output path is fixed and is not configurable:

```text
log/<run_name>/importance_correlation.csv
```

Each completed calculation is appended immediately. Each row contains round,
stage, device, full-vector parameter count, baseline/reference names, and one
column per enabled measurement. Spearman uses `spearman`; top-k columns encode
the fraction, for example `top_k_overlap_0.1`. Each device/reference pair is
measured once over its complete selected parameter vector. Raw scores,
ranks, and selected coordinate sets are never written.
