# P3-06 — Multi-seed Repetition

## Goal

A P3 conclusion must not depend on one lucky initialization or one lucky batch
order.

Every qualifying experiment is repeated with at least:

```text
17 / 29 / 43
```

Additional predeclared seeds are allowed, but these three may not be omitted.

## What must stay identical?

Across seeds, keep fixed:

```text
dataset / task view
feature cache
backbone model_id + exact revision
hidden-state layer
Mean DR architecture
embedding_dim = 64
SCAF K / margin / scale
optimizer
learning rate
weight decay
epoch count
classes_per_batch
samples_per_class
dev metric definition
```

Only the random seed changes.

## What may the seed change?

The seed may change:

```text
Mean DR Linear initialization
SCAF center initialization
balanced-sampler draw/order
other seeded PyTorch / NumPy / Python random operations
```

This is intentional. P3-06 measures sensitivity to those stochastic choices.

## Critical ordering rule

A common reproducibility bug is:

```text
construct model
↓
set seed
↓
train
```

That is too late: the model has already been randomly initialized.

P3-06 enforces:

```text
seed_everything(seed)
↓
construct Mean DR
↓
construct SCAF
↓
construct seed-aware sampler
↓
train
```

`run_training()` seeds again internally for the training phase, but P3-06's
outer seed is what guarantees initialization is controlled.

## Immutable directories

One experiment becomes:

```text
experiment_dir/
├── seed_0017/
│   ├── checkpoints/
│   ├── metrics.jsonl
│   └── checkpoint_hashes.jsonl
├── seed_0029/
├── seed_0043/
└── multiseed_summary.json
```

Existing non-empty seed run directories are never silently overwritten.

## P3-06 does not select checkpoints

This phase repeats training. It does **not** choose `best.pt`.

For monitoring only, `multiseed_summary.json` aggregates the **same final
epoch** across seeds.

Checkpoint selection remains P3-07 and uses only:

```text
generic_dev_score = prototype_macro_f1
```

## Mean ± standard deviation

P3-06 records the final-epoch diagnostic as:

```text
mean ± sample standard deviation
```

with:

```text
ddof = 1
```

The sample standard deviation is:

\[
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}
\]

For the frozen three seeds, `n=3`.

After P3-07 selects one checkpoint per seed, the same aggregation convention
should be used for the selected-run comparison.

## Failed or weak seeds

A normally completed low-scoring seed is **not discarded**.

Examples:

```text
17 -> 0.84
29 -> 0.82
43 -> 0.69
```

The 0.69 result is evidence of instability and must remain in the report.

A run may be excluded only for a documented execution/data failure such as
corrupt cache, NaN failure, OOM, or interrupted/incomplete execution. Such a
rerun must keep the same seed.

## P3-06 Gate

PASS requires:

1. seeds 17 / 29 / 43 are all present and unique;
2. the seed is set before model/SCAF construction;
3. the same training budget is used for all seed runs;
4. each seed gets an independent immutable run directory;
5. all normally completed seeds remain in the summary;
6. aggregate reports mean and sample std (`ddof=1`);
7. P3-06 performs no checkpoint selection.
