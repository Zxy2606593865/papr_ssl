# P3-07 — Dev-only Checkpoint Selection

## Frozen rule

Every checkpoint is compared using only:

```text
generic_dev_score
```

whose P3-05 definition is frozen as:

```text
prototype_macro_f1
```

The selection rule is:

```text
maximize generic_dev_score
```

No other metric can override it.

## Why one metric?

P3-05 records many diagnostics:

```text
prototype_accuracy
prototype_macro_f1
intra_class_cosine_mean
wrong_class_cosine_max_mean
top1_top2_margin_mean
...
```

Those diagnostics help explain the embedding space, but if we are allowed to
choose whichever metric makes a checkpoint look best after training, model
selection becomes subjective.

Therefore P3-07 has exactly one ranking metric.

## Tie break

If two checkpoints have exactly the same `generic_dev_score`:

```text
choose the earliest epoch
```

This is frozen in advance.

The reason is not that earlier is always scientifically better. It is simply a
deterministic predeclared rule that avoids consulting another metric after a
tie.

## Test remains sealed

P3-07 must not receive or inspect any `generic_test` value.

The selector fails closed if its input metrics contain any test-related field.

The normal P3 data path already prevents test construction in P3-04.

## Checkpoint integrity

Before a checkpoint can be selected, P3-07 recomputes its SHA-256 and verifies
it against the hash recorded by P3-05.

A mismatched or missing checkpoint fails selection.

## No `best.pt` copy

P3-07 writes a small immutable pointer:

```text
seed_0017/
└── selected_checkpoint.json
```

It does not create another untracked binary copy named `best.pt`.

The pointer records:

```text
epoch
checkpoint path
checkpoint SHA-256
generic_dev_score
generic_dev_score_name
selection metric
tie-break policy
```

For a multi-seed experiment, one checkpoint is selected independently for each
seed:

```text
seed 17 -> one dev-selected checkpoint
seed 29 -> one dev-selected checkpoint
seed 43 -> one dev-selected checkpoint
```

A top-level `selected_checkpoints.json` can then reference all three.

## P3-07 Gate

PASS requires:

1. only `generic_dev_score` determines ranking;
2. `generic_dev_score_name == prototype_macro_f1`;
3. direction is maximize;
4. exact score ties use earliest epoch;
5. checkpoint SHA-256 is verified before selection;
6. test-related input is rejected;
7. selection creates a pointer JSON, not a duplicate `best.pt`;
8. one checkpoint is selected independently per seed.
