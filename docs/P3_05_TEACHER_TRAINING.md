# P3-05 — Train Mean DR + SCAF

## Scope

P3-05 performs the first real parameter learning in P3.

The frozen SSL backbone is not loaded into the training loop. Training consumes
the P3-02 offline masked-mean SSL cache:

```text
cached SSL vector [B,D]
        ↓
canonical Mean DR
(one-frame equivalence)
        ↓
Linear(D -> 64), trainable
        ↓
L2-normalized 64D embedding
        ↓
SCAF centers, trainable
        ↓
cross-entropy / backward
```

Only Mean DR and SCAF parameters are given to AdamW.

## Reusing the existing Mean DR with an offline cache

The project Mean DR is:

```text
masked mean [B,T,D]
-> Linear(D,64)
-> L2 normalize
```

P3-02 already performed the masked mean before caching. Therefore a cached
vector `[B,D]` is passed to the existing Mean DR as a one-frame sequence:

```text
[B,D] -> [B,1,D]
mask   -> [B,1] = True
```

The masked mean of one valid frame equals the cached vector exactly. This avoids
creating a second projection-head implementation and keeps checkpoint semantics
aligned with the canonical Mean DR.

## Frozen P3 development score

P3-05 freezes:

```text
generic_dev_score = prototype_macro_f1
```

Definition:

1. Encode the complete TRAIN reference set using the current Mean DR.
2. Average unit embeddings per class and L2-normalize the average to form one
   TRAIN prototype per class.
3. Encode DEV.
4. Classify each DEV sample by maximum cosine similarity to TRAIN prototypes.
5. Compute macro F1 across the frozen train classes.

This score is closed-set and threshold-free.

It is used for P3/P4 checkpoint/backbone/layer comparison only. Open-set
threshold calibration and Correct-Accept / Wrong-Intent / Reject metrics remain
P6 responsibilities.

The DEV labels are never used to build prototypes.

## Recorded dev embedding metrics

Each epoch records:

```text
prototype_accuracy
prototype_macro_f1
intra_class_cosine_mean
wrong_class_cosine_max_mean
top1_top2_margin_mean
prototype_inter_cosine_mean
embedding_norm_mean
embedding_norm_max_abs_error
embedding_std_mean
collapsed
generic_dev_score
```

These are diagnostics. P3-07 must compare checkpoints using only the frozen
`generic_dev_score`.

## Unit norm contract

Every train and dev Mean DR output must be:

```text
float32 [B,64]
finite
L2 norm ~= 1
```

Violation fails immediately.

## Collapse diagnostic

P3-05 records:

```text
collapsed = mean per-dimension std(dev_embedding) < 1e-6
```

This is a hard diagnostic signal, not a model-selection score.

## Optimization

Initial transparent P3-05 optimization defaults:

```text
AdamW
epochs = 20
learning_rate = 1e-3
weight_decay = 1e-4
grad_clip_norm = 5.0
```

These are training-budget parameters, not part of the P3-03 architecture
contract. They must remain fixed for fair P4 backbone/layer comparisons.

## Checkpoints

P3-05 saves **every epoch**, rather than creating a `best.pt` file:

```text
run_dir/
├── checkpoints/
│   ├── epoch_0001.pt
│   ├── epoch_0002.pt
│   └── ...
├── metrics.jsonl
└── checkpoint_hashes.jsonl
```

Every checkpoint stores:

```text
Mean DR state_dict
SCAF state_dict
optimizer state_dict
epoch
seed
train metrics
dev metrics
```

Every checkpoint file receives a SHA-256 hash.

Checkpoint selection is intentionally deferred to P3-07.

## Test isolation

P3-05 accepts TRAIN reference data and DEV only.

P3-04 already refuses construction of test datasets, so P3-05 has no normal
path to `generic_test`.

## P3-05 Gate

PASS requires:

1. finite train loss;
2. gradients update Mean DR + SCAF;
3. every produced embedding satisfies 64D unit norm;
4. dev embedding metrics are finite and machine-readable;
5. `generic_dev_score` is exactly `prototype_macro_f1`;
6. every epoch checkpoint has a SHA-256 hash;
7. run directories are not silently overwritten;
8. no sealed test access.
