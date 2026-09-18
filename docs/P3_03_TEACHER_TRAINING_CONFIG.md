# P3-03 — Teacher Training Config

## Position inside P3

P3 establishes the simplest comparable Teacher baseline:

```text
Frozen pretrained SSL backbone
        ↓
selected outputs.hidden_states[layer]
        ↓
Masked Mean
        ↓
Linear(D -> 64), trainable
        ↓
L2 normalization
        ↓
SCAF centers, trainable
```

P3-03 freezes the **experiment identity and model hyperparameter contract**.
It does not yet define the sampler (P3-04), optimizer/training loop (P3-05),
multi-seed repetition (P3-06), checkpoint selection rule (P3-07), or final run
manifest (P3-08).

## Frozen fields

Every Teacher run config must explicitly contain:

```text
backbone.name
backbone.model_id
backbone.model_revision
backbone.layer
backbone.frozen = true
backbone.force_eval = true
backbone.use_offline_cache = true

head.kind = mean_dr
head.embedding_dim = 64
head.l2_normalize = true

scaf.k = 3
scaf.margin = 0.2
scaf.scale = 30

seed
```

`backbone.layer` always means the exact tuple index in
`outputs.hidden_states`.

## Base configs

P3-03 provides one canonical base config for each frozen P1 backbone:

```text
wav2vec2-base   -> hidden_states[12]
wavlm-large     -> hidden_states[24]
w2v-bert-2.0   -> hidden_states[24]
```

These are baseline/default final-layer configs, **not the result of the future
layer sweep**. A layer-sweep run may override only `backbone.layer` while
preserving the same Mean-DR/SCAF contract.

## Seed policy

The base config uses:

```text
seed = 17
```

P3-06 will repeat qualifying experiments with:

```text
17 / 29 / 43
```

The seed belongs in the config now so every run is reproducible from the start.

## Relationship to P3-02 cache

P3-03 records the exact model revision and hidden-state layer used by P3-02.
Therefore a training run must load only a cache whose identity matches:

```text
model_id
model_revision
layer
pooling = masked_mean
```

A cache mismatch is a configuration error, not something to silently ignore.

## What is not frozen here

P3-03 intentionally does not add:

```text
dataset/task view
batch sampler
batch size
learning rate
optimizer
epoch count
generic_dev_score formula
checkpoint path/hash
```

Those belong to P3-04 through P3-08. Keeping them out of this step prevents
P3-03 from absorbing later responsibilities.

## P3-level review note

The overall P3 plan is coherent. Two rules should be kept explicit:

1. P3-02 caches the **masked-mean pre-projection SSL vector [D]**. The trainable
   `Linear(D -> 64)` remains part of Mean DR and must not be precomputed.
2. `generic_dev_score` must have one frozen mathematical definition before
   P3-05/P3-07 start comparing checkpoints. The test split remains sealed.

## Gate

P3-03 passes when all three backbone configs:

- validate against their exact P1 model revision;
- use an in-range exact hidden-state index;
- keep the backbone frozen/eval/cache policy;
- use `mean_dr`;
- output 64D L2-normalized embeddings by contract;
- share SCAF `(K=3, m=0.2, s=30)`;
- carry an explicit seed.
