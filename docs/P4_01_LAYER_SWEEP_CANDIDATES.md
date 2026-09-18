# P4-01 — Backbone / Hidden-State Sweep Candidate Freeze

## 1. Purpose

P4 changes only:

```text
Backbone
selected outputs.hidden_states index
```

Everything else remains frozen from P3:

```text
MDSC-Core30 train/dev
Mean DR -> 64D -> L2
SCAF K=3, margin=0.2, scale=30
AdamW
20 epochs
lr=1e-3
weight_decay=1e-4
grad_clip=5
8-way x 4-shot
seeds 17/29/43
checkpoint selection by generic_dev_score only
generic_test sealed
```

## 2. Exact index semantics

The P4 unit is always:

```text
outputs.hidden_states[i]
```

where `i` is the exact Hugging Face tuple index observed and validated in P1.

P4 does **not** translate a paper phrase such as `Layer 16` into
`hidden_states[16]` by assumption.

If a paper-layer comparison is needed later, a separate architecture-specific
mapping must first be verified from the implementation.

## 3. Frozen candidates

P1 validated:

```text
Wav2Vec2-base
hidden_states count = 13
valid indices = 0..12

WavLM-large
hidden_states count = 25
valid indices = 0..24

W2v-BERT 2.0
hidden_states count = 25
valid indices = 0..24
```

P4-01 deliberately keeps **all** validated indices rather than cherry-picking
"interesting" middle layers.

Total:

```text
13 + 25 + 25 = 63 Backbone/Layer configurations
```

This is computationally practical because the P3-02 cache design can fan out
multiple hidden-state layers from one frozen SSL forward. The expensive SSL
forward therefore need not be repeated once per layer.

Index 0 is retained because it is a real P1-validated member of
`outputs.hidden_states`. It is reported as exact index 0 and is not mislabeled
as paper "Layer 0" or Transformer block 0.

## 4. P4 selection metric

The P4 gate and the user's frozen plan require one primary ranking quantity:

```text
generic_dev_score = prototype_macro_f1
```

For every Backbone/Layer configuration:

```text
seed17 selected generic_dev_score
seed29 selected generic_dev_score
seed43 selected generic_dev_score
        ↓
mean
sample standard deviation
95% Student-t confidence interval, df=2
```

Because n=3, the 95% CI must use Student-t rather than `mean ± 1.96*SE`.

The ranking score is the **3-seed mean generic_dev_score**.

Exact-mean tie break is frozen as:

```text
lower hidden_state_index
```

This is only a deterministic exact-tie rule; it is not a claim that shallower
layers are intrinsically better.

## 5. Metrics that should NOT select P4

The following metrics are valuable, but they require an acceptance/rejection
threshold and open-set calibration:

```text
Correct Accept Rate
Wrong Intent Rate
Known Reject Rate / FRR
Unknown Reject Rate / FAR
ACC/DET @ 1% FAR
```

Using them during P4 would mix Backbone/Layer selection with the later
open-set/threshold qualification problem.

Therefore they are deferred from P4 selection. They belong to the later
Teacher qualification stage once the single Backbone/Layer mainline is frozen.

P4 may still record threshold-free embedding diagnostics:

```text
prototype_accuracy
intra_class_cosine_mean
wrong_class_cosine_max_mean
top1_top2_margin_mean
prototype_inter_cosine_mean
embedding norm error
collapse flag
```

but none may override `generic_dev_score`.

## 6. P4 execution logic

```text
P4-01 candidate freeze
        ↓
P4-02 Wav2Vec2 0..12 sweep
        ↓
P4-03 WavLM 0..24 sweep
        ↓
P4-04 W2v-BERT 2.0 0..24 sweep
        ↓
P4-05 3-seed mean / sample std / 95% t-CI
        ↓
P4-06 best layer per Backbone, then compare three winners
        ↓
P4-07 attach resource-cost report
        ↓
P4-08 freeze one Backbone + one exact hidden_state_index
```

## 7. P4-01 Gate

PASS when:

1. exact P1 model ids/revisions are frozen;
2. candidate indices are a subset of P1-valid indices;
3. this baseline uses full P1-valid coverage: 13 / 25 / 25;
4. index semantics are explicitly `outputs.hidden_states[i]`;
5. no paper-layer alias is inferred;
6. selection metric remains `generic_dev_score=prototype_macro_f1`;
7. seeds remain 17/29/43;
8. generic_test remains sealed.
