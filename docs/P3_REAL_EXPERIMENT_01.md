# P3 Scientific Gate — Real Experiment 01

This is **not P3-09**. P3-01 through P3-08 are already the complete engineering
plan. This document executes those contracts on real data.

## First real condition

```text
Task:      MDSC-Core30
Backbone:  facebook/wav2vec2-base
Revision:  0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
Layer:     outputs.hidden_states[12]
Pooling:   masked mean, offline cache
Head:      Mean DR -> 64D L2
Loss:      SCAF K=3, m=0.2, s=30
Seed:      17 first
Test:      sealed
```

The frozen Core30 policy contains 3756 train and 442 dev utterances across 30
classes. The first real cache contains train+dev only; the Core30 test audio is
not materialized for P3 model development.

## Why Wav2Vec2 first?

The goal is to validate one real end-to-end P3 condition before expanding the
matrix. Wav2Vec2-base is the smallest of the three P1 backbones and its final
hidden-state index is already frozen and validated.

## Step 1 — preflight

From the project root:

```powershell
python scripts/preflight_p3_real_run01.py --check-all-audio
```

Expected:

```text
train rows: 3756
dev rows:   442
classes:    30
REAL RUN 01 PREFLIGHT: PASS
```

The default MDSC root is:

```text
datasets/public/mdsc
```

Override `--mdsc-root` only if your local P2 audit uses another location.

## Step 2 — materialize the real SSL cache

```powershell
python scripts/materialize_p3_core30_wav2vec2_cache.py `
  --layers 12 `
  --device cuda `
  --batch-size 4
```

Important: Wav2Vec2 is forwarded only in exact semantic waveform-length groups.
No right-padding is introduced into the SSL forward, matching the safe P2-06
policy.

The cache is written under:

```text
artifacts/ssl_feature_cache/
facebook__wav2vec2-base__rev-0b5b8e868dd8__layer-12__masked-mean-f32/
```

Do not regenerate the cache for seed 29/43. The SSL backbone is frozen, so all
three seeds reuse exactly this cache.

If CUDA memory is insufficient, lower:

```text
--batch-size 4
```

to `2` or `1`. This changes throughput, not the semantic feature definition,
because every chunk contains only equal-length waveforms and no padding.

## Step 3 — first real Teacher run

```powershell
python scripts/run_p3_core30_wav2vec2_seed17.py --device cuda
```

This performs:

```text
untrained random 64D projection baseline
        ↓
20-epoch Mean DR + SCAF training
        ↓
dev metrics every epoch
        ↓
P3-07 dev-only checkpoint selection
        ↓
checkpoint SHA-256 validation
        ↓
P3-08 run_manifest.json
```

The baseline and trained model use the same cached SSL features and the same
seed-17 random initialization before training. Therefore the comparison isolates
the effect of Mean DR + SCAF learning.

Outputs:

```text
artifacts/p3_runs/
└── mdsc_core30__wav2vec2_base__layer12/
    └── seed_0017/
        ├── untrained_projection_baseline.json
        ├── checkpoints/
        ├── metrics.jsonl
        ├── checkpoint_hashes.jsonl
        ├── selected_checkpoint.json
        ├── baseline_vs_trained.json
        └── run_manifest.json
```

## Interpretation

Do not declare the overall P3 scientific gate after seed 17.

Seed 17 answers only:

> Can the real Core30 + Wav2Vec2 baseline train successfully and improve over
> its own untrained projection?

If it succeeds, run the same frozen experiment with seeds 29 and 43. The P3
scientific conclusion must use the three-seed result, not the best single seed.

## Scientific gate evidence to collect

For the real run, verify:

```text
finite train loss
no NaN / Inf
64D unit-norm error within contract
collapsed = false
selected checkpoint based only on generic_dev_score
trained dev score vs untrained projection
checkpoint SHA-256
run manifest
generic_test sealed
```

Only after the three seeds are complete should the project decide whether this
Backbone/Layer satisfies the P3 scientific gate.
