# WavLM Multi-Layer Fusion Experiment

## Scientific question

The current Project-1 Teacher uses:

```text
WavLM-large hidden_states[15]
-> Attention DR
-> 256D
-> SCAF
```

The dimension sweep improved the closed-set 3-seed mean to approximately:

```text
0.912667 Macro-F1
```

This experiment asks a new, isolated question:

> Can nearby WavLM layers provide complementary information beyond layer 15?

## Arms

### Control

```text
WavLM layer 15
-> same Attention DR
-> 256D
-> same SCAF
```

### Learnable multi-layer fusion

```text
WavLM layers 13,14,15,16,17
        |
        v
global trainable softmax scalar weights
        |
        v
weighted sum [T,1024]
        |
        v
same Attention DR
        |
        v
256D
        |
        v
same SCAF
```

The trainable fusion has only five scalar logits:

```math
alpha_l = exp(w_l) / sum_j exp(w_j)

H_fused = sum_l alpha_l H^(l)
```

WavLM remains completely frozen.

## Cache

The experiment must first build a train/dev-only hidden-state cache for layers 13..17.

Default storage is float16 to reduce disk usage. Both the control and fusion
arms use this same cache, so precision is controlled.

Expected size is roughly five times one layer, halved by float16. With the prior
frame count this should be around 4.9 GiB of feature tensors.

No test rows are loaded.

## Run order

### 1. Build cache

```powershell
python scripts/build_wavlm_layers13_17_cache.py --device cuda
```

### 2. Smoke test

```powershell
python scripts/run_wavlm_multilayer_fusion.py `
  --smoke `
  --output-dir artifacts/p6_wavlm_multilayer_fusion_smoke
```

### 3. Full experiment

```powershell
python scripts/run_wavlm_multilayer_fusion.py
```

### 4. Summarize

```powershell
python scripts/summarize_wavlm_multilayer_fusion.py
```

### 5. Audit

```powershell
python scripts/audit_wavlm_multilayer_fusion.py
```

## Frozen settings

```text
MDSC Core30 train/dev
Attention DR
256D embedding
SCAF K=3
margin=0.2 rad
scale=30
20 epochs
AdamW lr=1e-3
weight decay=1e-4
grad clip=5
8-way x 4-shot
seeds 17/29/43
selection = dev prototype Macro-F1
generic_test = sealed
```

## Promotion rule

Do not promote multi-layer fusion solely because one seed is higher.

First require:
- a reasonable 3-seed mean improvement over the same-cache single15 control;
- no pathological variance;
- sensible learned layer weights.

If that passes, the next experiment is the same 15-shot personalized/open-set
comparison already used for the 256D Teacher.

The canonical 64D Teacher remains untouched for Project 2.
