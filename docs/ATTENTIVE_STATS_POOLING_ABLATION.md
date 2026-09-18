# Attentive Statistics Pooling Ablation

## Goal

The current Project-1 Teacher is:

```text
WavLM-large
-> hidden_states[15]
-> Attention Pooling
-> 256D
-> SCAF
```

The multi-layer fusion experiment did not improve the model, so that branch is
closed.

This experiment asks a new question:

> Does retaining temporal dispersion information improve the 256D Teacher?

## Control: matched attentive mean

```text
WavLM[15] [T,1024]
        |
        v
Additive Attention
        |
        v
weighted mean mu [1024]
        |
        v
Linear 1024 -> 256
        |
        v
L2 Norm
        |
        v
SCAF
```

## Candidate: Attentive Statistics Pooling

```text
WavLM[15] [T,1024]
        |
        v
same Additive Attention
       / \
      /   \
     v     v
    mu    sigma
      \   /
       \ /
     concat [2048]
        |
        v
Linear 2048 -> 256
        |
        v
L2 Norm
        |
        v
SCAF
```

Equations:

```math
alpha_t = softmax(v^T tanh(W h_t))

mu = sum_t alpha_t h_t

sigma = sqrt(sum_t alpha_t h_t^2 - mu^2)

z = L2Norm(Linear([mu, sigma]))
```

The only conceptual difference is whether temporal standard deviation is
retained.

## Attention architecture control

The scripts inspect the current `build_dr("attention")` implementation and infer
the hidden size of its additive attention scorer. The matched control and ASP
candidate both use that same hidden size.

This avoids silently changing the attention capacity.

## Frozen settings

```text
Backbone: WavLM-large
Layer: hidden_states[15]
Feature cache: existing P5 layer15 cache
Embedding: 256D
SCAF: K=3, margin=0.2 rad, scale=30
Epochs: 20
Optimizer: AdamW
LR: 1e-3
Weight decay: 1e-4
Grad clip: 5
Sampler: 8-way x 4-shot
Seeds: 17 / 29 / 43
Selection: dev prototype Macro-F1
generic_test: sealed
```

## Run order

### 1. Pre-flight audit

```powershell
python scripts/audit_attentive_stats_setup.py
```

### 2. Smoke test

```powershell
python scripts/run_attentive_stats_ablation.py `
  --smoke `
  --output-dir artifacts/p6_attentive_stats_ablation_smoke
```

### 3. Full experiment

```powershell
python scripts/run_attentive_stats_ablation.py
```

### 4. Summarize

```powershell
python scripts/summarize_attentive_stats_ablation.py
```

### 5. Audit

```powershell
python scripts/audit_attentive_stats_ablation.py
```

## Decision

Do not promote ASP based on one seed.

First require:
- the matched attentive-mean control reproduces the prior 256D level within the
  predefined 0.02 absolute tolerance;
- ASP has a better three-seed mean;
- ASP does not show pathological instability.

If those conditions are met, run the same 15-shot personalized/open-set
evaluation used for the current 256D Teacher.

If ASP shows no gain, close the pooling-variant branch and move to the next
substantive optimization: temporal latent sequence + DTW.

The canonical 64D Teacher remains untouched for the hardware project.
