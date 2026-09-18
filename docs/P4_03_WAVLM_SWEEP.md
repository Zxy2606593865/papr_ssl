# P4-03 — WavLM-large Hidden-State Sweep

P4-02 is closed. P4-03 changes only the Backbone to:

```text
microsoft/wavlm-large
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
outputs.hidden_states indices 0..24
```

Everything else remains frozen from P3/P4-01.

## 1. WavLM safe batching

P2-06C experimentally validated:

```text
WavLM -> native right-padded batch
```

Therefore P4-03 does **not** use Wav2Vec2's exact-length grouping.

Instead:

```text
variable-length WAVs
        ↓
length-sort rows
        ↓
small native padded batches
        ↓
sample attention_mask
        ↓
WavLM one forward
        ↓
_get_feature_vector_attention_mask
        ↓
valid frame_mask
        ↓
masked mean for all 25 hidden states
```

Length sorting is only a throughput/memory optimization. It reduces wasted
padding without changing the model input semantics.

Default cache batch size is 2 because WavLM-large and long MDSC utterances can
be substantially heavier than Wav2Vec2-base on a laptop GPU.

If CUDA OOM occurs:

```powershell
--batch-size 1
```

This does not change the feature definition.

## 2. Materialize all 25 caches

```powershell
python scripts/materialize_p4_03_wavlm_all_layers.py `
  --device cuda `
  --batch-size 2
```

One WavLM forward returns all 25 hidden states. The same audio is not forwarded
25 separate times.

Expected:

```text
layer  0: COMPLETE (created)
...
layer 24: COMPLETE (created)

P4-03 WAVLM CACHE FANOUT: PASS
```

## 3. Run 25 x 3 frozen Teacher conditions

```powershell
python scripts/run_p4_03_wavlm_sweep.py --device cuda
```

This means:

```text
25 hidden-state indices
x 3 seeds
= 75 runs
```

Each run remains:

```text
Core30 train/dev
Mean DR -> 64D L2
SCAF K=3, m=0.2, s=30
20 epochs
AdamW lr=1e-3
weight_decay=1e-4
grad_clip=5
8-way x 4-shot
generic_dev_score = prototype_macro_f1
generic_test sealed
```

## 4. Completion audit

After all 75 conditions:

```powershell
python scripts/audit_p4_03_wavlm_sweep.py
```

PASS requires:

```text
25 layers
3 seeds/layer
75 results
generic_test accessed = NO
ranking performed = NO
best layer selected = NO
```

P4-03 deliberately does not choose WavLM's best layer. P4-05 will compute
three-seed mean / sample std / 95% Student-t CI, and P4-06 will select the
per-Backbone winner.
