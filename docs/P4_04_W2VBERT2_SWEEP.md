# P4-04 — W2v-BERT 2.0 Hidden-State Sweep

P4-03 is closed. P4-04 changes only the Backbone to:

```text
facebook/w2v-bert-2.0
revision da985ba0987f70aaeb84a80f2851cfac8c697a7b
outputs.hidden_states indices 0..24
```

Everything else remains frozen from P3/P4-01.

## 1. W2v-BERT 2.0 safe frontend policy

P1 validated that W2v-BERT 2.0 does not consume raw waveform directly.
It uses:

```text
exact semantic waveform
        ↓
AutoFeatureExtractor
        ↓
input_features [B,T,160]
attention_mask [B,T]
        ↓
Wav2Vec2BertModel
        ↓
hidden_states[0..24] [B,T,1024]
```

P2 validated native padded acoustic-feature batching against the
correctness-first singleton path.

P4-04 therefore trims every sample to the exact source waveform *before* the
feature extractor. The feature extractor then performs only the model-native
acoustic-feature padding and returns the authoritative frame mask.

Default batch size is 1 for a laptop GPU:

```powershell
--batch-size 1
```

After confirming sufficient CUDA memory, `--batch-size 2` is allowed. This is
only a throughput choice and does not change the feature definition.

## 2. Materialize all 25 caches

```powershell
python scripts/materialize_p4_04_w2vbert2_all_layers.py `
  --device cuda `
  --batch-size 1
```

One model forward returns all 25 hidden states, so the same WAV is not run
through W2v-BERT 25 separate times.

Expected final audit:

```text
layer  0: COMPLETE (created)
...
layer 24: COMPLETE (created)

P4-04 W2V-BERT 2.0 CACHE FANOUT: PASS
```

## 3. Run 25 x 3 frozen Teacher conditions

```powershell
python scripts/run_p4_04_w2vbert2_sweep.py --device cuda
```

This executes:

```text
25 hidden-state indices
x 3 seeds
= 75 runs
```

Every run remains:

```text
MDSC-Core30 train/dev
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

After all runs:

```powershell
python scripts/audit_p4_04_w2vbert2_sweep.py
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

The audit writes:

```text
artifacts/p4_runs/w2v_bert2/p4_04_raw_results.json
```

After this, P4-05 may finally aggregate all three Backbone sweeps into
3-seed mean / sample standard deviation / 95% Student-t confidence intervals.
