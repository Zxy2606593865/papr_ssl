# P4-02 — Wav2Vec2 Hidden-State Sweep

P4-01 froze the exact candidates:

```text
facebook/wav2vec2-base
revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
outputs.hidden_states indices 0..12
```

P4-02 executes these 13 conditions. It does **not** select a best layer.
Aggregation belongs to P4-05 and best-layer selection belongs to P4-06.

## Cache strategy

P3 already produced a complete real Core30 cache for index 12.

Therefore:

```text
layer 12 -> validate and reuse
layers 0..11 -> materialize in one new frozen Wav2Vec2 pass
```

The model already returns all hidden states in one forward, so P4 does not
forward the same WAV eleven separate times.

Run:

```powershell
python scripts/materialize_p4_02_wav2vec2_all_layers.py `
  --device cuda `
  --batch-size 4
```

Expected final audit:

```text
layer  0: COMPLETE (created)
...
layer 11: COMPLETE (created)
layer 12: COMPLETE (reused)

P4-02 WAV2VEC2 CACHE FANOUT: PASS
```

## Training sweep

The frozen condition remains:

```text
MDSC-Core30 train/dev
Mean DR -> 64D L2
SCAF K=3, m=0.2, s=30
20 epochs
AdamW lr=1e-3, wd=1e-4
grad clip=5
8-way x 4-shot
seeds 17/29/43
generic_dev_score = prototype_macro_f1
generic_test sealed
```

Run all missing layer/seed conditions:

```powershell
python scripts/run_p4_02_wav2vec2_sweep.py --device cuda
```

By default layer 12 reuses the already completed P3 17/29/43 evidence, because
that condition has exactly the same task, data, head, loss, sampler, optimizer
and training budget. Rerunning it would only waste compute and introduce a
duplicate result.

The new P4 runs are stored under:

```text
artifacts/p4_runs/wav2vec2_base/
├── layer_00/
│   ├── seed_0017/
│   ├── seed_0029/
│   └── seed_0043/
...
└── layer_11/
```

Layer 12 remains referenced from:

```text
artifacts/p3_runs/mdsc_core30__wav2vec2_base__layer12/
```

## Completion audit

After the sweep:

```powershell
python scripts/audit_p4_02_wav2vec2_sweep.py
```

PASS requires exactly:

```text
13 layers x 3 seeds = 39 results
generic_test accessed = NO
ranking performed = NO
best layer selected = NO
```

The audit writes:

```text
artifacts/p4_runs/wav2vec2_base/p4_02_raw_results.json
```

P4-05 will later add mean / sample std / 95% Student-t CI. P4-06 will use those
aggregated results to select the Wav2Vec2 best layer.
