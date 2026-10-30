# P4-02 — Wav2Vec2 Hidden-State Sweep

P4-01 已冻结以下精确候选项：

```text
facebook/wav2vec2-base
revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
outputs.hidden_states indices 0..12
```

P4-02 执行这 13 种条件，**不**选择最佳层。汇总由 P4-05 负责，最佳层选择由 P4-06 负责。

## Cache 策略

P3 already produced a complete real Core30 cache for index 12.

Therefore:

```text
layer 12 -> validate and reuse
layers 0..11 -> materialize in one new frozen Wav2Vec2 pass
```

模型在一次 forward 中就会返回所有 hidden states，因此 P4 不会对同一 WAV 分别执行 11 次 forward。

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

## 训练 Sweep

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

运行所有尚缺的 layer/seed 条件：

```powershell
python scripts/run_p4_02_wav2vec2_sweep.py --device cuda
```

默认情况下，layer 12 会复用已完成的 P3 17/29/43 证据，因为该条件的 task、data、head、loss、sampler、optimizer 和训练预算完全相同。重跑只会浪费算力并产生重复结果。

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

Layer 12 仍引用自：

```text
artifacts/p3_runs/mdsc_core30__wav2vec2_base__layer12/
```

## 完成审计

Sweep 完成后运行：

```powershell
python scripts/audit_p4_02_wav2vec2_sweep.py
```

通过时必须恰好满足：

```text
13 layers x 3 seeds = 39 results
generic_test accessed = NO
ranking performed = NO
best layer selected = NO
```

审计结果写入：

```text
artifacts/p4_runs/wav2vec2_base/p4_02_raw_results.json
```

之后，P4-05 会补充 mean / sample std / 95% Student-t CI；P4-06 再根据汇总结果选择 Wav2Vec2 最佳层。
