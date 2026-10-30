# P4-04 — W2v-BERT 2.0 Hidden-State Sweep

P4-03 已完成。P4-04 只将 Backbone 改为：

```text
facebook/w2v-bert-2.0
revision da985ba0987f70aaeb84a80f2851cfac8c697a7b
outputs.hidden_states indices 0..24
```

其余内容均沿用 P3/P4-01 的冻结设置。

## 1. W2v-BERT 2.0 安全 Frontend 策略

P1 已验证 W2v-BERT 2.0 不会直接接收原始 waveform，而是使用：

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

P2 已将原生 padded acoustic-feature batching 与优先保证正确性的 singleton 路径进行比较并完成验证。

因此，P4-04 会在 feature extractor 之前将每个样本裁到源 waveform 的精确长度。之后，feature extractor 只执行模型原生的 acoustic-feature padding，并返回权威的帧 mask。

笔记本 GPU 的默认 batch size 为 1：

```powershell
--batch-size 1
```

确认 CUDA 显存充足后，可以使用 `--batch-size 2`。这只是吞吐量设置，不会改变特征定义。

## 2. 物化全部 25 份 caches

```powershell
python scripts/materialize_p4_04_w2vbert2_all_layers.py `
  --device cuda `
  --batch-size 1
```

一次模型 forward 会返回全部 25 个 hidden states，因此同一 WAV 不会分别通过 W2v-BERT 运行 25 次。

Expected final audit:

```text
layer  0: COMPLETE (created)
...
layer 24: COMPLETE (created)

P4-04 W2V-BERT 2.0 CACHE FANOUT: PASS
```

## 3. 运行 25 x 3 个冻结的 Teacher 条件

```powershell
python scripts/run_p4_04_w2vbert2_sweep.py --device cuda
```

该命令会执行：

```text
25 hidden-state indices
x 3 seeds
= 75 runs
```

每个 run 均保持以下设置：

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

## 4. 完成审计

所有 run 完成后运行：

```powershell
python scripts/audit_p4_04_w2vbert2_sweep.py
```

通过时必须满足：

```text
25 layers
3 seeds/layer
75 results
generic_test accessed = NO
ranking performed = NO
best layer selected = NO
```

审计结果写入：

```text
artifacts/p4_runs/w2v_bert2/p4_04_raw_results.json
```

完成后，P4-05 才能将三个 Backbone sweeps 汇总为 3-seed mean / sample standard deviation / 95% Student-t confidence intervals。
