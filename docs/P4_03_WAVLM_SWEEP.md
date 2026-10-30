# P4-03 — WavLM-large Hidden-State Sweep

P4-02 已完成。P4-03 只将 Backbone 改为：

```text
microsoft/wavlm-large
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
outputs.hidden_states indices 0..24
```

其余内容均沿用 P3/P4-01 的冻结设置。

## 1. WavLM 安全批处理

P2-06C experimentally validated:

```text
WavLM -> native right-padded batch
```

因此，P4-03 **不**使用 Wav2Vec2 的精确长度分组。

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

长度排序只用于提升吞吐量、减少内存开销。它能减少无效 padding，但不会改变模型输入语义。

默认 cache batch size 为 2，因为在笔记本 GPU 上，WavLM-large 和较长的 MDSC utterances 可能比 Wav2Vec2-base 占用更多资源。

若发生 CUDA OOM：

```powershell
--batch-size 1
```

这不会改变特征定义。

## 2. 物化全部 25 份 caches

```powershell
python scripts/materialize_p4_03_wavlm_all_layers.py `
  --device cuda `
  --batch-size 2
```

一次 WavLM forward 会返回全部 25 个 hidden states。同一段音频不会被单独 forward 25 次。

Expected:

```text
layer  0: COMPLETE (created)
...
layer 24: COMPLETE (created)

P4-03 WAVLM CACHE FANOUT: PASS
```

## 3. 运行 25 x 3 个冻结的 Teacher 条件

```powershell
python scripts/run_p4_03_wavlm_sweep.py --device cuda
```

即：

```text
25 hidden-state indices
x 3 seeds
= 75 runs
```

每个 run 均保持以下设置：

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

## 4. 完成审计

75 种条件全部完成后运行：

```powershell
python scripts/audit_p4_03_wavlm_sweep.py
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

P4-03 有意不选择 WavLM 最佳层。P4-05 会计算 three-seed mean / sample std / 95% Student-t CI，P4-06 再为每个 Backbone 选出最佳项。
