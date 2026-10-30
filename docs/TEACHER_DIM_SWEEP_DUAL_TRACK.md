# Teacher Embedding Dimension Sweep — 双轨策略

现在将两个项目目标区分开来。

## Project 1：Agent 语音前端
优先考虑性能。如果能提升识别效果，服务端 Teacher 可以使用更大的 embedding。

## Project 2：边缘硬件
优先考虑效率。保留现有 canonical 64D Teacher，用于后续 Student 蒸馏。

因此，本实验**不会**取代 64D。

### 冻结 canonical 64D

```text
WavLM-large
hidden_states[15]
-> Attention DR
-> 64D L2 embedding
-> SCAF
```

### 新一轮 sweep

仅改变 projection 输出维度：

```text
WavLM[15] cached [T,1024]
        |
        v
same Attention Pooling
        |
        v
64 / 128 / 256 / 512
        |
        v
L2 Norm
        |
        v
SCAF (K=3, margin=0.2 rad, scale=30)
```

已冻结的训练设置：

```text
MDSC Core30 train/dev
20 epochs
AdamW lr=1e-3
weight_decay=1e-4
grad clip=5
8-way x 4-shot
seeds 17 / 29 / 43
dev prototype Macro-F1 selection
generic_test sealed
```

## 运行顺序

```powershell
python scripts/audit_teacher_dim_sweep.py
```

然后执行 smoke test：

```powershell
python scripts/run_teacher_dim_sweep.py `
  --smoke `
  --output-dir artifacts/p6_teacher_dim_sweep_smoke
```

若结果为 PASS，则运行完整 sweep：

```powershell
python scripts/run_teacher_dim_sweep.py
```

最后运行：

```powershell
python scripts/summarize_teacher_dim_sweep.py
```

64D sweep 结果仅作为复现对照，不会覆盖 canonical 64D Teacher。

完成此 sweep 后，使用相同的 15-shot personalized/open-set 评估，对比 `canonical 64D` 与 `best high-dimensional candidate`，再决定是否将候选提升为新的 Project-1 Teacher。
