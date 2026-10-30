# P6 Enrollment Shot 消融：10-shot 与 15-shot 对比

本实验只考察一个具体问题：

> Can the current frozen Teacher pass the application Gate simply by using
> more REAL enrollment utterances per intent?

不重新训练 Teacher。

## 冻结设置

```text
Backbone   microsoft/wavlm-large
Layer      outputs.hidden_states[15]
Head       AttentionDR -> 64D
Checkpoint representative median-seed checkpoint
Decision   score + top1-top2 margin
Gate       unchanged
Dev split  unchanged
Test       sealed
```

## 改变的因素

```text
10 real enrollment utterances / intent
vs
15 real enrollment utterances / intent
```

两者均采用：

```text
all N embeddings
-> arithmetic mean
-> L2 normalize
-> one mean prototype per intent
```

10-shot 集合严格包含于 15-shot 集合。

本实验有意保持简单，不加入 K=2/K=3 clustering。如果 mean prototype 在 10-shot 或 15-shot 下已通过，就暂时无需引入 subprototype 复杂度。

## 重要解读

这仍是**基于通用公开数据的开发集消融实验**，尚非最终的同一用户个性化基准。

此外，早期 P6 迭代已查看过 `generic_dev_score`，因此它现在属于开发数据。尚未查看的公开 `generic_test` 仍保留给最终冻结评估。

## 运行

```powershell
python scripts/run_p6_10_15_shot_ablation.py --device cuda
python scripts/audit_p6_10_15_shot_ablation.py
```

脚本只为 open-set DEV 生成一次 embeddings，随后在 `generic_dev_cal` 上通过精确经验 threshold calibration 评估两种 shot 数量。
