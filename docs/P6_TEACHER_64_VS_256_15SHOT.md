# P6 — 相同 15-shot personalized/open-set 协议下的 64D 与 256D Teacher 对比

本实验是 embedding-dimension sweep 的应用级后续评估。

Closed-set result so far:

```text
canonical 64D Teacher   ~0.893376 Macro-F1
256D candidate          ~0.912667 mean Macro-F1
```

只有当 256D Teacher 也能改善真实 15-shot personalized/open-set metrics 时，才会将其提升为 Project 1 的方案。

无论结果如何，Project 2 都继续使用规范的 64D Teacher。

## 代表性 256D Checkpoint

Exporter 会读取 seeds 17/29/43，并选择 selected-dev-score 的中位数 checkpoint，不挑选表现最好的单一 seed。

根据当前结果预计会选中 seed 43，但实际选择由脚本根据结果文件计算。

## 协议

Same as the strongest existing 64D downstream evaluation:

```text
15-shot enrollment
mean prototype for intent prediction
retain all 15 enrollment embeddings
enrollment-consistency features
balanced logistic rejector
20 repeated DEV splits
calibration half / score half
generic_test sealed
```

## 运行

```powershell
python scripts/export_teacher256_15shot_embeddings.py --device cuda
python scripts/run_teacher256_vs64_15shot.py
python scripts/audit_teacher256_vs64_15shot.py
```

输出写入：

```text
artifacts/p6_teacher_256_15shot/
```

不会覆盖现有的 64D artifact。
