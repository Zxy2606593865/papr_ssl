# 当前 DTW 系统的闭集评估

DTW 提升个性化 open-set 系统后，需要进行此诊断。

本评估旨在回答：

> If we remove Unknown/Reject entirely and only ask the system to classify one of the 30 known Core30 phrases, does DTW still help?

## 协议

```text
MDSC Core30 DEV only
30 known classes
442 known DEV utterances
same 15-shot enrollment
no unknown utterances
no rejector
no score threshold
no ACCEPT/REJECT
```

对 20 个重复的仅含已知类 split，分别执行：

```text
calibration half
-> choose lambda from {0.50, 0.75, 1.00}

held-out score half
-> evaluate 30-way closed-set Macro-F1 + Accuracy
```

这既遵循现有开发协议，也避免在同一批样本上选择 lambda 并汇报结果。

## 运行

```powershell
python scripts/run_closedset_dtw_eval.py
python scripts/audit_closedset_dtw_eval.py
```

## 结果解读

此结果与此前约 91.27% 的 Teacher 闭集结果含义不同。

The earlier result measures:
```text
trained Teacher representation
-> prototype classification using full train reference set
-> Core30 DEV
```

This new result measures:
```text
15-shot enrollment only
-> personalized prototype / DTW
-> held-out Core30 DEV
```

因此，本实验用于量化 DTW 在个性化闭集场景中的价值，不应替代原始 Teacher 闭集基准。

`generic_test` 继续保持封存。
