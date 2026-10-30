# P6 15-shot 重复划分稳定性诊断

The single 15-shot development split produced:

```text
Macro F1       0.8145
Correct Accept 0.7376
Wrong Intent   0.0181
Known Reject   0.2443
Unknown Reject 0.8727
```

Calibration half 包含数千个可行 threshold pairs，但留出的 dev-score half 主要未通过 Correct Accept / Known Reject 指标。

本方案用于检查该失败情况在重复开发集划分中是否稳定。

## 本实验的内容

- 15-shot mean prototype is frozen.
- Teacher checkpoint is frozen.
- Core30 DEV (442) is repeatedly split about 50/50 by class.
- Open-set DEV (1178) is repeatedly split 50/50.
- Each split calibrates exact score+margin thresholds on its calibration half.
- Those thresholds are then evaluated on the other half.
- Default = 20 repeated splits.

## 本实验不是什么

它不是新的最终 Gate。

此前的 P6 工作已查看过 DEV pool。因此本实验仅用于诊断开发集稳定性。公开 `generic_test` 继续封存。

## 运行

先导出一份可复用的全 DEV score file：

```powershell
python scripts/export_p6_15shot_all_dev_scores.py --device cuda
```

随后运行：

```powershell
python scripts/run_p6_15shot_split_stability.py
```

Then:

```powershell
python scripts/audit_p6_15shot_split_stability.py
```

主要输出：

```text
calibration feasible rate
absolute Gate pass rate

Macro-F1 mean/std/p10/median/p90
Correct Accept mean/std/p10/median/p90
Known Reject mean/std/p10/median/p90
Unknown Reject mean/std/p10/median/p90

per-component Gate pass rates
```

结果解读：

- 如果大多数划分的 Correct Accept 低于 0.80，而 Unknown Reject 经常高于 0.85，则当前 embedding/prototype 几何结构是主要限制因素。
- 如果完整 Gate 的通过与否随划分大幅波动，则 calibration/sample-size instability 影响较大，应在重新训练前先解决。
