# P6 Gate 失败诊断

The exact K=2 threshold audit established:

```text
659,344 empirical threshold pairs
0 feasible pairs
```

因此，threshold search 分辨率不足已不太可能是失败原因。

该诊断不选择新 threshold，而是衡量冻结的 K=2 score+margin 规则在 `generic_dev_cal` 上的 Pareto 极限。

报告以下指标：

1. Maximum Correct Accept achievable while Unknown Reject >= 0.85.
2. Maximum Unknown Reject achievable while Correct Accept >= 0.80.
3. Maximum Unknown Reject while Wrong Intent <= 0.10 and Known Reject <= 0.20.
4. The nearest empirical operating point to the full absolute Gate.
5. Score/margin quantiles for:
   - correctly classified known samples,
   - wrongly classified known samples,
   - unknown samples.

运行：

```powershell
python scripts/diagnose_p6_k2_exact_gate.py
```

本流程仅用于诊断。不得据此修改 thresholds 后，又声称同一校准集仍是未触碰的 Gate。`generic_test` 继续封存。
