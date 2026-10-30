# P6 K=2 精确经验 Threshold Calibration

## 修订原因

The previous K=2 calibrator searched a `101 x 101` quantile grid and returned:

```text
calibration_feasible = False
```

该结果仅证明粗网格上未找到可行 threshold pair，并不能证明不存在经验可行的 threshold pair。

本修订由 calibration set 的实际结果触发。

## 保持冻结的内容

```text
Backbone             WavLM-large
Layer                outputs.hidden_states[15]
Head                 Attention DR 64D
Checkpoint policy    representative median seed
Prototype policy     K=2 independent subprototypes
generic_dev_cal      unchanged
generic_dev_score    unchanged
Gate thresholds      unchanged
generic_test         sealed
```

只调整 threshold search 的分辨率。

## 精确经验搜索

Decision rule:

```text
ACCEPT iff
score >= score_threshold
AND
margin >= margin_threshold
```

在有限 calibration set 上，只有当 threshold 越过某个已观测的 score 或 margin 时，accept/reject 划分才会改变。

Therefore the exact candidate set is:

```text
all unique observed score values
+ one boundary below min
+ one boundary above max

x

all unique observed margin values
+ one boundary below min
+ one boundary above max
```

该方法枚举双 threshold policy 能表示的所有不同经验决策划分。

## 运行

先导出冻结的逐样本 dev scores：

```powershell
python scripts/export_p6_k2_dev_scores.py --device cuda
```

然后执行精确校准：

```powershell
python scripts/run_p6_k2_exact_calibration.py
```

之后进行审计：

```powershell
python scripts/audit_p6_k2_exact_calibration.py
```

结果解读：

- If `calibration_feasible=False` again, the current K=2 two-threshold policy
  truly has no empirical feasible point on `generic_dev_cal`.
- If calibration becomes feasible but `generic_dev_score` fails, the threshold
  policy does not generalize to the independent dev-score split.
- Only if calibration is feasible AND `generic_dev_score` passes all absolute
  Gates may P6-06 freezing resume.

`generic_test` remains sealed in all cases.
