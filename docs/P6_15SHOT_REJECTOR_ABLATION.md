# P6 15-shot 轻量拒识器消融实验

## 为什么下一步进行此实验

已冻结的 15-shot 系统表现出：

- strong known-intent discrimination,
- very low Wrong Intent,
- unstable Known/Unknown separation.

因此，本实验不会更改 Teacher 本身。

## 已冻结的组件

```text
WavLM-large hidden_states[15]
AttentionDR -> 64D
15-shot mean prototype
top-1 intent prediction
```

## 唯一变更的组件

Old rejection rule:

```text
top1 cosine >= score threshold
AND
(top1 - top2) >= margin threshold
```

New rejection rule:

```text
[top1 cosine,
 margin,
 score^2,
 margin^2,
 score*margin]
        ->
balanced logistic regression
        ->
P(KNOWN)
        ->
one calibrated probability threshold
```

术语说明：

- Logistic regression = 逻辑回归
- Rejector = 拒识器
- Calibration = 校准
- Known = 已知短语
- Unknown = 未知短语

拒识器刻意保持轻量，用于检验问题是否主要来自决策边界形状，而非 64D 表征。

## 实验控制

- Same 20 repeated DEV splits.
- Rejector fits only on each calibration half.
- Threshold is chosen only on the calibration half.
- Score half is held out for that split.
- No generic_test access.
- Intent class prediction itself is never changed.

## 运行

此前的 15-shot 全 DEV 分数导出和重复 split 基线必须已经生成。

```powershell
python scripts/run_p6_15shot_rejector_ablation.py
python scripts/audit_p6_15shot_rejector_ablation.py
```

## 本实验后的决策规则

如果学习得到的拒识器明显提升 Correct Accept / Unknown Reject，并提高重复 split 的 Gate 通过率，则保持 Teacher 冻结并继续开发决策层。

如果提升有限，或 Gate 通过率仍接近零，则停止调优决策层，转向 open-set-aware 表征训练。
