# P6 15-shot lightweight rejector ablation

## Why this experiment comes next

The frozen 15-shot system has already shown:

- strong known-intent discrimination,
- very low Wrong Intent,
- unstable Known/Unknown separation.

The Teacher itself is therefore not changed in this experiment.

## Frozen components

```text
WavLM-large hidden_states[15]
AttentionDR -> 64D
15-shot mean prototype
top-1 intent prediction
```

## Only changed component

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

Chinese interpretation:

- Logistic regression = 逻辑回归
- Rejector = 拒识器
- Calibration = 校准
- Known = 已知短语
- Unknown = 未知短语

The rejector is deliberately tiny. It tests whether the problem is mainly the
shape of the decision boundary rather than the 64D representation.

## Scientific control

- Same 20 repeated DEV splits.
- Rejector fits only on each calibration half.
- Threshold is chosen only on the calibration half.
- Score half is held out for that split.
- No generic_test access.
- Intent class prediction itself is never changed.

## Run

The previous 15-shot all-DEV score export and repeated-split baseline must
already exist.

```powershell
python scripts/run_p6_15shot_rejector_ablation.py
python scripts/audit_p6_15shot_rejector_ablation.py
```

## Decision rule after this experiment

If the learned rejector materially improves Correct Accept / Unknown Reject and
raises repeated-split Gate pass rate, keep the Teacher frozen and develop the
decision layer.

If the improvement is small or Gate pass rate remains near zero, stop tuning the
decision layer and move to open-set-aware representation training.
