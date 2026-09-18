# P6 K=2 Exact Empirical Threshold Calibration

## Why this amendment exists

The previous K=2 calibrator searched a `101 x 101` quantile grid and returned:

```text
calibration_feasible = False
```

That result proves only that no feasible threshold pair was found on the coarse
grid. It does not prove that no empirical threshold pair exists.

This amendment is triggered by the calibration-set result itself.

## What stays frozen

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

Only the threshold search resolution changes.

## Exact empirical search

Decision rule:

```text
ACCEPT iff
score >= score_threshold
AND
margin >= margin_threshold
```

On a finite calibration set, the accept/reject partition changes only when a
threshold crosses an observed score or margin.

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

This enumerates every distinct empirical decision partition expressible by the
two-threshold policy.

## Run

First export frozen per-sample dev scores:

```powershell
python scripts/export_p6_k2_dev_scores.py --device cuda
```

Then run exact calibration:

```powershell
python scripts/run_p6_k2_exact_calibration.py
```

Then audit:

```powershell
python scripts/audit_p6_k2_exact_calibration.py
```

Interpretation:

- If `calibration_feasible=False` again, the current K=2 two-threshold policy
  truly has no empirical feasible point on `generic_dev_cal`.
- If calibration becomes feasible but `generic_dev_score` fails, the threshold
  policy does not generalize to the independent dev-score split.
- Only if calibration is feasible AND `generic_dev_score` passes all absolute
  Gates may P6-06 freezing resume.

`generic_test` remains sealed in all cases.
