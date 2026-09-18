# P6 15-shot repeated-split stability diagnostic

The single 15-shot development split produced:

```text
Macro F1       0.8145
Correct Accept 0.7376
Wrong Intent   0.0181
Known Reject   0.2443
Unknown Reject 0.8727
```

The calibration half contained thousands of feasible threshold pairs, but the
held-out dev-score half failed mainly on Correct Accept / Known Reject.

This bundle tests whether that failure is stable across repeated development
partitions.

## What this experiment is

- 15-shot mean prototype is frozen.
- Teacher checkpoint is frozen.
- Core30 DEV (442) is repeatedly split about 50/50 by class.
- Open-set DEV (1178) is repeatedly split 50/50.
- Each split calibrates exact score+margin thresholds on its calibration half.
- Those thresholds are then evaluated on the other half.
- Default = 20 repeated splits.

## What this experiment is NOT

It is not a new final Gate.

The DEV pool has already been observed in prior P6 work. Therefore this is a
development-stability diagnostic only. Public `generic_test` remains sealed.

## Run

First export one reusable all-DEV score file:

```powershell
python scripts/export_p6_15shot_all_dev_scores.py --device cuda
```

Then:

```powershell
python scripts/run_p6_15shot_split_stability.py
```

Then:

```powershell
python scripts/audit_p6_15shot_split_stability.py
```

Key outputs:

```text
calibration feasible rate
absolute Gate pass rate

Macro-F1 mean/std/p10/median/p90
Correct Accept mean/std/p10/median/p90
Known Reject mean/std/p10/median/p90
Unknown Reject mean/std/p10/median/p90

per-component Gate pass rates
```

Interpretation:

- If Correct Accept is below 0.80 on most splits while Unknown Reject often
  passes 0.85, the current embedding/prototype geometry is the limiting factor.
- If full-Gate pass/fail swings widely across splits, calibration/sample-size
  instability is important and should be addressed before retraining.
