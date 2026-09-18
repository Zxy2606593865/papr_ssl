# P6 Gate Failure Diagnostic

The exact K=2 threshold audit established:

```text
659,344 empirical threshold pairs
0 feasible pairs
```

Therefore threshold search resolution is no longer a plausible explanation.

This diagnostic does not select a new threshold. It measures the Pareto limit
of the frozen K=2 score+margin rule on `generic_dev_cal`.

It reports:

1. Maximum Correct Accept achievable while Unknown Reject >= 0.85.
2. Maximum Unknown Reject achievable while Correct Accept >= 0.80.
3. Maximum Unknown Reject while Wrong Intent <= 0.10 and Known Reject <= 0.20.
4. The nearest empirical operating point to the full absolute Gate.
5. Score/margin quantiles for:
   - correctly classified known samples,
   - wrongly classified known samples,
   - unknown samples.

Run:

```powershell
python scripts/diagnose_p6_k2_exact_gate.py
```

This is diagnostic-only. It must not be used to alter thresholds and then claim
the same calibration set as an untouched Gate. `generic_test` remains sealed.
