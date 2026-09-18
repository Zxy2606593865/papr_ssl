# Closed-set evaluation for current DTW system

This is a diagnostic requested after DTW improved the personalized open-set system.

The goal is to answer:

> If we remove Unknown/Reject entirely and only ask the system to classify one of the 30 known Core30 phrases, does DTW still help?

## Protocol

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

For each of 20 repeated known-only splits:

```text
calibration half
-> choose lambda from {0.50, 0.75, 1.00}

held-out score half
-> evaluate 30-way closed-set Macro-F1 + Accuracy
```

This preserves the existing development protocol and avoids selecting lambda on the same examples used for reporting.

## Run

```powershell
python scripts/run_closedset_dtw_eval.py
python scripts/audit_closedset_dtw_eval.py
```

## Interpretation

This result is different from the earlier ~91.27% Teacher closed-set result.

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

So use this experiment to quantify the value of DTW in the personalized closed-set setting, not to replace the original Teacher closed-set benchmark.

generic_test remains sealed.
