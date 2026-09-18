# P6 Unknown Exposure Audit v3 — train/dev-only task-index fix

The v2 audit failed because it called:

```python
load_real_task_rows(..., splits=("train", "dev", "test"))
```

but the existing PAPR-SSL helper intentionally blocks `test` access during
development:

```text
ValueError: real P3 execution may load train/dev only
```

This is a protection mechanism, not a data error.

For the purpose of recovering the 30 Core30 phrase identities, `train + dev`
is sufficient. v3 therefore changes only this call to:

```python
load_real_task_rows(..., splits=("train", "dev"))
```

The open-set task index is still read with `splits=("dev",)` only.

No generic test audio, embeddings, or labels are loaded.

Run:

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v3.py
python scripts/review_p6_unknown_exposure_audit_v3.py
```

Outputs go to:

```text
artifacts/p6_unknown_exposure_audit_v3/
```
