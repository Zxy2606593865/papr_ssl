# P6 Unknown Exposure Audit v2 — Core30 schema fix

The previous version failed with:

```text
RuntimeError: expected 30 normalized Core30 phrases, got 0
```

This was a schema-parsing bug in the audit script, not a data problem.

The P2 task-index files should be parsed through the project's existing:

```python
papr_ssl.training.teacher.real_experiment.load_real_task_rows
```

rather than assuming that the JSONL task index directly stores phrase text in
a top-level `label` or `label_text` field.

## Run

Use the new v2 output directory, so the failed v1 artifact is not overwritten:

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v2.py
python scripts/review_p6_unknown_exposure_audit_v2.py
```

If the full MDSC manifest cannot be auto-discovered, pass it explicitly:

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v2.py `
  --mdsc-manifest path\to\full_mdsc_manifest.jsonl
```

No model training is performed. `generic_test` remains sealed.
