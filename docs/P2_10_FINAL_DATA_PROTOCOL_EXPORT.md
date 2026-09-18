# P2-10 — Final Data Protocol Export

P2-10 is the archival/finalization stage of P2. It does not rescan raw audio, rebuild manifests, redefine classes, change splits, or perform semantic merging.

It packages the already-frozen P2 outputs into:

```text
docs/DATA_PROTOCOL.md

artifacts/data_audit/
├── gsc_raw_audit.json
├── mdsc_raw_audit.json
├── mdsc_phrase_inventory_summary.json
├── mdsc_task_policy_v2_summary.json
├── p2_final_closure.json
├── data_audit_summary.json
└── manifest.json
```

Run after the full test suite is green:

```powershell
python scripts/export_p2_final_data_protocol.py `
  --unit-tests-passed 73
```

If the raw-audit source paths differ locally, pass `--gsc-raw-audit` and `--mdsc-raw-audit` explicitly.
