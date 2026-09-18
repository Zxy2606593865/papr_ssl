# P6 Unknown Exposure Audit v4 — `task_label` schema fix

The v3 log exposed the actual task-row schema:

```text
{
  'dataset': 'mdsc',
  'utt_id': '...',
  'split': 'dev',
  'task_label': '小度小度',
  'audio_relpath': None
}
```

So the phrase text is stored in `task_label`, not `label` / `label_text`.

v4 adds `task_label` as the highest-priority phrase field for:
- Core30 task rows,
- open-set task rows,
- tolerant raw-manifest parsing.

It still:
- loads Core30 from train+dev only,
- loads open-set only from dev,
- does not access generic_test audio/features.

Run:

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v4.py
python scripts/review_p6_unknown_exposure_audit_v4.py
```

Outputs:

```text
artifacts/p6_unknown_exposure_audit_v4/
```
