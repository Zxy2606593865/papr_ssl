# P6 Unknown Exposure Audit v4 — `task_label` Schema 修复

v3 日志显示了实际 task-row schema：

```text
{
  'dataset': 'mdsc',
  'utt_id': '...',
  'split': 'dev',
  'task_label': '小度小度',
  'audio_relpath': None
}
```

因此，短语文本存储在 `task_label`，而非 `label` / `label_text`。

v4 将 `task_label` 设为以下场景中优先级最高的短语字段：
- Core30 task rows,
- open-set task rows,
- tolerant raw-manifest parsing.

同时仍然：
- loads Core30 from train+dev only,
- loads open-set only from dev,
- does not access generic_test audio/features.

运行：

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v4.py
python scripts/review_p6_unknown_exposure_audit_v4.py
```

输出：

```text
artifacts/p6_unknown_exposure_audit_v4/
```
