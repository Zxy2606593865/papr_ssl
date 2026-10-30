# P6 未知类别暴露审计 v5

v4 虽然成功生成了结果，但该结果在科学上无效：

```text
manifest = gsc_v2.jsonl
84,843 train rows
domain = standard
```

这是 Google Speech Commands，而不是 MDSC。

由于 open-set 任务索引对拒识类别使用通用任务标签，报告中也只出现一个 open-set DEV 短语。必须将每条 open-set DEV `utt_id` 映射回完整 MDSC manifest，恢复真实未知短语。

v5 修复了这两个问题：

1. Auto-discovery requires at least 5,000 rows explicitly identified as MDSC.
2. Open-set DEV real phrase identities are recovered from the full MDSC
   manifest using the 1,178 open-set DEV utterance IDs.
3. Candidate exposure is restricted to MDSC train only.
4. Core30 and open-set DEV phrase identities are excluded.
5. `generic_test` 继续保持封存。

运行：

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v5.py
python scripts/review_p6_unknown_exposure_audit_v5.py
```

如果自动发现无法选定 manifest，请列出可能的 MDSC manifest：

```powershell
Get-ChildItem artifacts -Recurse -Filter *.jsonl |
  Where-Object {$_.FullName -match 'mdsc'} |
  Select-Object FullName
```

然后显式传入正确的完整 MDSC manifest：

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v5.py `
  --mdsc-manifest "artifacts\...\mdsc....jsonl"
```
