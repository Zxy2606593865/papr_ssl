# P6 Unknown Exposure Audit v2 — Core30 Schema 修复

上一版本因以下错误失败：

```text
RuntimeError: expected 30 normalized Core30 phrases, got 0
```

这是 audit script 的 schema 解析错误，不是数据问题。

P2 task-index files 应通过项目现有的以下函数解析：

```python
papr_ssl.training.teacher.real_experiment.load_real_task_rows
```

不应假设 JSONL task index 会把短语文本直接存放在顶层 `label` 或 `label_text` 字段。

## 运行

使用新的 v2 输出目录，以免覆盖失败的 v1 artifact：

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v2.py
python scripts/review_p6_unknown_exposure_audit_v2.py
```

如果无法自动发现完整的 MDSC manifest，请显式传入：

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v2.py `
  --mdsc-manifest path\to\full_mdsc_manifest.jsonl
```

本流程不执行模型训练。`generic_test` 继续封存。
