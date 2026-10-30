# P6 Unknown Exposure Audit v3 — 仅限 Train/Dev 的 Task-Index 修复

v2 audit 因调用以下代码而失败：

```python
load_real_task_rows(..., splits=("train", "dev", "test"))
```

但现有 PAPR-SSL helper 会在开发阶段有意阻止访问 `test`：

```text
ValueError: real P3 execution may load train/dev only
```

这是保护机制，并非数据错误。

为了恢复 30 个 Core30 短语身份，`train + dev` 已足够。因此，v3 只将该调用改为：

```python
load_real_task_rows(..., splits=("train", "dev"))
```

Open-set task index 仍只通过 `splits=("dev",)` 读取。

不加载任何 generic test audio、embeddings 或 labels。

运行：

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v3.py
python scripts/review_p6_unknown_exposure_audit_v3.py
```

输出写入：

```text
artifacts/p6_unknown_exposure_audit_v3/
```
