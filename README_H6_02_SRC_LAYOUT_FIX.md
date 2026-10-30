# H6-02 src-layout 修复

项目实际的 package 路径为：

```text
<repo>/src/papr_ssl/
```

因此，H6 runtime 必须安装在：

```text
src/papr_ssl/inference/
├─ __init__.py
└─ h6_personalized_runtime.py
```

不要将其放在：

```text
<repo>/papr_ssl/inference/
```

按以下顺序运行：

```powershell
python scripts/check_h6_02_import.py
python scripts/run_h6_02_cached_runtime_demo.py
python scripts/audit_h6_02_runtime.py
```

预期的 import 诊断信息：

```text
papr_ssl loaded from: ...\src\papr_ssl\__init__.py
runtime module import: PASS
```
