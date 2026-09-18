# H6-02 src-layout fix

Your actual package is:

```text
<repo>/src/papr_ssl/
```

Therefore H6 runtime must be installed at:

```text
src/papr_ssl/inference/
├─ __init__.py
└─ h6_personalized_runtime.py
```

Do NOT place it under:

```text
<repo>/papr_ssl/inference/
```

Run in this order:

```powershell
python scripts/check_h6_02_import.py
python scripts/run_h6_02_cached_runtime_demo.py
python scripts/audit_h6_02_runtime.py
```

Expected import diagnostic:

```text
papr_ssl loaded from: ...\src\papr_ssl\__init__.py
runtime module import: PASS
```
