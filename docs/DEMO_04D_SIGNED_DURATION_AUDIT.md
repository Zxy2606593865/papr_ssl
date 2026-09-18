# Demo-04D — Signed Duration-Direction Audit

## Why this step exists

The current frozen H6 duration feature is:

```text
abs(log(T_query / median(T_support)))
```

This has an important limitation:

```text
query is shorter by 28%
```

and

```text
query is longer by 39%
```

can produce the same absolute log-ratio magnitude.

For the real hard negative:

```text
Unknown:    大家好
Registered: 大家好 我叫王灏
```

the fact that the query is **shorter** may be more informative than duration
mismatch magnitude alone.

## Existing evidence already rules out a simple absolute-duration threshold

Correct known samples include absolute duration-log-ratio values up to roughly:

```text
0.3808
```

while the `大家好` false accept is:

```text
0.3285
```

Therefore a rule such as:

```text
duration_log_ratio > 0.30 -> REJECT
```

would also hit valid known samples.

## Demo-04D

Measure, without fitting:

```text
signed_duration_log_ratio = log(T_query / median(T_support))
query_support_ratio       = T_query / median(T_support)
```

Interpretation:

```text
signed < 0
→ query shorter than enrolled phrase

signed > 0
→ query longer than enrolled phrase
```

This is a diagnostic only. It does not change H6.

## Run

From the `papr_ssl` root:

```powershell
python scripts/audit_demo04d_signed_duration.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2" `
  --predictions artifacts\demo_04a_wanghao_runtime_v2\predictions.jsonl `
  --memory artifacts\demo_04a_wanghao_runtime_v2\wanghao_user_memory.pt `
  --output-dir artifacts\demo_04d_signed_duration_audit
```

## Output

```text
artifacts/demo_04d_signed_duration_audit/
├── signed_duration_audit.csv
└── summary.json
```

The key lines are:

```text
UNKNOWN FALSE ACCEPT DIRECTION CHECK

Q/S = ?
signed_log = ?

known correct equally/more short = ?/24
```

If the prefix false accept is substantially shorter than every valid known query,
then a **signed shortfall feature** becomes worth evaluating.

If valid known queries overlap strongly in the same negative direction, duration
direction alone is also insufficient and we should move to a richer prefix-specific
temporal feature rather than a hand-written threshold.
