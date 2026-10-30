# Demo-04D — 时长方向符号审计

## 本步骤的目的

The current frozen H6 duration feature is:

```text
abs(log(T_query / median(T_support)))
```

该特征有一个重要限制：

```text
query is shorter by 28%
```

and

```text
query is longer by 39%
```

can produce the same absolute log-ratio magnitude.

对于真实 hard negative：

```text
Unknown:    大家好
Registered: 大家好 我叫王灏
```

query **更短**这一方向信息可能比单独使用时长差异幅度更有价值。

## 现有证据已排除简单的绝对时长阈值

Correct known samples include absolute duration-log-ratio values up to roughly:

```text
0.3808
```

while the `大家好` false accept is:

```text
0.3285
```

因此，类似以下规则：

```text
duration_log_ratio > 0.30 -> REJECT
```

也会误伤有效的 known 样本。

## Demo-04D

只测量、不拟合：

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

这仅用于诊断，不会改变 H6。

## 运行

在 `papr_ssl` 根目录运行：

```powershell
python scripts/audit_demo04d_signed_duration.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2" `
  --predictions artifacts\demo_04a_wanghao_runtime_v2\predictions.jsonl `
  --memory artifacts\demo_04a_wanghao_runtime_v2\wanghao_user_memory.pt `
  --output-dir artifacts\demo_04d_signed_duration_audit
```

## 输出

```text
artifacts/demo_04d_signed_duration_audit/
├── signed_duration_audit.csv
└── summary.json
```

重点查看以下行：

```text
UNKNOWN FALSE ACCEPT DIRECTION CHECK

Q/S = ?
signed_log = ?

known correct equally/more short = ?/24
```

如果 prefix false accept 明显短于所有有效 known query，则值得评估**带符号的时长不足特征**。

如果有效 known query 在相同负方向上有明显重叠，那么仅靠时长方向也不够；应改为探索更丰富的 prefix 专属时序特征，而非手写阈值。
