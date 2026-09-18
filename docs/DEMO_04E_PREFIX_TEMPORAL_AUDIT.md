# Demo-04E — Prefix / Endpoint Temporal Audit

## What Demo-04D established

The false accept:

```text
Unknown: 大家好
Top1:    大家好 我叫王灏
Q/S = 0.720
```

is shorter than the enrolled phrase.

But three correct Known queries are equally or even more short:

```text
known correct equally/more short = 3/24
```

Therefore:

```text
signed duration alone
```

is not a safe decision rule.

## Next hypothesis

A true prefix hard negative should have a different **temporal shape** from a
legitimate faster/shorter pronunciation of the full phrase.

For the top-1 enrolled intent, Demo-04E measures:

```text
full_dtw_sim
prefix_dtw_sim
suffix_dtw_sim
prefix_minus_suffix

start_cosine
end_cosine
start_minus_end
```

Interpretation:

```text
prefix_minus_suffix >> 0
```

means the query resembles the beginning of the enrolled phrase more than its
ending.

```text
start_minus_end >> 0
```

means the start endpoint is much more compatible than the end endpoint.

The important comparison is NOT against all queries only. It explicitly prints
the valid Known queries that are equally or more short than the hard negative.

## Why this is better than another hand-written duration threshold

The current H6 DTW produces a high score for:

```text
大家好
vs
大家好 我叫王灏
```

so duration alone cannot tell whether the query is:
- a faster full utterance, or
- only the prefix of the enrolled utterance.

Endpoint/prefix diagnostics test this missing distinction.

## Run

From `papr_ssl` root:

```powershell
python scripts/audit_demo04e_prefix_temporal.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2" `
  --predictions artifacts\demo_04a_wanghao_runtime_v2\predictions.jsonl `
  --memory artifacts\demo_04a_wanghao_runtime_v2\wanghao_user_memory.pt `
  --output-dir artifacts\demo_04e_prefix_temporal_audit
```

No training, threshold fitting, runtime changes, or generic_test access.

## Decision after this audit

If the hard negative has a clear prefix/endpoint signature that the equally-short
Known controls do not have, the next step is to prototype a **prefix-risk
feature** in the decision layer.

If the distributions still overlap strongly, do not add another heuristic.
The better next step is to collect more real hard negatives and fit/evaluate a
small open-set decision feature set on session-separated real-user data.
