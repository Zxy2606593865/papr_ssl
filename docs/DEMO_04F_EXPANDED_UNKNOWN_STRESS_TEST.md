# Demo-04F — Expanded Wanghao Unknown Stress Test

## Why this is the next step

Demo-04E did **not** produce a robust prefix feature:

```text
Unknown false accept:
prefix-suffix DTW = -0.0019
start-end cosine  = -0.1033
```

Both values overlap the broader Known / Unknown distributions.

The one interesting observation is that, among the five equally-or-more-short
Known controls printed by Demo-04E, the hard negative has a more negative
`start-end` value. But there is only:

```text
1 hard-negative sample
```

That is not enough to create a production rule.

Before changing H6, use the data that already exists.

## Existing confirmed unregistered Wanghao intents

The toolkit contains human-confirmed intents that are NOT among the five
registered Demo intents, including:

```text
greet
see_you_again
coffee_ready
hold_on
make_coffee
michael_jordan
teacher
```

Demo-04F evaluates **all available confirmed samples** from every such
unregistered intent against the exact frozen Wanghao-v2 memory.

The main question is:

```text
Does "大家好" -> Self-introduction false accept repeat across all greet samples?
```

If it repeats systematically, we have evidence for a real semantic-prefix
failure mode.

If only one sample fails and the other greet samples reject correctly, avoid
designing a special rule around one observation.

## Run

From `papr_ssl` root:

```powershell
python scripts/run_demo04f_expanded_unknown.py `
  --toolkit-root "..\papr_audio_toolkit" `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2" `
  --memory artifacts\demo_04a_wanghao_runtime_v2\wanghao_user_memory.pt `
  --output-dir artifacts\demo_04f_unknown_expansion
```

## Outputs

```text
artifacts/demo_04f_unknown_expansion/
├── predictions.jsonl
├── per_intent_summary.csv
└── result.json
```

## Interpretation

This is still an engineering stress-test, not an independent benchmark:
- one real user;
- human-confirmed intents;
- several unregistered intents have only one source recording;
- no session-separated collection protocol.

No threshold or policy is changed in this step.
