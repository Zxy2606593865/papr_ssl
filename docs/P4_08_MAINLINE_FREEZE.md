# P4-08 — Freeze the Single Teacher Mainline

P4-08 performs no new model training and no new ranking.

It consumes:
- P4-06 selected Teacher feature source
- P4-07 resource-cost record

and freezes exactly one P5 mainline:

```text
Backbone:
microsoft/wavlm-large

revision:
c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c

outputs.hidden_states index:
15

hidden dimension:
1024
```

P4 result used for the freeze:

```text
3-seed mean generic_dev_score = 0.886472...
sample std                    = 0.006107...
95% Student-t CI              = [0.871301..., 0.901644...]
```

The other two Backbone winners are retained only as ablation baselines.

## Run

```powershell
python scripts/freeze_p4_08_mainline.py
python scripts/audit_p4_08_mainline.py
```

Outputs:

```text
artifacts/p4_runs/p4_08_mainline/
├── p4_08_mainline_lock.json
└── p5_input_contract.json
```

## P5 handoff

P5 keeps the P4-08 Backbone and hidden-state source fixed and compares only:

```text
Mean DR
vs
Attention DR
```

with seeds:

```text
17, 29, 43
```

Important: current P4 caches are masked-mean `[D]` caches. Attention DR requires
frame-level `[T,D]` features, so P5 must materialize a frame-level cache only
for the frozen WavLM-large `hidden_states[15]` finalist.

No other Backbone is reopened in the P5 mainline.
generic_test remains sealed.
