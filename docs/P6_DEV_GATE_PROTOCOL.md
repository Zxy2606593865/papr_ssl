# P6 — Teacher E0 / E0b development qualification

This bundle intentionally stops before `generic_test`.

It implements P6-01 through P6-06 only:

```text
P6-01  generic_enrollment -> prototype, primary K=1
P6-02  generic_dev_cal -> score/margin thresholds
P6-03  generic_dev_score -> qualification metrics
P6-04  absolute Gate
P6-05  E0b head-variant non-degradation Gate
P6-06  freeze Teacher/checkpoint/threshold/code state
```

`P6-07 generic_test` is deliberately not included here so test cannot be
accidentally accessed before the freeze.

## Frozen checkpoint policy

P5 produced 3 Attention checkpoints and 3 Mean checkpoints.

P6 must qualify one actual checkpoint, not an average imaginary model.

The representative checkpoint is selected before P6 scoring as:

```text
median selected-dev score among seeds 17 / 29 / 43
```

This explicitly avoids choosing the best seed.

## Generic protocol

Primary enrollment:

```text
K=1 per intent
30 intents -> 30 enrollment utterances
selection = deterministic stable hash from Core30 train
selection never examines embedding similarity or P6 scores
```

Known development data:

```text
Core30 dev (442)
-> deterministic class-stratified 50/50 split
-> generic_dev_cal_known
-> generic_dev_score_known
```

Unknown development data:

```text
Core30 open-set dev (1178)
-> deterministic 50/50 split
-> generic_dev_cal_unknown
-> generic_dev_score_unknown
```

No test row is touched.

## Threshold policy

Prototype decision:

```text
s1 = highest cosine similarity
s2 = second highest cosine similarity
margin = s1 - s2

ACCEPT iff:
    s1 >= score_threshold
    AND
    margin >= margin_threshold
otherwise REJECT
```

Thresholds are selected only on `generic_dev_cal`.

The calibration grid is derived from calibration-score quantiles. Among
threshold pairs satisfying all absolute Gates, choose lexicographically:

```text
1. max Macro F1
2. max Correct Accept
3. max Unknown Reject
4. min Wrong Intent
5. min Known Reject
```

If no threshold pair satisfies the Gates, the script writes the least-violating
diagnostic pair but returns nonzero and P6-06 cannot freeze.

## P6-04 absolute Gate

```text
Macro F1       >= 0.80
Correct Accept >= 0.80
Wrong Intent   <= 0.10
Known Reject   <= 0.20
Unknown Reject >= 0.85
```

Macro F1 is the 30-class known-intent Macro F1 after rejection. A rejected
known utterance contributes a false negative to its true class. Open-set
false accepts are evaluated separately by Unknown Reject / FAR.

## P6-05 E0b

The P5 Mean 64D head is the frozen head-variant baseline.

The P5 Attention 64D head is the mainline.

Both use the same:
- WavLM-large hidden_states[15]
- enrollment utterances
- cal/score partitions
- prototype rule
- threshold calibration algorithm

E0b passes when:

```text
Mean Macro-F1 - Attention Macro-F1 <= 0.02
```

This is a head-variant non-degradation Gate, not a 1024D-vs-64D compression
claim.

## Run

First confirm the open-set index path exists. Default:

```text
artifacts/p2_07/mdsc_policy_v2/
mdsc_core30_open_set_eval.index.jsonl
```

If your file name differs, pass `--open-index <actual path>`.

Then:

```powershell
python scripts/prepare_p6_protocol.py

python scripts/run_p6_dev_qualification.py --device cuda
```

Only if the development Gate passes:

```powershell
python scripts/freeze_p6_teacher.py
python scripts/audit_p6_dev_gate.py
```

Expected freeze:

```text
P6-06 STATUS: FROZEN / READY FOR P6-07
generic_test accessed: NO
```

Only after that should P6-07 code be added and generic_test opened once with
the already-frozen Teacher and thresholds.
