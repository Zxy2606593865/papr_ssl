# P6 15-shot Enrollment-Consistency Rejector

## Motivation

The previous lightweight rejector used only:

```text
top-1 cosine
top1-top2 margin
and simple quadratic terms
```

It did not improve the full Gate.

However, a 15-shot system contains much more information than one mean
prototype. A true known query should often be consistently similar to MANY of
the 15 enrollment utterances, while an unknown query may match the centroid or
one enrollment sample by chance.

This experiment therefore uses the full enrollment similarity distribution.

## Frozen

```text
WavLM-large hidden_states[15]
AttentionDR -> 64D
15-shot enrollment set
mean-prototype intent prediction
generic_test sealed
```

## New rejector evidence

For the class predicted by the unchanged mean-prototype classifier, compute
cosine similarity to all 15 enrollment embeddings.

Features include:

```text
top1 centroid score
top1-top2 margin
mean similarity to 15 enrollments
standard deviation
minimum
25th percentile
median
75th percentile
maximum
top-3 mean
top-5 mean
bottom-3 mean
centroid minus enrollment mean
max minus min
```

A tiny balanced logistic regression is trained on each calibration half only.
The probability threshold is selected on calibration only and tested on the
held-out score half.

## Run

```powershell
python scripts/export_p6_15shot_consistency_embeddings.py --device cuda
python scripts/run_p6_15shot_consistency_rejector.py
python scripts/audit_p6_15shot_consistency_rejector.py
```

## Decision after this experiment

If this materially improves repeated-split full-Gate pass rate, keep the
Teacher frozen and use enrollment-distribution evidence in the personalized
decision layer.

If it still remains near 0/20 Gate passes, decision-layer tuning is exhausted
enough for this stage. The next justified step is open-set-aware representation
training rather than more rejector complexity.
