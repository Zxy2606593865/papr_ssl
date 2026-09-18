# P2-07C Personalized MDSC Protocol Integrity

This is the final P2 gate before Teacher experiments.

The frozen P2-07B task assumes speaker-specific enrollment prototypes. Therefore
the dataset must actually support that protocol at the speaker level.

The validator checks:

```text
8 personalized dysarthria speakers
dev/test speaker disjointness
one split per speaker
all 10 wake words present in enrollment for every speaker
all 10 wake words present in eval for every speaker
non-wake eval trials present for every speaker
```

It also reports the observed number of enrollment/eval repetitions per wake
without hard-coding those counts into the feasibility gate.

A PASS means the public dataset and frozen task semantics are sufficient to
construct the later protocol:

```text
speaker enrollment
    -> 10 speaker-specific wake prototypes
eval wake
    -> positive trials
eval non-wake
    -> false-alarm / rejection trials
```

Threshold tuning remains development-only; test remains locked.
