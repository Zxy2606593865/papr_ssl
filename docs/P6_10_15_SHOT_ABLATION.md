# P6 Enrollment Shot Ablation: 10-shot vs 15-shot

This experiment asks one narrow question:

> Can the current frozen Teacher pass the application Gate simply by using
> more REAL enrollment utterances per intent?

No Teacher retraining is performed.

## Frozen

```text
Backbone   microsoft/wavlm-large
Layer      outputs.hidden_states[15]
Head       AttentionDR -> 64D
Checkpoint representative median-seed checkpoint
Decision   score + top1-top2 margin
Gate       unchanged
Dev split  unchanged
Test       sealed
```

## Changed factor

```text
10 real enrollment utterances / intent
vs
15 real enrollment utterances / intent
```

Both use:

```text
all N embeddings
-> arithmetic mean
-> L2 normalize
-> one mean prototype per intent
```

The 10-shot set is a strict subset of the 15-shot set.

This is intentionally simpler than adding K=2/K=3 clustering. If mean
prototype already passes at 10 or 15 shots, there is no need to introduce
subprototype complexity yet.

## Important interpretation

This remains a *generic public-data development ablation*. It is not yet the
final same-user personalized benchmark.

Also, `generic_dev_score` has already been observed in earlier P6 iterations.
Therefore it is now development data. The still-unseen public `generic_test`
remains reserved for the final frozen evaluation.

## Run

```powershell
python scripts/run_p6_10_15_shot_ablation.py --device cuda
python scripts/audit_p6_10_15_shot_ablation.py
```

The script embeds open-set DEV only once, then evaluates both shot counts with
exact empirical threshold calibration on `generic_dev_cal`.
