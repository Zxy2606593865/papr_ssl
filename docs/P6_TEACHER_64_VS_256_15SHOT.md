# P6 — 64D vs 256D Teacher, same 15-shot personalized/open-set protocol

This experiment is the application-level follow-up to the embedding-dimension sweep.

Closed-set result so far:

```text
canonical 64D Teacher   ~0.893376 Macro-F1
256D candidate          ~0.912667 mean Macro-F1
```

The 256D Teacher is promoted for Project 1 only if it also improves the real
15-shot personalized/open-set metrics.

Project 2 keeps the canonical 64D Teacher regardless of this result.

## Representative 256D checkpoint

The exporter reads seeds 17/29/43 and chooses the median selected-dev-score
checkpoint. It does not cherry-pick the best single seed.

With the current results, this is expected to be seed 43, but the script derives
that from the result files.

## Protocol

Same as the strongest existing 64D downstream evaluation:

```text
15-shot enrollment
mean prototype for intent prediction
retain all 15 enrollment embeddings
enrollment-consistency features
balanced logistic rejector
20 repeated DEV splits
calibration half / score half
generic_test sealed
```

## Run

```powershell
python scripts/export_teacher256_15shot_embeddings.py --device cuda
python scripts/run_teacher256_vs64_15shot.py
python scripts/audit_teacher256_vs64_15shot.py
```

Outputs go to:

```text
artifacts/p6_teacher_256_15shot/
```

No existing 64D artifact is overwritten.
