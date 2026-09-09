# PAPR-SSL Evaluation Protocol

## Purpose

This protocol prevents test-set leakage while comparing SSL backbone and exact
hidden-state index under the same waveform, dataset, split, Mean DR, 64D, SCAF,
training budget, threshold policy, and metrics. This document defines contracts;
it contains no fabricated result.

## Development partitions

Large public datasets use four development partitions:

| Partition | Sole responsibility |
|---|---|
| `generic_train` | Train Mean DR and SCAF under the fixed training budget. |
| `generic_enrollment` | Build development prototypes for each few-shot condition. |
| `generic_dev_cal` | Select score and top1-top2 margin thresholds. |
| `generic_dev_score` | Select backbone, exact hidden-state index, checkpoint, and future DR variants. |

An utterance belongs to exactly one partition. Split construction should isolate
speaker, recording, and session whenever dataset metadata permits. All split
manifests and hashes must be frozen before comparison runs.

`generic_test` must not be used to select a model, backbone, hidden layer,
checkpoint, threshold, optimizer, training budget, or hyperparameter. Repeated
inspection of test results during development is prohibited.

## Fair backbone and layer comparison

Each comparison fixes:

- identical waveform/window contract and sample IDs;
- identical train/enrollment/calibration/scoring split;
- identical Mean DR implementation and 64D output;
- identical SCAF implementation and `K=3`, margin `0.2 rad`, scale `30`;
- identical optimizer policy, step/epoch budget, early-stop rule, and seeds;
- identical prototype construction and open-set decision protocol;
- identical checkpoint-selection metric and evaluation implementation.

Only one independent variable changes per comparison: SSL backbone or exact
`outputs.hidden_states` index. Invalid real-model indices are recorded as
`INVALID_LAYER`; they are never clamped, substituted, or mapped to the final
layer.

## Threshold calibration

Score and top1-top2 margin thresholds are determined only on `generic_dev_cal`.
The complete threshold policy is frozen before `generic_dev_score` selection and
before any `generic_test` access. Test-domain labels cannot be used to recalibrate
thresholds unless a separate, predeclared cross-domain calibration experiment is
reported outside the primary frozen-threshold result.

## Final generic test

`generic_test` may be opened only after the backbone, exact hidden-state index,
DR, checkpoint, evaluation code, and threshold policy are frozen. It is split
internally into:

- `test_support`: prototype enrollment for declared 1-shot, 5-shot, and 10-shot
  conditions;
- `known_query`: utterances from enrolled intents;
- `unknown_query`: utterances from intents absent from enrollment.

`test_support` and `known_query` must never share an utterance. Prefer isolation
by speaker, recording, and session. Each shot condition has a frozen support
manifest. Development thresholds are applied unchanged; calibration on
`generic_test` is forbidden.

## Within-domain and cross-domain tests

Within-domain evaluation trains on Dataset A and evaluates on a held-out,
sealed Dataset A test partition.

Cross-domain evaluation trains the Teacher on Dataset A and evaluates on Dataset
B. The Teacher is not retrained or fine-tuned on Dataset B. A predeclared
few-shot protocol may build Dataset B test prototypes from `test_support`, but
the threshold policy remains frozen before Dataset B labels/results are opened.
Within-domain and cross-domain results must be reported separately.

## Three evidence levels

Final claims require three distinct evidence levels:

1. Generic public benchmark.
2. Atypical/dysarthric benchmark.
3. Target-user sealed test.

Strong generic results do not establish that a PAPR target-user profile has
passed. Target-user evidence remains a separate sealed gate.

## Metric contract

Report counts and denominators with every rate. At minimum:

| Metric | Definition |
|---|---|
| Known Macro F1 | Unweighted mean of per-intent F1 over known intents. |
| Correct Accept Rate | Known queries accepted with the correct intent / all known queries. |
| Wrong Intent Rate | Known queries accepted as an incorrect intent / all known queries. |
| Known Reject Rate | Known queries rejected / all known queries. |
| Unknown Reject Rate | Unknown queries rejected / all unknown queries. |
| FAR | Unknown queries incorrectly accepted / all unknown queries. |
| FRR | Known queries incorrectly rejected / all known queries. |
| ACC/DET @ 1% FAR | Accuracy/detection rate at an operating point constrained to 1% FAR. |
| ACC/DET @ 5% FAR | Accuracy/detection rate at an operating point constrained to 5% FAR. |
| AUC | Area under the declared detection ROC curve. |

The known-query accounting identity is checked:

```text
Correct Accept Rate + Wrong Intent Rate + Known Reject Rate = 1
```

Embedding diagnostics include mean intra-class cosine similarity, mean
inter-class cosine similarity or distance, and the full top1-top2 margin
distribution. Report 1-shot, 5-shot, and 10-shot results independently, with
confidence intervals or repeated support draws when the frozen protocol defines
them.

## Reproducibility record

Every reported run must retain config, code revision, immutable model revision,
split-manifest hashes, seed, checkpoint hash, threshold values, support-shot
manifest, metric implementation version, and environment/dependency lock. A
model-selection table records rejected candidates without opening test results.
