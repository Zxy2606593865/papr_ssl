# P6 Unknown Exposure Train-Side Data Audit

## Goal

Before adding an open-set training loss, first verify that the existing MDSC
training split contains enough **non-Core30** speech to act as unknown exposure.

Chinese terms:

- Unknown Exposure = 未知样本暴露训练
- Known = 已知短语
- Unknown = 未知短语
- Phrase-disjoint = 短语类别互斥
- Leakage = 数据泄漏

This step does **not** train a model.

## Frozen development policy

Candidate unknown exposure rows must satisfy:

```text
split == train
normalized phrase NOT in Core30
normalized phrase NOT in current open-set DEV phrase identities
```

The script deliberately does not use generic_test for candidate selection.

It therefore reports:

```text
generic_test used for selection: NO
generic_test audio/features: NO
```

A test-phrase overlap audit is deferred until the final model is frozen. It
must be descriptive only and must not be used to retune the model.

## Run

Normally:

```powershell
python scripts/audit_p6_unknown_exposure_candidates.py
python scripts/review_p6_unknown_exposure_audit.py
```

The first script attempts to auto-discover the full MDSC JSONL manifest.

If auto-discovery fails or picks the wrong file:

```powershell
python scripts/audit_p6_unknown_exposure_candidates.py `
  --mdsc-manifest path\to\full_mdsc_manifest.jsonl
```

## Outputs

```text
artifacts/p6_unknown_exposure_audit/
├── p6_unknown_exposure_audit.json
├── unknown_exposure_train_candidates.jsonl
└── unknown_exposure_phrase_summary.jsonl
```

## What to inspect

The important numbers are:

```text
candidate rows
candidate phrases
candidate speakers
Control / Dysarthria domain counts
phrase support >= 5 / 10 / 15
Core30 overlap = 0
open-set DEV phrase overlap = 0
```

Only after these numbers are reviewed should an open-set-aware Teacher
experiment be frozen.
