# P3-08 — Machine-readable Run Manifest

## Goal

Every completed P3 seed run must leave one machine-readable provenance record:

```text
run_manifest.json
```

It records the exact task/data identity, SSL revision/layer/cache, Teacher
architecture, seed, sampler, optimization budget, metrics/checkpoint artifacts,
selection result, code revision, environment, and sealed-test status.

## Per-seed run layout

```text
seed_0017/
├── checkpoints/
├── metrics.jsonl
├── checkpoint_hashes.jsonl
├── selected_checkpoint.json
└── run_manifest.json
```

The manifest records paths plus SHA-256 hashes instead of duplicating large
binary artifacts.

## Experiment fingerprint

`experiment_fingerprint` excludes seed and selected-checkpoint identity. It
therefore identifies the non-seed experimental condition:

```text
task view + task-index hash
backbone model_id + exact revision + layer
Mean DR + 64D + SCAF contract
sampler composition
optimization budget
```

Seed 17 / 29 / 43 must share one experiment fingerprint.

## Run fingerprint

`run_fingerprint` additionally includes:

```text
seed
selected checkpoint SHA-256
```

so each seed run has a different run identity.

## Required integrity checks

P3-08 fails if:

- sampler seed differs from Teacher config seed;
- feature-cache model/revision/layer differs from Teacher config;
- selected checkpoint SHA-256 no longer matches;
- P3-07 selection did not use `generic_dev_score`;
- metric definition is not `prototype_macro_f1`;
- P3-07 reports any generic-test access;
- a run manifest already exists and would be overwritten.

## Machine-readable provenance

The manifest includes:

```text
task-view and task-index SHA-256
Teacher config file SHA-256
cache manifest/index SHA-256
metrics.jsonl SHA-256
checkpoint_hashes.jsonl SHA-256
selected_checkpoint.json SHA-256
selected checkpoint SHA-256
Git commit/branch/dirty state when available
Python / PyTorch / CUDA / platform / device
```

## Manifest self-hash

The semantic payload is protected by:

```text
manifest_sha256
```

`verify_run_manifest()` recomputes it and detects later edits.

## Multi-seed experiment manifest

After all seed runs have a `run_manifest.json`, P3-08 can create:

```text
experiment_manifest.json
```

It verifies that all seed runs share the exact same non-seed
`experiment_fingerprint`.

This catches accidental changes such as:

```text
seed 17 -> layer 24
seed 29 -> layer 18
seed 43 -> layer 24
```

which would otherwise no longer be a valid multi-seed repetition.

## P3-08 Gate

PASS requires:

1. one immutable `run_manifest.json` per completed seed run;
2. exact Teacher config/cache identity agreement;
3. sampler seed equals run seed;
4. data/config/metrics/selection/checkpoint hashes are recorded;
5. selected checkpoint SHA-256 verifies;
6. generic test remains `sealed_not_accessed`;
7. run-manifest self-hash verifies;
8. multi-seed runs share one experiment fingerprint.
