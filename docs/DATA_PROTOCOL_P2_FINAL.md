# PAPR-SSL — P2 Final Data Protocol

Status: **FROZEN**

This document is the final P2 data-policy reference before P3 model training.

## 1. Scope

P2 defines dataset structure, task views, audio handling, batching safety, and
evaluation roles. It does not train the representation head, SCAF head, or
student model.

## 2. Public datasets

### GSC v2

Role:

- generic keyword / embedding sanity benchmark;
- clean standard-speech classification reference;
- model and pipeline regression testing.

Frozen task view:

- 35-class closed-set benchmark.

### MDSC

MDSC is treated as a **multi-use dysarthria speech resource**, not as a
wake-word-only dataset and not as a 3784-class phrase dataset.

The raw manifest remains unchanged. Task views are derived from it.

## 3. MDSC task views

### 3.1 MDSC-Core30 — primary public phrase benchmark

Purpose:

- dysarthria phrase-representation benchmark;
- backbone/layer/head comparison;
- Control vs Dysarthria analysis.

Definition:

- 30 high-coverage exact phrases;
- 10 canonical wake phrases;
- 20 common non-wake commands;
- surface normalization only (`<p>`, whitespace/case handling already approved);
- no semantic synonym merging.

Observed frozen counts:

```text
5083 utterances
30 classes

train  3756
dev     442
test    885

control     2763
dysarthria  2320
```

The class set is justified by strong **train-side speaker coverage**. Dev/test
are evaluation partitions and must not be used to redefine the class inventory.

### 3.2 MDSC-Command20 — command-only diagnostic

Purpose:

- test phrase representation without relying on wake-word semantics;
- provide a cleaner command-recognition diagnostic.

Observed frozen counts:

```text
2783 utterances
20 classes

train  2056
dev     242
test    485

control     1513
dysarthria  1270
```

### 3.3 MDSC-WWS10 — auxiliary personalization benchmark

Purpose:

- 1-shot personalized wake-word spotting;
- prototype / few-shot personalization analysis.

This is **not** the main product task and does **not** imply that the deployed
system must use wake-word activation.

Activation remains orthogonal to phrase recognition.

### 3.4 MDSC-Core30 open-set evaluation

All dev/test speech outside Core30 is reserved as open-set/rejection material.

Observed frozen counts:

```text
3533 utterances

dev   1178
test  2355

control     1177
dysarthria  2356
```

Important:

```text
The open-set pool is NOT a supervised SCAF class.
```

Unrelated long-tail utterances must not be collapsed into one metric-learning
class, because that would force semantically unrelated speech embeddings
together.

Allowed uses:

- open-set threshold calibration on dev;
- final rejection evaluation on test;
- hard-negative analysis;
- future auxiliary objectives.

## 4. Label semantics are task-dependent

The same utterance can play different roles in different task views.

Example:

```text
"打开空调"

WWS10 view      -> non-wake / negative
Core30 view     -> target class "打开空调"
Open-set view   -> not open-set because it belongs to Core30
```

Therefore labels such as `NON_WAKE` or `OPEN_SET` are **task-view labels**, not
global semantic labels stored back into the raw manifest.

## 5. Long-tail transcript policy

The thousands of MDSC long-tail transcripts are not discarded, but P2 does not
invent a new intent taxonomy for them.

Frozen policy:

- do not create thousands of SCAF classes;
- do not perform manual semantic merging in P2;
- do not rewrite the raw manifest;
- reserve them for rejection, hard negatives, and future auxiliary objectives.

## 6. Audio policy

### GSC

- model-side fixed 16000 samples;
- right-pad short utterances;
- reject speech longer than the frozen one-second policy where required by the
  current GSC task protocol.

### MDSC

- preserve the full utterance;
- stereo -> mono by channel mean when necessary;
- variable-length batches use right padding and waveform masks.

## 7. SSL batching policy

Frozen safe policy:

- WavLM: native padded batch;
- W2v-BERT 2.0: semantic trim before frontend/native batching;
- Wav2Vec2: group by exact semantic waveform length, forward each group, restore
  original order.

A correct frame mask alone does not guarantee padding-invariant SSL features.

## 8. Train/dev/test rule

- train: fitting / supervised learning;
- dev: model selection, calibration, thresholds;
- test: sealed final evaluation.

The test split must not be used to choose classes, thresholds, checkpoints, or
model variants.

## 9. P2 closure criterion

P2 is CLOSED only when:

1. the full unit-test suite passes;
2. `mdsc_task_policy_v2_summary.json` reports `overall = PASS`;
3. `scripts/validate_p2_final_closure.py` reports `P2 FINAL STATUS: PASS / CLOSED`.

After closure, model development continues in P3 without reopening the frozen
data semantics unless a documented data bug is discovered.
