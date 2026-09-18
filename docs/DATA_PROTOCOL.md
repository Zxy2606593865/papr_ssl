# PAPR-SSL Data Protocol

Status: **FROZEN — P2 PASS / CLOSED**

## 1. Principle

P2 standardizes data access, manifests, audio handling, safe batching, task views, and evaluation roles. Raw audio and raw manifests are not rewritten. Task-specific labels are derived views.

## 2. GSC v2

- Role: generic standard-speech KWS / embedding benchmark.
- Task: **GSC-35** closed-set classification.
- Audio: 16 kHz; model-side length 16000 samples; short clips are right-padded; valid lengths are retained for masking.

## 3. MDSC

MDSC is treated as a multi-use Mandarin dysarthria speech resource, not as a 3784-class phrase dataset.

After approved surface normalization:

```text
normalized phrases: 3784
cross-domain phrases: 2549
```

No semantic synonym merging is performed.

### 3.1 MDSC-Core30 — primary public phrase benchmark

```text
utterances: 5083
classes:    30
train/dev/test: 3756 / 442 / 885
control/dysarthria: 2763 / 2320
```

Purpose: dysarthria phrase representation, SSL backbone/layer/head comparison, and Control-vs-Dysarthria analysis. The class inventory is justified by stable train-side speaker coverage.

### 3.2 MDSC-Command20 — command-only diagnostic

```text
utterances: 2783
classes:    20
train/dev/test: 2056 / 242 / 485
```

Purpose: evaluate ordinary command representation without relying on wake-word semantics.

### 3.3 MDSC-WWS10 — auxiliary personalization benchmark

Purpose: 1-shot personalized wake-word spotting and prototype/few-shot personalization analysis. It does not define the final product activation mechanism.

### 3.4 MDSC open-set / rejection evaluation

```text
utterances: 3533
dev/test: 1178 / 2355
control/dysarthria: 1177 / 2356
```

Open-set utterances are **not** collapsed into one supervised SCAF class. They are used for dev-side threshold calibration, sealed test-side rejection evaluation, hard-negative analysis, or future auxiliary objectives.

## 4. Task-dependent labels

Example:

```text
"打开空调"
WWS10   -> non-wake / negative
Core30  -> target class "打开空调"
OpenSet -> not open-set because it belongs to Core30
```

Therefore NON_WAKE and OPEN_SET are task-view labels, not global labels written into the raw manifest.

## 5. MDSC audio policy

- Preserve the full utterance.
- Stereo -> mono by channel mean in memory.
- Variable-length batches use right padding and waveform masks.

## 6. Safe SSL batching

- WavLM: native padded batch.
- W2v-BERT 2.0: semantic trim before frontend/native batching.
- Wav2Vec2: group by exact semantic waveform length, forward each group, then restore original order.

## 7. Split policy

```text
train -> fitting / supervised learning
dev   -> model selection / threshold calibration
test  -> sealed final evaluation
```

## 8. Final P2 task map

```text
GSC-35          -> generic standard-speech embedding benchmark
MDSC-Core30     -> primary dysarthria phrase representation benchmark
MDSC-Command20  -> command-only diagnostic
MDSC-WWS10      -> auxiliary 1-shot personalization benchmark
MDSC long-tail  -> open-set / rejection evaluation
```

After P2-10, P2 is reopened only for a documented data bug.
