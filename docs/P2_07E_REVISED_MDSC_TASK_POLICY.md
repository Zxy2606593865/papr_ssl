# P2-07E — Revised MDSC Task Policy

MDSC is no longer treated as a wake-word-only dataset.

## Frozen public task views

### 1. MDSC-Core30

Thirty exact phrases with the strongest speaker coverage:

- 10 canonical wake phrases
- 20 common commands

Purpose:
- dysarthria phrase-representation benchmark;
- backbone/layer/head comparison;
- Control vs Dysarthria analysis.

No semantic synonym merging is performed.

### 2. MDSC-Command20

The 20 non-wake commands from Core30.

Purpose:
- command-only diagnostic;
- prevents the main phrase analysis from depending on wake-word semantics.

### 3. MDSC-WWS10

Existing one-shot personalized wake-word protocol.

Purpose:
- auxiliary personalization benchmark only.

It does not define the final product activation mechanism.

### 4. MDSC Core30 open-set evaluation

All dev/test utterances whose task-normalized transcript is outside Core30 are
rejection trials.

Important:

```text
They are NOT collapsed into one SCAF training class.
```

The long-tail set is multimodal. Treating thousands of unrelated phrases as one
metric-learning class would force unrelated speech embeddings together.

These utterances are reserved for:
- open-set threshold calibration/evaluation;
- hard-negative analysis;
- future auxiliary representation objectives.

## Frozen boundary

P2 does not create a new semantic intent taxonomy for the long-tail transcripts.
That would require a separate annotation protocol and is outside the current
public-data engineering scope.
