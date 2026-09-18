# Minimal Temporal DTW Reranker

## Will this make the model too redundant?

No. This experiment is explicitly designed to avoid that.

It does **not** add a second encoder.

The existing Project-1 Teacher remains:

```text
Audio
-> WavLM-large hidden_states[15]
-> Attention
-> 256D global embedding
```

The DTW branch reuses the **same WavLM layer-15 frame features** and even reuses
the already-trained Teacher `1024 -> 256` projection frame-by-frame.

So the added branch is only:

```text
same WavLM[15] frames
-> same trained 1024->256 projection
-> temporal mean-pool every 3 frames
-> DTW
```

There is no second WavLM, no second CNN, and no second Transformer.

## Why DTW is kept cheap

The global 256D prototype system remains the primary classifier.

Only after the global system generates its top-3 candidate classes do we run
DTW.

For each class we keep only 2 temporal enrollment templates.

Therefore each query requires only:

```text
3 candidate classes x 2 templates = 6 DTW comparisons
```

instead of:

```text
30 classes x 15 enrollment utterances = 450 DTW comparisons
```

That is a 75x reduction in the number of DTW template comparisons.

## Architecture

```text
                         WavLM-large[15]
                         /             \
                        /               \
                       v                 v
              Attention pooling     same frame projection
                       |                 |
                       v                 v
                    256D global      [T',256] temporal
                       |                 |
                       v                 |
              30-class prototype         |
                       |                 |
                       v                 |
                  top-3 classes ---------+
                                         |
                                         v
                             DTW to 2 templates/class
                                         |
                                         v
                               conservative score fusion
                                         |
                                         v
                              consistency + rejector
```

## Frozen DTW choices

To avoid an uncontrolled search:

```text
temporal downsample factor = 3
top-k candidate classes = 3
templates/class = 2
DTW local cost = 1 - cosine
Sakoe-Chiba band ratio = 0.25
fusion lambda grid = {0.50, 0.75, 1.00}
```

Lambda is selected using **known calibration data only**. Ties prefer the larger
lambda, meaning the system remains closer to the existing global Teacher unless
DTW provides a measurable benefit.

## Run order

### 1. Build temporal features

```powershell
python scripts/build_temporal_dtw_features.py --device cuda
```

### 2. Precompute DTW evidence

```powershell
python scripts/precompute_dtw_rerank.py
```

This is the slowest CPU stage. It is computed once and reused by all repeated
splits.

### 3. Run 20-split application ablation

```powershell
python scripts/run_temporal_dtw_ablation.py
```

### 4. Audit

```powershell
python scripts/audit_temporal_dtw_ablation.py
```

## Promotion rule

Do not keep DTW merely because it is academically interesting.

Keep it only if it improves the 20-split application metrics or Gate stability,
especially:

```text
Macro-F1
Correct Accept
Unknown Reject
Gate pass rate
```

without materially increasing Wrong Intent.

If DTW does not help, remove it and keep the simpler 256D Teacher.

## Project separation

Project 1 / Agent:
- DTW is allowed because inference is server-side.

Project 2 / hardware:
- canonical 64D path remains unchanged.
- this DTW branch is NOT automatically inherited by the edge Student.

generic_test remains sealed.
