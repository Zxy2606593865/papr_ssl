# P2-06C Safe SSL Batching Policy

## P2-06B measured result

Native right-padded batching was compared with exact-length singleton reference.

### WavLM
PASS on GSC and MDSC.

### W2v-BERT 2.0
PASS on GSC and MDSC after trimming each raw row to semantic length before the
shared feature-extractor call.

### Wav2Vec2
Frame counts matched exactly, but shorter-row feature values did not.

Observed examples:

```text
GSC short row
mean frame cosine ~= 0.127
pooled cosine     ~= 0.153

MDSC shorter row
mean frame cosine ~= 0.859
pooled cosine     ~= 0.927
```

The longest row in each padded batch remained essentially identical to the
singleton reference, which localizes the discrepancy to raw right-padding
interaction rather than hidden-layer selection or frame-count conversion.

## Frozen safe policy

```text
WavLM
  -> native padded batch

W2v-BERT 2.0
  -> trim variable raw rows
  -> one shared feature-extractor call
  -> one shared neural-model forward

Wav2Vec2
  -> group rows by exact semantic waveform length
  -> trim group tensor to that exact length
  -> one model forward per unique length
  -> restore row order
  -> feature-pad after the backbone
```

This avoids silently accepting numerically corrupted short-sample
representations.

## Performance implication

For GSC this remains reasonably efficient because the majority of speech files
have exactly 16,000 samples and can share large Wav2Vec2 batches.

For highly variable-length MDSC, exact-length grouping may create many small
Wav2Vec2 groups. That is accepted for correctness in P2. A more invasive
"per-sample convolutional frontend, batched transformer" optimization can be
considered later only if Wav2Vec2 remains a competitive Teacher candidate.
