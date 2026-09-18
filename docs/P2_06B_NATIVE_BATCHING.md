# P2-06B Native Batched Masking

## Source inspection result

The current project wrappers already contain native padded batching for:

- Wav2Vec2: `forward(waveforms, sample_mask=None)`
- WavLM: `forward(waveforms, sample_mask=None)`

Both pass `sample_mask` into the Hugging Face model as `attention_mask` and
convert sample validity to selected-feature validity with the model helper.

W2v-BERT 2.0 previously lacked this input-side mask contract.

## W2v-BERT 2.0 native policy

Given:

```text
waveforms   [B,Smax]
sample_mask [B,Smax]
```

the wrapper derives semantic raw lengths:

```text
Li = sum(sample_mask[i])
```

and constructs one variable-length Python list:

```text
[
  waveform[0, :L0],
  waveform[1, :L1],
  ...
]
```

This list is passed to `AutoFeatureExtractor` once with `padding=True`.
The feature extractor creates:

```text
input_features [B,Tmax,F]
attention_mask [B,Tmax]
```

and the W2v-BERT neural model receives that entire batch in one forward call.

## Why compare with P2-06A?

P2-06A is the correctness reference because every source waveform is forwarded
at exact semantic length.

P2-06B compares native batching against that reference at two levels:

1. Structural
   - exact valid frame count match.

2. Numerical
   - mean frame-wise cosine similarity;
   - minimum frame-wise cosine similarity (reported);
   - pooled-feature cosine similarity;
   - MAE and max absolute difference (reported).

Default numerical Gate:

```text
mean frame cosine >= 0.999
pooled cosine     >= 0.999
```

These thresholds are intentionally diagnostic rather than assumed facts.

## Important interpretation

A native batching FAIL does NOT imply that the pretrained model is broken.

It means:

```text
padded batched inference
!=
exact-length singleton reference
```

to the selected tolerance.

This may reveal a frontend normalization/padding interaction and should lead to
a model-specific batching policy rather than weakening the mask semantics.

## P2-06B Gate

Run all three real candidates. Only after observing their measured
reference-vs-native behavior should the project decide whether each backbone
may use native padded batching for large-scale feature extraction.
