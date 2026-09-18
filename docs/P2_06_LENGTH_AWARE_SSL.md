# P2-06A Length-Aware SSL Reference Path

## Purpose

P2-05B introduced semantic waveform lengths and a prefix-valid waveform mask.
The P1 SSL wrappers were validated only for waveform-only input. P2-06A bridges
the two without immediately modifying three already validated model-specific
wrappers.

## Reference algorithm

For a padded waveform batch:

```text
waveforms [B,Tmax]
lengths   [B]
```

P2-06A performs:

```text
sample i
  -> waveform[i, :length_i]
  -> existing SSL backbone singleton forward
  -> features_i [Ti,D]
```

Then:

```text
pad_sequence(features_i)
  -> features [B,Tfeature_max]

feature lengths
  -> frame_mask [B,Tfeature_max]
```

This produces exact, model-observed feature lengths for Wav2Vec2, WavLM and
W2v-BERT 2.0 without hard-coding convolution stride formulas.

## Why not native padded batching yet?

This is the correctness reference implementation.

Native padded batching is faster, but requires model-specific validity
propagation:

- Wav2Vec2 / WavLM: raw-sample validity must be converted through the
  convolutional frontend.
- W2v-BERT 2.0: waveform validity must be reflected in the feature extractor's
  frame-level attention mask.

Before adding that optimization, the project needs a trusted reference result
against which native batching can be compared.

## Gate

P2-06A passes only when:

1. offline adapter tests pass;
2. real GSC short/full examples produce different valid feature lengths;
3. real variable-length MDSC examples produce different valid feature lengths;
4. all three pinned SSL candidates pass the same real-data smoke;
5. all historical unit tests remain green.

No training, SCAF update, Attention DR, KD or Student work belongs here.
