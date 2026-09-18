# P2-05B Public Audio Preprocessing Policy

## Frozen decisions

### GSC v2

- Raw source: 16 kHz, mono.
- One-second model container: 16,000 samples.
- Source utterances shorter than 16,000 samples are right-zero-padded in memory.
- `valid_num_samples` remains the source length.
- Padding is therefore explicitly masked and is not treated as speech.
- A GSC speech item longer than 16,000 samples fails fast.
- `_background_noise_` assets are not consumed by the speech loader.

### MDSC / AISHELL-6B

- Raw source: 16 kHz.
- Full utterance is preserved.
- No 1-second crop.
- No truncation.
- Multi-channel source is averaged across channels to mono in memory.
- Raw files remain unchanged.

### Batch contract

For examples with semantic lengths \(L_i\), let

\[
T_{\max}=\max_i T_i
\]

where \(T_i\) is the stored waveform length after dataset-specific
example preprocessing.

The collate function emits:

```text
waveforms      float32 [B, T_max]
lengths        int64   [B]
waveform_mask  bool    [B, T_max]
```

with

\[
m_{i,t} = [t < L_i].
\]

All samples outside valid lengths must be exactly zero.

## Why GSC stores 16,000 samples but keeps a shorter valid length

GSC source files include utterances shorter than one second. The fixed
one-second container preserves the standard GSC representation, while
`valid_num_samples` prevents artificial right-padding from being confused
with observed source speech.

## Important boundary

P2-05B only defines waveform-side preprocessing and batching.

It does NOT yet define how each SSL implementation converts:

```text
waveform_mask / lengths
        ↓
frame_mask [B, T_feature]
```

That conversion is backbone/frontend-specific and belongs to the next
integration step.
