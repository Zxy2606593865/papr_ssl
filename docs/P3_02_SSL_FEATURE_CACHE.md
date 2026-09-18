# P3-02 — Offline SSL Feature Cache Strategy

## Decision

P3 uses an **offline masked-mean SSL feature cache** as the default feature
strategy for backbone/layer sweeps.

The cached payload is:

```text
one utterance
    ↓
frozen SSL backbone
    ↓
selected hidden_states layer: [T,D]
    ↓
frame_mask
    ↓
masked mean
    ↓
cached vector: [D], float32
```

The trainable P3 head then operates entirely offline:

```text
cached [D]
   ↓
Linear(D -> 64)
   ↓
L2 Norm
   ↓
SCAF
```

This avoids repeatedly running the large SSL backbone for every epoch and every
representation-head experiment.

## Why cache masked means instead of all frame features?

Frame-level caches are extremely large.

For the P3 baseline, the pooling strategy is already frozen to Masked Mean.
Therefore caching [D] is enough to compare:

- Wav2Vec2 vs WavLM vs W2v-BERT 2.0;
- candidate hidden layers;
- 64D projection settings;
- SCAF settings.

Frame-level caching is deferred until a later Attention-Pooling experiment.
At that time, only the small number of finalist backbone/layer combinations
should be materialized at frame level.

This avoids turning a coarse sweep into hundreds of gigabytes of cache data.

## One model forward, many layer caches

`output_hidden_states=True` exposes all hidden-state layers from a single model
forward.

Therefore P3-02 explicitly forbids:

```text
forward layer 3
forward layer 6
forward layer 9
forward layer 12
```

for the same audio batch.

Instead:

```text
ONE frozen SSL forward
          ↓
hidden_states
 ├─ layer 3  -> masked mean -> cache
 ├─ layer 6  -> masked mean -> cache
 ├─ layer 9  -> masked mean -> cache
 └─ layer 12 -> masked mean -> cache
```

The storage engine supports this via `fanout_multilayer_masked_mean()`.

## Cache identity

Every cache is identified by:

```text
model_id
exact model_revision
exact outputs.hidden_states layer index
pooling = masked_mean
feature_dtype = float32
```

`model_revision` must be an exact 40-character commit SHA. Moving references
such as `main` are rejected.

Known P1 revisions:

```text
facebook/wav2vec2-base
0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8

microsoft/wavlm-large
c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c

facebook/w2v-bert-2.0
da985ba0987f70aaeb84a80f2851cfac8c697a7b
```

## Layer semantics

`layer` always means:

```text
the exact tuple index in outputs.hidden_states
```

It does not mean a loosely named "Transformer layer number".

This matches the P1 contract.

## Sample hash

Every cached sample stores:

```text
sample_hash = SHA-256(
    schema
    + sample_rate
    + valid semantic sample count
    + float32 semantic waveform bytes
)
```

Padding after `valid_num_samples` is excluded.

Therefore:

- different batch padding does not invalidate the cache;
- changed semantic audio does invalidate the cache;
- stale cache entries fail at read time.

## Audio-centric, not task-centric

The cache does NOT store labels.

For example, the same MDSC utterance may be:

```text
WWS10      -> negative
Core30     -> target phrase
Command20  -> target command
```

The SSL feature is the same audio representation, so it must be cached only
once and reused by all task views.

## Storage

Each model/revision/layer cache uses sharded `safetensors`:

```text
artifacts/ssl_feature_cache/
└── <model>__rev-<sha>__layer-XX__masked-mean-f32/
    ├── manifest.json
    ├── index.jsonl
    ├── shard_00000.safetensors
    ├── shard_00001.safetensors
    └── ...
```

`index.jsonl` includes, for every sample:

```text
utt_id
dataset
split
sample_hash
frame_count
model_id
model_revision
layer
pooling
feature_dtype
shard
row
```

## P3-02 gate

P3-02 passes when tests prove:

1. sample hash is deterministic;
2. batch padding does not change sample hash;
3. semantic waveform changes do change sample hash;
4. revision must be exact and layer is part of cache identity;
5. masked mean ignores invalid frames;
6. stale sample hashes are rejected;
7. wrong revision/layer caches are rejected;
8. multiple selected layers can be cached from one forward.

Actual large-scale cache materialization is the next integration step using the
already frozen P1 real backbones and P2 audio pipelines.
