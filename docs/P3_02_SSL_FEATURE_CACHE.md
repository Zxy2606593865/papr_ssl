# P3-02 — 离线 SSL 特征缓存策略

## 决策

P3 将**离线 masked-mean SSL feature cache** 作为 backbone/layer sweep 的默认特征策略。

缓存数据流如下：

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

之后，P3 中可训练的 head 完全离线运行：

```text
cached [D]
   ↓
Linear(D -> 64)
   ↓
L2 Norm
   ↓
SCAF
```

这样可以避免每个 epoch、每次 representation-head 实验都反复运行大型 SSL backbone。

## 为什么缓存 masked means，而不是所有 frame features？

Frame-level caches 占用空间很大。

P3 基线已冻结为 Masked Mean pooling 策略，因此缓存 `[D]` 即足以比较：

- Wav2Vec2、WavLM 与 W2v-BERT 2.0；
- 候选 hidden layers；
- 64D projection 设置；
- SCAF 设置。

Frame-level caching 延后到之后的 Attention-Pooling 实验。届时只需将少量进入最终筛选的 backbone/layer 组合物化为 frame-level 缓存。

这样可以避免粗粒度 sweep 产生数百 GB 的缓存数据。

## 一次模型 forward，生成多个 layer caches

`output_hidden_states=True` 会在一次模型 forward 中提供所有 hidden-state layers。

因此，P3-02 明确禁止对同一 audio batch 分别执行：

```text
forward layer 3
forward layer 6
forward layer 9
forward layer 12
```

而应执行：

```text
ONE frozen SSL forward
          ↓
hidden_states
 ├─ layer 3  -> masked mean -> cache
 ├─ layer 6  -> masked mean -> cache
 ├─ layer 9  -> masked mean -> cache
 └─ layer 12 -> masked mean -> cache
```

存储引擎通过 `fanout_multilayer_masked_mean()` 支持此流程。

## Cache identity

每份 cache 由以下字段唯一标识：

```text
model_id
exact model_revision
exact outputs.hidden_states layer index
pooling = masked_mean
feature_dtype = float32
```

`model_revision` 必须是精确的 40 字符 commit SHA。`main` 等可变引用会被拒绝。

已知的 P1 revisions：

```text
facebook/wav2vec2-base
0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8

microsoft/wavlm-large
c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c

facebook/w2v-bert-2.0
da985ba0987f70aaeb84a80f2851cfac8c697a7b
```

## Layer 语义

`layer` 始终表示：

```text
the exact tuple index in outputs.hidden_states
```

它不是宽泛意义上的“Transformer layer 编号”。

这与 P1 契约一致。

## Sample hash

每个缓存样本都会存储：

```text
sample_hash = SHA-256(
    schema
    + sample_rate
    + valid semantic sample count
    + float32 semantic waveform bytes
)
```

不将 `valid_num_samples` 之后的 padding 纳入计算。

因此：

- batch padding 不同，不会导致 cache 失效；
- semantic audio 变化会导致 cache 失效；
- 读取时会拒绝过期 cache entries。

## 以音频为中心，而非以 task 为中心

Cache **不存储 labels**。

例如，同一条 MDSC utterance 在不同 task view 中可能是：

```text
WWS10      -> negative
Core30     -> target phrase
Command20  -> target command
```

SSL feature 表示相同的音频，因此只应缓存一次，并由所有 task views 复用。

## 存储

每份 model/revision/layer cache 使用分片 `safetensors`：

```text
artifacts/ssl_feature_cache/
└── <model>__rev-<sha>__layer-XX__masked-mean-f32/
    ├── manifest.json
    ├── index.jsonl
    ├── shard_00000.safetensors
    ├── shard_00001.safetensors
    └── ...
```

`index.jsonl` 为每个样本记录：

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

## P3-02 Gate

满足以下测试要求时，P3-02 通过：

1. sample hash 是确定性的；
2. batch padding 不改变 sample hash；
3. semantic waveform 改变会改变 sample hash；
4. revision 必须精确，layer 属于 cache identity；
5. masked mean 会忽略无效帧；
6. 拒绝过期的 sample hashes；
7. 拒绝 revision/layer 错误的 caches；
8. 能够通过一次 forward 缓存多个选定 layers。

下一步集成将使用已冻结的 P1 真实 backbones 和 P2 audio pipelines 执行实际的大规模 cache materialization。
