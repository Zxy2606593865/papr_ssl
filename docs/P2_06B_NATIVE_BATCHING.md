# P2-06B 原生批处理 mask

## 源码检查结果

当前项目 wrappers 已为以下模型实现原生 padded batching：

- Wav2Vec2：`forward(waveforms, sample_mask=None)`
- WavLM：`forward(waveforms, sample_mask=None)`

二者都会把 `sample_mask` 作为 `attention_mask` 传入 Hugging Face 模型，并使用模型 helper 将采样点有效性转换为所选特征帧的有效性。

W2v-BERT 2.0 先前尚无输入侧 mask 契约。

## W2v-BERT 2.0 原生策略

给定：

```text
waveforms   [B,Smax]
sample_mask [B,Smax]
```

wrapper 推导语义原始长度：

```text
Li = sum(sample_mask[i])
```

并构造一个变长 Python 列表：

```text
[
  waveform[0, :L0],
  waveform[1, :L1],
  ...
]
```

该列表通过一次 `AutoFeatureExtractor` 调用传入，并设置 `padding=True`。Feature extractor 生成：

```text
input_features [B,Tmax,F]
attention_mask [B,Tmax]
```

之后，W2v-BERT 神经网络模型在一次 forward 调用中处理整个 batch。

## 为什么要与 P2-06A 比较？

P2-06A 是正确性参考，因为它会按每条源 waveform 的精确语义长度分别执行 forward。

P2-06B 从两个层级将原生 batching 与该参考结果比较：

1. 结构层面
   - 有效帧数必须完全匹配。

2. 数值层面
   - 帧级 cosine similarity 均值；
   - 帧级 cosine similarity 最小值（报告）；
   - pooled-feature cosine similarity；
   - MAE 与最大绝对差（报告）。

默认数值 Gate：

```text
mean frame cosine >= 0.999
pooled cosine     >= 0.999
```

这些阈值用于诊断，不应被视为预设事实。

## 结果解读要点

原生 batching 未通过，并不意味着预训练模型本身损坏。

这表示在设定容差下：

```text
padded batched inference
!=
exact-length singleton reference
```

这可能揭示 frontend normalization 与 padding 之间的相互作用。应据此制定模型专属 batching 策略，而不是放宽 mask 语义。

## P2-06B Gate

运行全部三个真实候选项。观察实测的 reference-vs-native 行为后，再决定各 backbone 是否可在大规模特征提取中采用原生 padded batching。
