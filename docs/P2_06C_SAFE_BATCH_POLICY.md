# P2-06C 安全 SSL 批处理策略

## P2-06B 实测结果

已将原生右侧补零 batching 与精确长度 singleton reference 进行比较。

### WavLM
在 GSC 和 MDSC 上均通过。

### W2v-BERT 2.0
在每一行进入共享 feature-extractor 调用前，先按语义长度裁剪原始 waveform；此后在 GSC 和 MDSC 上均通过。

### Wav2Vec2
帧数完全匹配，但较短行的特征值不匹配。

观测示例：

```text
GSC short row
mean frame cosine ~= 0.127
pooled cosine     ~= 0.153

MDSC shorter row
mean frame cosine ~= 0.859
pooled cosine     ~= 0.927
```

每个 padded batch 中最长行与 singleton reference 基本一致，因此差异定位在原始右侧 padding 的相互作用，而不是 hidden-layer 选择或帧数转换。

## 冻结的安全策略

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

该策略避免静默接受因数值损坏而产生的短样本表示。

## 性能影响

对 GSC 而言，该策略仍较有效率，因为大部分语音文件恰好为 16,000 个采样点，可以共用较大的 Wav2Vec2 batch。

对变长程度很高的 MDSC，精确长度分组可能产生许多较小的 Wav2Vec2 groups。P2 阶段为保证正确性接受这一开销。只有在 Wav2Vec2 仍是有竞争力的 Teacher 候选项时，才考虑侵入性更大的“逐样本卷积 frontend、批量 transformer”优化。
