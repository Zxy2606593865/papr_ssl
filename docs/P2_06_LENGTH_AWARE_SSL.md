# P2-06A 长度感知 SSL 参考路径

## 目的

P2-05B 引入了语义 waveform 长度和前缀有效 waveform mask。P1 SSL wrappers 仅针对 waveform-only 输入完成验证。P2-06A 在不立即修改三个已验证的模型专属 wrapper 的前提下，连接这两套契约。

## 参考算法

对于补齐后的 waveform batch：

```text
waveforms [B,Tmax]
lengths   [B]
```

P2-06A 按以下方式处理：

```text
sample i
  -> waveform[i, :length_i]
  -> existing SSL backbone singleton forward
  -> features_i [Ti,D]
```

随后：

```text
pad_sequence(features_i)
  -> features [B,Tfeature_max]

feature lengths
  -> frame_mask [B,Tfeature_max]
```

这样可以获得 Wav2Vec2、WavLM 和 W2v-BERT 2.0 实际观测到的特征长度，无需硬编码卷积 stride 公式。

## 为什么暂不采用原生 padded batching？

这是用于验证正确性的参考实现。

原生 padded batching 更快，但需要按模型传播有效长度：

- Wav2Vec2 / WavLM：必须将原始采样点有效性转换并传播过卷积 frontend。
- W2v-BERT 2.0：必须通过 feature extractor 的帧级 attention mask 反映 waveform 有效性。

在加入该优化之前，项目需要可信的参考结果，以便与原生 batching 进行比较。

## Gate

P2-06A 仅在满足以下条件时通过：

1. 离线 adapter tests 通过；
2. 真实 GSC 短样本和完整样本产生不同的有效特征长度；
3. 真实变长 MDSC 样本产生不同的有效特征长度；
4. 三个固定 revision 的 SSL 候选项都通过相同的真实数据 smoke；
5. 所有历史单元测试仍然通过。

本阶段不包含训练、SCAF 更新、Attention DR、KD 或 Student 工作。
