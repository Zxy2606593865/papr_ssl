# P2-05B 公开音频预处理策略

## 冻结的决策

### GSC v2

- 原始来源：16 kHz、单声道。
- 一秒模型输入容器：16,000 个采样点。
- 源 utterance 短于 16,000 个采样点时，在内存中向右补零。
- `valid_num_samples` 保留源音频的实际长度。
- 因此，padding 会通过显式 mask 标记，不会被视为语音。
- GSC 语音条目长于 16,000 个采样点时立即报错。
- 语音加载器不读取 `_background_noise_` 资源。

### MDSC / AISHELL-6B

- 原始来源：16 kHz。
- 保留完整 utterance。
- 不裁剪为一秒。
- 不截断。
- 多通道来源在内存中对通道求平均并转为单声道。
- 不改写原始文件。

### Batch 契约

对于语义长度为 (L_i) 的样本，令

\[
T_{\max}=\max_i T_i
\]

其中 (T_i) 是经过数据集专属样本预处理后的 waveform 存储长度。

collate 函数输出：

```text
waveforms      float32 [B, T_max]
lengths        int64   [B]
waveform_mask  bool    [B, T_max]
```

其中

\[
m_{i,t} = [t < L_i].
\]

有效长度之外的所有采样点必须恰好为零。

## 为什么 GSC 存储 16,000 个采样点，但保留更短的有效长度

GSC 源文件中包含短于一秒的 utterances。固定的一秒容器保留了标准 GSC 表示；`valid_num_samples` 则避免把人为添加的右侧 padding 误当作源语音。

## 重要边界

P2-05B 只定义 waveform 侧的预处理和 batching。

本阶段尚未定义各 SSL 实现如何将以下输入转换为特征帧 mask：

```text
waveform_mask / lengths
        ↓
frame_mask [B, T_feature]
```

该转换取决于 backbone/frontend，属于下一步集成工作。
