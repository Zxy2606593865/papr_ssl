# PAPR-SSL 数据协议

状态：**FROZEN — P2 PASS / CLOSED**

## 1. 原则

P2 统一数据访问、manifests、音频处理、安全 batching、task views 和评估角色。不会改写原始音频或 raw manifests。Task-specific labels 均由派生视图生成。

## 2. GSC v2

- 角色：通用标准语音 KWS / embedding benchmark。
- Task：**GSC-35** closed-set classification。
- 音频：16 kHz；模型输入长度为 16000 个 samples；短音频在右侧补齐，并保留有效长度以供 mask 使用。

## 3. MDSC

MDSC 作为多用途的普通话构音障碍语音资源处理，而非包含 3784 个类别的短语数据集。

After approved surface normalization:

```text
normalized phrases: 3784
cross-domain phrases: 2549
```

不执行语义同义词合并。

### 3.1 MDSC-Core30 — primary public phrase benchmark

```text
utterances: 5083
classes:    30
train/dev/test: 3756 / 442 / 885
control/dysarthria: 2763 / 2320
```

用途：构音障碍短语表征、SSL backbone/layer/head 比较，以及 Control-vs-Dysarthria 分析。类别清单依据稳定的 train-side speaker 覆盖情况确定。

### 3.2 MDSC-Command20 — command-only diagnostic

```text
utterances: 2783
classes:    20
train/dev/test: 2056 / 242 / 485
```

用途：评估普通命令表征，不依赖 wake-word 语义。

### 3.3 MDSC-WWS10 — auxiliary personalization benchmark

用途：1-shot 个性化 wake-word spotting 和 prototype/few-shot 个性化分析。它不定义最终产品的激活机制。

### 3.4 MDSC open-set / rejection evaluation

```text
utterances: 3533
dev/test: 1178 / 2355
control/dysarthria: 1177 / 2356
```

Open-set utterances **不会**合并为一个有监督的 SCAF class。它们用于 dev-side threshold calibration、封存 test-side rejection evaluation、hard-negative 分析或未来的辅助目标。

## 4. Task-dependent Labels

Example:

```text
"打开空调"
WWS10   -> non-wake / negative
Core30  -> target class "打开空调"
OpenSet -> not open-set because it belongs to Core30
```

因此，NON_WAKE 和 OPEN_SET 是 task-view labels，而不是写入 raw manifest 的全局标签。

## 5. MDSC 音频策略

- Preserve the full utterance.
- Stereo -> mono by channel mean in memory.
- Variable-length batches use right padding and waveform masks.

## 6. 安全 SSL Batching

- WavLM: native padded batch.
- W2v-BERT 2.0: semantic trim before frontend/native batching.
- Wav2Vec2: group by exact semantic waveform length, forward each group, then restore original order.

## 7. Split 策略

```text
train -> fitting / supervised learning
dev   -> model selection / threshold calibration
test  -> sealed final evaluation
```

## 8. 最终 P2 Task Map

```text
GSC-35          -> generic standard-speech embedding benchmark
MDSC-Core30     -> primary dysarthria phrase representation benchmark
MDSC-Command20  -> command-only diagnostic
MDSC-WWS10      -> auxiliary 1-shot personalization benchmark
MDSC long-tail  -> open-set / rejection evaluation
```

P2-10 完成后，只有发现已记录的数据错误，才会重新打开 P2。
