# PAPR-SSL — P2 最终数据协议

状态：**FROZEN**

本文档是 P3 模型训练开始前 P2 数据策略的最终参考。

## 1. 范围

P2 定义数据集结构、任务视图、音频处理、批处理安全策略和评估职责，不负责训练 representation head、SCAF head 或 student model。

## 2. 公共数据集

### GSC v2

用途：

- 通用关键词 / embedding 合理性基准；
- 标准语音分类参考；
- 模型和 pipeline 回归测试。

冻结的任务视图：

- 35-class closed-set benchmark.

### MDSC

MDSC 被视为**多用途构音障碍语音资源**，而非仅用于 wake-word 的数据集，也不是包含 3784 个类别的短语数据集。

原始 manifest 保持不变。各任务视图由该 manifest 派生。

## 3. MDSC 任务视图

### 3.1 MDSC-Core30 — 主要公开短语基准

用途：

- 构音障碍短语表征基准；
- backbone/layer/head 对比；
- Control 与 Dysarthria 分析。

Definition:

- 30 high-coverage exact phrases;
- 10 canonical wake phrases;
- 20 common non-wake commands;
- surface normalization only (`<p>`, whitespace/case handling already approved);
- no semantic synonym merging.

Observed frozen counts:

```text
5083 utterances
30 classes

train  3756
dev     442
test    885

control     2763
dysarthria  2320
```

类别集合依据充分的**训练集说话人覆盖率**确定。Dev/test 是评估分区，不得用于重新定义类别清单。

### 3.2 MDSC-Command20 — 仅命令诊断视图

用途：

- 在不依赖 wake-word 语义的情况下测试短语表征；
- 提供更清晰的命令识别诊断。

Observed frozen counts:

```text
2783 utterances
20 classes

train  2056
dev     242
test    485

control     1513
dysarthria  1270
```

### 3.3 MDSC-WWS10 — 辅助个性化基准

用途：

- 1-shot 个性化 wake-word spotting；
- prototype / few-shot 个性化分析。

这**不是**主要产品任务，也**不意味着**部署系统必须使用 wake-word 激活。

激活机制与短语识别保持正交。

### 3.4 MDSC-Core30 open-set 评估

Core30 之外的所有 dev/test 语音均保留为 open-set/rejection 素材。

Observed frozen counts:

```text
3533 utterances

dev   1178
test  2355

control     1177
dysarthria  2356
```

Important:

```text
open-set 池不是受监督的 SCAF 类别。
```

不得将无关的长尾 utterance 合并为一个 metric-learning 类别，否则会迫使语义无关的语音 embedding 聚集在一起。

允许的用途：

- open-set threshold calibration on dev;
- final rejection evaluation on test;
- hard-negative analysis;
- future auxiliary objectives.

## 4. Label 语义取决于任务

同一条 utterance 在不同任务视图中可以承担不同角色。

Example:

```text
"打开空调"

WWS10 view      -> non-wake / negative
Core30 view     -> target class "打开空调"
Open-set view   -> not open-set because it belongs to Core30
```

因此，`NON_WAKE` 或 `OPEN_SET` 等标签是**任务视图标签**，而不是写回原始 manifest 的全局语义标签。

## 5. 长尾转写策略

数千条 MDSC 长尾转写不会被丢弃，但 P2 不会为它们另行创建意图分类体系。

Frozen policy:

- 不创建数千个 SCAF 类别；
- 不在 P2 阶段进行人工语义合并；
- 不改写原始 manifest；
- 将其保留用于拒识、hard negatives 和未来辅助目标。

## 6. 音频策略

### GSC

- model-side fixed 16000 samples;
- right-pad short utterances;
- reject speech longer than the frozen one-second policy where required by the
  current GSC task protocol.

### MDSC

- preserve the full utterance;
- stereo -> mono by channel mean when necessary;
- variable-length batches use right padding and waveform masks.

## 7. SSL 批处理策略

已冻结的安全策略：

- WavLM: native padded batch;
- W2v-BERT 2.0: semantic trim before frontend/native batching;
- Wav2Vec2: group by exact semantic waveform length, forward each group, restore
  original order.

仅有正确的 frame mask 并不能保证 SSL feature 对 padding 不敏感。

## 8. Train/dev/test 规则

- train：拟合 / 监督学习；
- dev：模型选择、校准和 threshold；
- test：封存的最终评估。

不得使用 test split 选择类别、threshold、checkpoint 或模型变体。

## 9. P2 关闭条件

只有满足以下条件，P2 才算 CLOSED：

1. the full unit-test suite passes;
2. `mdsc_task_policy_v2_summary.json` reports `overall = PASS`;
3. `scripts/validate_p2_final_closure.py` reports `P2 FINAL STATUS: PASS / CLOSED`.

关闭后，模型开发将在 P3 继续；除非发现有记录的数据错误，否则不得重新打开已冻结的数据语义。
