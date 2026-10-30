# P2-07E — 修订后的 MDSC Task Policy

MDSC 不再仅作为 wake-word 数据集处理。

## 冻结的公开 Task Views

### 1. MDSC-Core30

Thirty exact phrases with the strongest speaker coverage:

- 10 canonical wake phrases
- 20 common commands

用途：
- 构音障碍短语表征 benchmark；
- backbone/layer/head 比较；
- Control 与 Dysarthria 分析。

不执行语义同义词合并。

### 2. MDSC-Command20

The 20 non-wake commands from Core30.

用途：
- 仅评估命令的诊断；
- 避免主要短语分析依赖 wake-word 语义。

### 3. MDSC-WWS10

Existing one-shot personalized wake-word protocol.

用途：
- 仅作为辅助个性化 benchmark。

它不定义最终产品的激活机制。

### 4. MDSC Core30 Open-Set 评估

所有 task-normalized transcript 不属于 Core30 的 dev/test utterances 均作为拒识 trials。

Important:

```text
They are NOT collapsed into one SCAF training class.
```

长尾集合包含多种不同内容。若把数千个无关短语视为一个 metric-learning class，就会迫使无关语音 embeddings 聚在一起。

这些 utterances 保留用于：
- open-set threshold calibration/evaluation;
- hard-negative analysis;
- future auxiliary representation objectives.

## 冻结边界

P2 不会为长尾 transcripts 建立新的语义 intent taxonomy。这需要单独的标注协议，不属于当前公开数据工程范围。
