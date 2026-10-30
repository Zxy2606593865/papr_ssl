# P3-03 — Teacher 训练配置

## P3 中的定位

P3 建立最简洁、可比较的 Teacher 基线：

```text
Frozen pretrained SSL backbone
        ↓
selected outputs.hidden_states[layer]
        ↓
Masked Mean
        ↓
Linear(D -> 64), trainable
        ↓
L2 normalization
        ↓
SCAF centers, trainable
```

P3-03 冻结**实验身份与模型超参数契约**。本阶段不定义 sampler（P3-04）、optimizer/training loop（P3-05）、multi-seed 重复实验（P3-06）、checkpoint 选择规则（P3-07）或最终 run manifest（P3-08）。

## 冻结字段

每个 Teacher run config 必须显式包含：

```text
backbone.name
backbone.model_id
backbone.model_revision
backbone.layer
backbone.frozen = true
backbone.force_eval = true
backbone.use_offline_cache = true

head.kind = mean_dr
head.embedding_dim = 64
head.l2_normalize = true

scaf.k = 3
scaf.margin = 0.2
scaf.scale = 30

seed
```

`backbone.layer` 始终表示 `outputs.hidden_states` 中精确的 tuple index。

## 基础配置

P3-03 为每个冻结的 P1 backbone 提供一份规范基础配置：

```text
wav2vec2-base   -> hidden_states[12]
wavlm-large     -> hidden_states[24]
w2v-bert-2.0   -> hidden_states[24]
```

这些是基线/默认最后层配置，**不是未来 layer sweep 的结果**。Layer-sweep run 只能覆盖 `backbone.layer`，同时保持 Mean-DR/SCAF 契约不变。

## Seed 策略

基础配置使用：

```text
seed = 17
```

P3-06 会使用以下 seeds 重复符合条件的实验：

```text
17 / 29 / 43
```

从一开始就将 seed 写入配置，确保每个 run 可复现。

## 与 P3-02 cache 的关系

P3-03 记录 P3-02 所用的精确 model revision 和 hidden-state layer。因此，训练 run 只能加载 identity 匹配以下字段的 cache：

```text
model_id
model_revision
layer
pooling = masked_mean
```

Cache 不匹配属于配置错误，不得静默忽略。

## 本阶段不冻结的内容

P3-03 有意不加入：

```text
dataset/task view
batch sampler
batch size
learning rate
optimizer
epoch count
generic_dev_score formula
checkpoint path/hash
```

这些内容分别属于 P3-04 至 P3-08。将它们排除在本阶段之外，可避免 P3-03 承担后续阶段的职责。

## P3 阶段审查说明

整体 P3 计划合理。以下两条规则应保持明确：

1. P3-02 缓存的是**masked mean 后、projection 前的 SSL 向量 `[D]`**。可训练的 `Linear(D -> 64)` 仍属于 Mean DR，不得预先计算。
2. 在 P3-05/P3-07 开始比较 checkpoints 前，必须冻结 `generic_dev_score` 的唯一数学定义。Test split 继续封存。

## Gate

满足以下条件时，P3-03 通过：

- 三份 backbone configs 均与各自精确的 P1 model revision 匹配；
- 使用范围内的精确 hidden-state index；
- backbone 保持 frozen/eval/cache 策略；
- 使用 `mean_dr`；
- 按契约输出 64D L2-normalized embeddings；
- 共用 SCAF `(K=3, m=0.2, s=30)`；
- 显式记录 seed。
