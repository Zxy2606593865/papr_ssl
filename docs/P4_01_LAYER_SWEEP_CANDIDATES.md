# P4-01 — 冻结 Backbone / Hidden-State Sweep 候选项

## 1. 目的

P4 changes only:

```text
Backbone
selected outputs.hidden_states index
```

Everything else remains frozen from P3:

```text
MDSC-Core30 train/dev
Mean DR -> 64D -> L2
SCAF K=3, margin=0.2, scale=30
AdamW
20 epochs
lr=1e-3
weight_decay=1e-4
grad_clip=5
8-way x 4-shot
seeds 17/29/43
checkpoint selection by generic_dev_score only
generic_test sealed
```

## 2. 精确索引语义

The P4 unit is always:

```text
outputs.hidden_states[i]
```

其中 `i` 是 P1 中观测并验证过的 Hugging Face tuple 精确索引。

P4 **不会**自行假设论文中的 `Layer 16` 等说法对应 `hidden_states[16]`。

如果后续需要按论文层号进行比较，必须先根据具体实现单独验证映射关系。

## 3. 冻结的候选项

P1 validated:

```text
Wav2Vec2-base
hidden_states count = 13
valid indices = 0..12

WavLM-large
hidden_states count = 25
valid indices = 0..24

W2v-BERT 2.0
hidden_states count = 25
valid indices = 0..24
```

P4-01 有意保留**所有**已验证索引，不挑选所谓“有趣”的中间层。

Total:

```text
13 + 25 + 25 = 63 Backbone/Layer configurations
```

该计算方案可行，因为 P3-02 cache 设计能从一次冻结的 SSL forward 扩展出多个 hidden-state layers。因此无需为每一层重复执行成本较高的 SSL forward。

保留索引 0，因为它确实是 P1 验证过的 `outputs.hidden_states` 成员。报告中将其标记为精确索引 0，不会误称为论文的“Layer 0”或 Transformer block 0。

## 4. P4 选择指标

P4 Gate 和用户冻结的计划要求使用一个主要排序指标：

```text
generic_dev_score = prototype_macro_f1
```

对每个 Backbone/Layer 配置：

```text
seed17 selected generic_dev_score
seed29 selected generic_dev_score
seed43 selected generic_dev_score
        ↓
mean
sample standard deviation
95% Student-t confidence interval, df=2
```

由于 n=3，95% CI 必须使用 Student-t 分布，而不能使用 `mean ± 1.96*SE`。

排序分数为 **3-seed mean generic_dev_score**。

Exact-mean tie break is frozen as:

```text
lower hidden_state_index
```

这只是处理完全平分的确定性规则，并不表示较浅层天然更好。

## 5. 不应用于选择 P4 的指标

以下指标有参考价值，但需要 acceptance/rejection threshold 和 open-set calibration：

```text
Correct Accept Rate
Wrong Intent Rate
Known Reject Rate / FRR
Unknown Reject Rate / FAR
ACC/DET @ 1% FAR
```

在 P4 使用这些指标会把 Backbone/Layer 选择与后续 open-set/threshold 资格评估混在一起。

因此，P4 选择阶段暂不使用这些指标。单一 Backbone/Layer 主线冻结后，再在 Teacher qualification 阶段评估。

P4 仍可记录无需 threshold 的 embedding 诊断指标：

```text
prototype_accuracy
intra_class_cosine_mean
wrong_class_cosine_max_mean
top1_top2_margin_mean
prototype_inter_cosine_mean
embedding norm error
collapse flag
```

但任何一项都不能取代 `generic_dev_score`。

## 6. P4 执行流程

```text
P4-01 candidate freeze
        ↓
P4-02 Wav2Vec2 0..12 sweep
        ↓
P4-03 WavLM 0..24 sweep
        ↓
P4-04 W2v-BERT 2.0 0..24 sweep
        ↓
P4-05 3-seed mean / sample std / 95% t-CI
        ↓
P4-06 best layer per Backbone, then compare three winners
        ↓
P4-07 attach resource-cost report
        ↓
P4-08 freeze one Backbone + one exact hidden_state_index
```

## 7. P4-01 Gate

满足以下条件时通过：

1. exact P1 model ids/revisions are frozen;
2. candidate indices are a subset of P1-valid indices;
3. this baseline uses full P1-valid coverage: 13 / 25 / 25;
4. index semantics are explicitly `outputs.hidden_states[i]`;
5. no paper-layer alias is inferred;
6. selection metric remains `generic_dev_score=prototype_macro_f1`;
7. seeds remain 17/29/43;
8. generic_test remains sealed.
