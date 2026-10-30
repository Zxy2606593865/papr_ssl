# P3-06 — 多 seed 重复实验

## 目标

P3 结论不得依赖一次碰巧有利的初始化或 batch 顺序。

Every qualifying experiment is repeated with at least:

```text
17 / 29 / 43
```

Additional predeclared seeds are allowed, but these three may not be omitted.

## 哪些内容必须保持一致？

Across seeds, keep fixed:

```text
dataset / task view
feature cache
backbone model_id + exact revision
hidden-state layer
Mean DR architecture
embedding_dim = 64
SCAF K / margin / scale
optimizer
learning rate
weight decay
epoch count
classes_per_batch
samples_per_class
dev metric definition
```

只改变随机 seed。

## Seed 可以改变哪些内容？

The seed may change:

```text
Mean DR Linear initialization
SCAF center initialization
balanced-sampler draw/order
other seeded PyTorch / NumPy / Python random operations
```

这是有意设计的。P3-06 用于衡量结果对这些随机因素的敏感程度。

## 关键执行顺序

常见的可复现性错误是：

```text
construct model
↓
set seed
↓
train
```

这样设置得太晚：模型已经完成随机初始化。

P3-06 enforces:

```text
seed_everything(seed)
↓
construct Mean DR
↓
construct SCAF
↓
construct seed-aware sampler
↓
train
```

`run_training()` 会在内部为训练阶段再次设置 seed；但只有 P3-06 外层设置的 seed 能保证初始化受控。

## 不可变目录

One experiment becomes:

```text
experiment_dir/
├── seed_0017/
│   ├── checkpoints/
│   ├── metrics.jsonl
│   └── checkpoint_hashes.jsonl
├── seed_0029/
├── seed_0043/
└── multiseed_summary.json
```

已有的非空 seed run directories 不会被静默覆盖。

## P3-06 不选择 checkpoints

本阶段重复训练，**不**选择 `best.pt`。

`multiseed_summary.json` 仅用于监控，会汇总各 seed **相同的最终 epoch**。

Checkpoint 选择仍由 P3-07 负责，且只使用：

```text
generic_dev_score = prototype_macro_f1
```

## 均值 ± 标准差

P3-06 将最终 epoch 的诊断结果记录为：

```text
mean ± sample standard deviation
```

with:

```text
ddof = 1
```

样本标准差计算如下：

\[
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}
\]

For the frozen three seeds, `n=3`.

P3-07 为每个 seed 选出一个 checkpoint 后，比较所选 run 时也应使用相同的汇总方式。

## 失败或表现较弱的 seeds

正常完成但分数较低的 seed **不得丢弃**。

Examples:

```text
17 -> 0.84
29 -> 0.82
43 -> 0.69
```

0.69 的结果说明存在不稳定性，必须保留在报告中。

只有发生有记录的执行或数据故障时，才可排除某个 run，例如 cache 损坏、NaN、OOM 或执行中断/未完成。重跑时必须沿用同一个 seed。

## P3-06 Gate

满足以下条件时通过：

1. seeds 17 / 29 / 43 are all present and unique;
2. the seed is set before model/SCAF construction;
3. the same training budget is used for all seed runs;
4. each seed gets an independent immutable run directory;
5. all normally completed seeds remain in the summary;
6. aggregate reports mean and sample std (`ddof=1`);
7. P3-06 performs no checkpoint selection.
