# P3-07 — 仅使用 Dev 选择 Checkpoint

## 冻结规则

每个 checkpoint 只能依据以下指标进行比较：

```text
generic_dev_score
```

P3-05 将其定义冻结为：

```text
prototype_macro_f1
```

选择规则为：

```text
maximize generic_dev_score
```

任何其他指标都不能取代此规则。

## 为什么只用一个指标？

P3-05 records many diagnostics:

```text
prototype_accuracy
prototype_macro_f1
intra_class_cosine_mean
wrong_class_cosine_max_mean
top1_top2_margin_mean
...
```

这些诊断指标有助于理解 embedding space，但如果训练后可以任意选择最有利于某个 checkpoint 的指标，模型选择就会变得主观。

因此，P3-07 只使用一个排序指标。

## 平分处理

If two checkpoints have exactly the same `generic_dev_score`:

```text
choose the earliest epoch
```

此规则需预先冻结。

这并不表示更早的 epoch 在科学上总是更好，而是一条预先声明的确定性规则，可避免平分后再参考其他指标。

## Test 保持封存

P3-07 不得接收或查看任何 `generic_test` 值。

如果输入 metrics 含有任何与 test 相关的字段，selector 会拒绝继续执行。

常规 P3 数据路径已通过 P3-04 阻止构造 test dataset。

## Checkpoint 完整性

选择 checkpoint 前，P3-07 会重新计算其 SHA-256，并与 P3-05 记录的 hash 比对。

Checkpoint 缺失或 hash 不匹配时，选择失败。

## 不复制为 `best.pt`

P3-07 writes a small immutable pointer:

```text
seed_0017/
└── selected_checkpoint.json
```

它不会再创建名为 `best.pt` 的未跟踪二进制副本。

The pointer records:

```text
epoch
checkpoint path
checkpoint SHA-256
generic_dev_score
generic_dev_score_name
selection metric
tie-break policy
```

对于 multi-seed 实验，每个 seed 都会独立选择一个 checkpoint：

```text
seed 17 -> one dev-selected checkpoint
seed 29 -> one dev-selected checkpoint
seed 43 -> one dev-selected checkpoint
```

之后，顶层的 `selected_checkpoints.json` 可以引用这三个结果。

## P3-07 Gate

满足以下条件时通过：

1. only `generic_dev_score` determines ranking;
2. `generic_dev_score_name == prototype_macro_f1`;
3. direction is maximize;
4. exact score ties use earliest epoch;
5. checkpoint SHA-256 is verified before selection;
6. test-related input is rejected;
7. selection creates a pointer JSON, not a duplicate `best.pt`;
8. one checkpoint is selected independently per seed.
