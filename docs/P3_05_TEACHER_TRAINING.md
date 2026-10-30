# P3-05 — 训练 Mean DR + SCAF

## 范围

P3-05 执行 P3 中首次真实的参数学习。

训练 loop 不会加载冻结的 SSL backbone，而是读取 P3-02 的离线 masked-mean SSL cache：

```text
cached SSL vector [B,D]
        ↓
canonical Mean DR
(one-frame equivalence)
        ↓
Linear(D -> 64), trainable
        ↓
L2-normalized 64D embedding
        ↓
SCAF centers, trainable
        ↓
cross-entropy / backward
```

只有 Mean DR 和 SCAF 参数会传给 AdamW。

## 使用离线 cache 复用现有 Mean DR

项目中的 Mean DR 定义为：

```text
masked mean [B,T,D]
-> Linear(D,64)
-> L2 normalize
```

P3-02 已在缓存前执行 masked mean。因此，缓存向量 `[B,D]` 会作为单帧序列传给现有 Mean DR：

```text
[B,D] -> [B,1,D]
mask   -> [B,1] = True
```

一帧有效特征的 masked mean 与缓存向量完全相同。这样无需再实现一份 projection head，并可使 checkpoint 语义与规范 Mean DR 保持一致。

## 冻结的 P3 development score

P3-05 冻结以下指标：

```text
generic_dev_score = prototype_macro_f1
```

定义如下：

1. 使用当前 Mean DR 编码完整的 TRAIN reference set。
2. 按类别对 unit embeddings 求均值，并对均值做 L2-normalize，得到每类一个 TRAIN prototype。
3. 编码 DEV。
4. 对每个 DEV sample，按其与 TRAIN prototypes 的最大 cosine similarity 分类。
5. 在冻结的 train classes 上计算 macro F1。

这是 closed-set、无需 threshold 的分数。

该分数仅用于 P3/P4 的 checkpoint/backbone/layer 比较。Open-set threshold calibration 以及 Correct-Accept / Wrong-Intent / Reject metrics 属于 P6。

构建 prototypes 时绝不使用 DEV labels。

## 记录的 dev embedding metrics

每个 epoch 会记录：

```text
prototype_accuracy
prototype_macro_f1
intra_class_cosine_mean
wrong_class_cosine_max_mean
top1_top2_margin_mean
prototype_inter_cosine_mean
embedding_norm_mean
embedding_norm_max_abs_error
embedding_std_mean
collapsed
generic_dev_score
```

这些指标用于诊断。P3-07 只能依据冻结的 `generic_dev_score` 比较 checkpoints。

## Unit norm 契约

每个 train 和 dev Mean DR 输出都必须满足：

```text
float32 [B,64]
finite
L2 norm ~= 1
```

任何违反都会立即失败。

## Collapse 诊断

P3-05 记录：

```text
collapsed = mean per-dimension std(dev_embedding) < 1e-6
```

这是硬性诊断信号，不是模型选择分数。

## Optimization

透明且初始固定的 P3-05 optimization defaults：

```text
AdamW
epochs = 20
learning_rate = 1e-3
weight_decay = 1e-4
grad_clip_norm = 5.0
```

这些是 training-budget 参数，不属于 P3-03 architecture 契约。为保证 P4 backbone/layer 比较公平，必须保持固定。

## Checkpoints

P3-05 保存**每个 epoch 的 checkpoint**，而不是创建 `best.pt`：

```text
run_dir/
├── checkpoints/
│   ├── epoch_0001.pt
│   ├── epoch_0002.pt
│   └── ...
├── metrics.jsonl
└── checkpoint_hashes.jsonl
```

每个 checkpoint 保存：

```text
Mean DR state_dict
SCAF state_dict
optimizer state_dict
epoch
seed
train metrics
dev metrics
```

每个 checkpoint 文件都会计算 SHA-256 hash。

Checkpoint 选择有意延后至 P3-07。

## Test 隔离

P3-05 仅接收 TRAIN reference data 和 DEV。

P3-04 已拒绝创建 test datasets，因此 P3-05 没有通往 `generic_test` 的正常路径。

## P3-05 Gate

满足以下条件时通过：

1. train loss 有限；
2. 梯度确实更新 Mean DR + SCAF；
3. 所有生成的 embeddings 都符合 64D unit norm；
4. dev embedding metrics 均有限且可机读；
5. `generic_dev_score` 精确等于 `prototype_macro_f1`；
6. 每个 epoch checkpoint 均有 SHA-256 hash；
7. run directories 不会被静默覆盖；
8. 不访问封存的 test。
