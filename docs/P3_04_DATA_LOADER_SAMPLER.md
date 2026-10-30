# P3-04 — 训练数据加载器与 Sampler

## 目标

明确 P3 训练 batch 的统计组成，并确保其可复现。

Train loader 不得静默继承源数据集的长尾频率分布。

## 数据路径

P3-04 使用 P3-02 的离线特征 cache 训练：

```text
task-view index
      +
SSL cache identity
      ↓
OfflineFeatureTaskDataset
      ↓
cached masked-mean vector [D]
      +
train-defined integer class id
      ↓
ClassBalancedBatchSampler
```

本阶段不执行 SSL backbone forward。

## Train class map

Class ids **仅依据 train 定义**：

```text
sorted unique train task labels
    ↓
0 .. C-1
```

Dev 必须复用完全相同的 train class map。

Dev 中出现 train 未包含的 label 属于协议错误。

## Train sampler

P3-04 使用显式的：

```text
C-way × K-shot
```

batch 组成。

默认 sampler 契约：

```text
classes_per_batch = 8
samples_per_class = 4
batch_size = 32
seed = Teacher config seed
batches_per_epoch = ceil(train_size / 32)
```

这些值可以配置，但每个 run 都必须在之后的 P3-08 run manifest 中记录。

### 为什么采用 class-balanced？

假设原始 train 分布为：

```text
A: 1000
B: 300
C: 80
```

使用普通 `shuffle=True` 时，SCAF 看到 class A 的次数会远多于 C。

P3 sampler 则独立于原始类别频次选择 classes。一个 epoch 内各类的暴露次数最多相差一个 batch。

对每个选中的 class：

- 若样本数不少于 K，该 batch 会抽取 K 个互不重复的 examples；
- 若样本数少于 K，则显式允许有放回抽样。

以下组合保证该行为确定：

```text
seed + epoch
```

每个 training epoch 前调用：

```python
sampler.set_epoch(epoch)
```

## Dev loader

Dev 与 train 的处理方式不同：

```text
shuffle = false
rebalancing = false
drop_last = false
full deterministic pass
```

Dev metrics 必须反映实际冻结的 dev set，而不是平衡重采样后的结果。

## Test 保护

P3-04 会有意拒绝创建 `split="test"` datasets。

P3-07 仍会保持 generic test 封存。只有模型/checkpoint 选择冻结后的最终评估阶段才能加载 test。

## Cache join 完整性

Dataset 使用以下键将 task rows 与缓存特征关联：

```text
(dataset, utt_id)
```

并验证：

- cache 中存在对应 row；
- split 一致；
- task index 若带有 `sample_hash`，必须与 cache 相符；
- cache identity 已由 P3-02 reader 验证。

Task label 仍不写入以音频为中心的 cache。

## Output batch 契约

Train/dev batches 包含：

```text
features      float32 [B,D]
labels        int64   [B]
task_labels   list[str]
datasets      list[str]
utt_ids       list[str]
splits        list[str]
sample_hashes list[str]
frame_counts  int64   [B]
```

P3-05 会使用缓存的 `[B,D]` vectors 和 labels 进行 Mean DR + SCAF 训练，并保留 metadata 供审计。

## P3-04 Gate

满足以下条件时通过：

1. Train batches 严格遵循 C-way × K-shot；
2. 即使源数据计数不平衡，各类暴露次数仍保持平衡；
3. 相同 seed/epoch 下采样结果一致；
4. `set_epoch()` 会改变下一 epoch 的抽样结果；
5. class map 由 train 定义并由 dev 复用；
6. dev 是确定性的、完整遍历且不重平衡的 pass；
7. cache/task split 与可选 sample-hash 不匹配时立即失败；
8. 拒绝构造 test dataset。
