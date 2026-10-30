# P2-07A Task View / Label Inventory 审计

## 目的

原始 MDSC manifest 包含 3,858 个不同的 transcript strings。P2-07A 不假设它们就是最终的 3,858 个 KWS classes。

相反，它仅使用已观测的 manifest metadata 构建描述性候选清单。

## Candidate MDSC inventories

- `all_transcripts`
- `seen_in_all_speakers`
- `control_and_dysarthria_overlap`
- `train_and_eval_overlap`
- `enrollment_and_eval_overlap`
- `train_enrollment_eval_overlap`
- `eval_only_relative_to_enrollment`
- `enrollment_only_relative_to_eval`

这些名称只描述集合之间的关系。

For example:

```text
enrollment_and_eval_overlap
```

表示完全相同的 transcript 至少在两种 role 中各出现一次。它**并不**表示该 transcript 已被正式定义为 target keyword。

## GSC 开发视图

用于 PAPR-SSL 表征/backbone 筛选时，P2-07A 记录了一个简单的项目开发视图：

```text
gsc_all_35_closed_set
```

使用观测到的全部 35 个 Speech Commands labels，同时保留官方 train/dev/test split。

这是项目侧的评估视图，并非声称 Speech Commands 只有这一种官方 benchmark protocol。

## 下一步决策：P2-07B

检查真实候选清单后，P2-07B 会冻结语义 task policy：

- which MDSC phrases are eligible personalized targets;
- what data can act as non-target / unknown speech;
- how enrollment and eval are separated;
- whether command phrases and wake words are evaluated together or separately;
- which view is used for backbone/layer selection versus final domain
  qualification.

P2-07B 冻结前不得开始 SCAF training。
