# P6 15-shot Enrollment-Consistency Rejector

## 动机

之前的轻量 rejector 只使用：

```text
top-1 cosine
top1-top2 margin
and simple quadratic terms
```

它未能改善完整 Gate 的结果。

然而，15-shot 系统包含的信息远多于一个 mean prototype。真正的已知 query 往往会与 15 条 enrollment utterances 中的多条持续相似；未知 query 则可能只是偶然接近 centroid 或某一条 enrollment sample。

因此，本实验使用完整的 enrollment similarity 分布。

## 冻结设置

```text
WavLM-large hidden_states[15]
AttentionDR -> 64D
15-shot enrollment set
mean-prototype intent prediction
generic_test sealed
```

## 新的 Rejector 证据

对未改变的 mean-prototype classifier 预测出的类别，计算 query 与全部 15 个 enrollment embeddings 的 cosine similarity。

Features include:

```text
top1 centroid score
top1-top2 margin
mean similarity to 15 enrollments
standard deviation
minimum
25th percentile
median
75th percentile
maximum
top-3 mean
top-5 mean
bottom-3 mean
centroid minus enrollment mean
max minus min
```

仅在每个 calibration half 上训练一个小型 balanced logistic regression。Probability threshold 也只用 calibration half 选择，再在留出的 score half 上测试。

## 运行

```powershell
python scripts/export_p6_15shot_consistency_embeddings.py --device cuda
python scripts/run_p6_15shot_consistency_rejector.py
python scripts/audit_p6_15shot_consistency_rejector.py
```

## 本实验后的决策

若本实验显著提高重复划分下完整 Gate 的通过率，则继续冻结 Teacher，并在个性化决策层使用 enrollment-distribution 证据。

若 Gate 通过率仍接近 0/20，则本阶段的决策层调优已基本穷尽。下一步应开展面向 open-set 的表征训练，而不是继续增加 rejector 复杂度。
