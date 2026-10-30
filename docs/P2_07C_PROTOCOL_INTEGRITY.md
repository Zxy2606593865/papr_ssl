# P2-07C 个性化 MDSC 协议完整性

这是 Teacher 实验之前的最后一个 P2 Gate。

冻结的 P2-07B task 假设每个 speaker 都有自己的 enrollment prototypes。因此，数据集必须确实支持 speaker 级别的该协议。

Validator 检查以下条件：

```text
8 personalized dysarthria speakers
dev/test speaker disjointness
one split per speaker
all 10 wake words present in enrollment for every speaker
all 10 wake words present in eval for every speaker
non-wake eval trials present for every speaker
```

它也会报告每个 wake 在 enrollment/eval 中观测到的重复次数，但不会把次数硬编码进可行性 Gate。

PASS 表示公开数据集和冻结的 task 语义足以构建后续协议：

```text
speaker enrollment
    -> 10 speaker-specific wake prototypes
eval wake
    -> positive trials
eval non-wake
    -> false-alarm / rejection trials
```

Threshold 调优仅可使用 development 数据；test 仍保持封存。
