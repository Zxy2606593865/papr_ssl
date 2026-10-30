# P2 收尾报告

## 最终状态

**P2：PASS / CLOSED**；唯一条件是最终 policy files 合并后，仓库全量测试仍通过。

## 已完成范围

- P2-01: public dataset selection
- P2-02: raw audits
- P2-03: unified manifest contract
- P2-04: dataset adapters
- P2-05: audio and length policy
- P2-06: length-aware safe SSL batching
- P2-07A: MDSC phrase inventory
- P2-07B: initial task views
- P2-07C: personalized WWS protocol integrity
- P2-07D: phrase coverage audit
- P2-07E: revised multi-view MDSC task policy

## 对 MDSC 的最终解读

MDSC 不作为包含 3784 个类别的短语数据集处理。

冻结的 task views：

- Core30: primary public dysarthria phrase benchmark
- Command20: command-only diagnostic
- WWS10: auxiliary 1-shot personalization benchmark
- Long-tail dev/test: open-set/rejection evaluation

P2 不创建语义同义词 taxonomy。

## 交接给 P3

P3 只能使用冻结的 train views，并必须保持 test 封存。

建议首先执行以下 P3 实验：

```text
Frozen SSL backbone
-> masked temporal pooling
-> Linear(D -> 64)
-> L2 normalization
-> SCAF
```

从最清晰的监督视图开始：

```text
GSC-35
MDSC-Core30
MDSC-Command20 diagnostic
```

将 MDSC-WWS10 和 open-set rejection 用作评估协议，而不是主要的 SCAF class inventory。
