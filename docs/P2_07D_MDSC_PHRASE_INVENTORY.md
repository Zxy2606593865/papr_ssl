# P2-07D — MDSC 短语清单审计

目的：统计有足够 speaker/domain 支持的 MDSC 短语数量，以便开展 10-wake-word 辅助基准之外的短语级 embedding 实验。

本阶段只做描述性统计：不改写 manifest、不合并语义同义词、不声明 3858 个类别，也不冻结 30/50/100 类 task。

候选短语支持度仅依据 train split 统计：

- Control/train speaker 覆盖数
- Dysarthria/train speaker 覆盖数

在短语集合冻结之前，dev/test 计数只作描述性报告。

输出：

- mdsc_phrase_inventory.csv
- mdsc_phrase_inventory_summary.json

Threshold sweep 用于观察是否存在自然形成的 30/50/100/200 类 operating point。
