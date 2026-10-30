# P6 未知样本暴露训练集数据审计

## 目标

在添加 open-set training loss 前，先确认现有 MDSC training split 中是否有足够的**非 Core30**语音用于未知样本暴露训练。

术语说明：

- Unknown Exposure = 未知样本暴露训练
- Known = 已知短语
- Unknown = 未知短语
- Phrase-disjoint = 短语类别互斥
- Leakage = 数据泄漏

此步骤**不会**训练模型。

## 已冻结的开发策略

Candidate unknown exposure rows must satisfy:

```text
split == train
normalized phrase NOT in Core30
normalized phrase NOT in current open-set DEV phrase identities
```

脚本有意不使用 `generic_test` 选择候选样本。

It therefore reports:

```text
generic_test used for selection: NO
generic_test audio/features: NO
```

test 短语重叠审计推迟到最终模型冻结后进行。该审计只能用于描述，不能用于重新调优模型。

## 运行

通常运行：

```powershell
python scripts/audit_p6_unknown_exposure_candidates.py
python scripts/review_p6_unknown_exposure_audit.py
```

第一个 script 会尝试自动发现完整的 MDSC JSONL manifest。

如果自动发现失败或选错文件：

```powershell
python scripts/audit_p6_unknown_exposure_candidates.py `
  --mdsc-manifest path\to\full_mdsc_manifest.jsonl
```

## 输出

```text
artifacts/p6_unknown_exposure_audit/
├── p6_unknown_exposure_audit.json
├── unknown_exposure_train_candidates.jsonl
└── unknown_exposure_phrase_summary.jsonl
```

## 检查内容

重点检查以下数据：

```text
candidate rows
candidate phrases
candidate speakers
Control / Dysarthria domain counts
phrase support >= 5 / 10 / 15
Core30 overlap = 0
open-set DEV phrase overlap = 0
```

审查这些数据后，才能冻结 open-set-aware Teacher 实验方案。
