# P6 — Teacher E0 / E0b 开发集资格评估

本方案有意止步于 `generic_test` 之前。

仅实现 P6-01 至 P6-06：

```text
P6-01  generic_enrollment -> prototype, primary K=1
P6-02  generic_dev_cal -> score/margin thresholds
P6-03  generic_dev_score -> qualification metrics
P6-04  absolute Gate
P6-05  E0b head-variant non-degradation Gate
P6-06  freeze Teacher/checkpoint/threshold/code state
```

此处有意不包含 `P6-07 generic_test`，避免冻结前意外访问 test。

## 冻结的 Checkpoint 策略

P5 生成了 3 个 Attention checkpoints 和 3 个 Mean checkpoints。

P6 必须评估一个真实 checkpoint，而不是虚构的平均模型。

The representative checkpoint is selected before P6 scoring as:

```text
median selected-dev score among seeds 17 / 29 / 43
```

此规则明确避免选择表现最好的 seed。

## 通用协议

主要 enrollment：

```text
K=1 per intent
30 intents -> 30 enrollment utterances
selection = deterministic stable hash from Core30 train
selection never examines embedding similarity or P6 scores
```

已知类开发数据：

```text
Core30 dev (442)
-> deterministic class-stratified 50/50 split
-> generic_dev_cal_known
-> generic_dev_score_known
```

未知类开发数据：

```text
Core30 open-set dev (1178)
-> deterministic 50/50 split
-> generic_dev_cal_unknown
-> generic_dev_score_unknown
```

不读取任何 test row。

## Threshold 策略

Prototype decision:

```text
s1 = highest cosine similarity
s2 = second highest cosine similarity
margin = s1 - s2

ACCEPT iff:
    s1 >= score_threshold
    AND
    margin >= margin_threshold
otherwise REJECT
```

Thresholds 仅使用 `generic_dev_cal` 选择。

Calibration grid 根据校准分数的 quantiles 生成。在满足所有绝对 Gate 的 threshold pairs 中，按以下优先顺序选择：

```text
1. max Macro F1
2. max Correct Accept
3. max Unknown Reject
4. min Wrong Intent
5. min Known Reject
```

若没有 threshold pair 满足 Gate，脚本会写出违规程度最小的诊断 pair，但会返回非零状态，且 P6-06 不得冻结。

## P6-04 绝对 Gate

```text
Macro F1       >= 0.80
Correct Accept >= 0.80
Wrong Intent   <= 0.10
Known Reject   <= 0.20
Unknown Reject >= 0.85
```

Macro F1 是拒识后的 30 类已知意图 Macro F1。被拒绝的已知 utterance 会计为其真实类别的 false negative。Open-set false accepts 由 Unknown Reject / FAR 单独评估。

## P6-05 E0b

P5 Mean 64D head 是冻结的 head-variant baseline。

P5 Attention 64D head 是主线方案。

二者使用相同的：
- WavLM-large hidden_states[15]
- enrollment utterances
- cal/score partitions
- prototype rule
- threshold calibration algorithm

E0b passes when:

```text
Mean Macro-F1 - Attention Macro-F1 <= 0.02
```

这是 head-variant 非退化 Gate，不是 1024D 对比 64D 的压缩结论。

## 运行

首先确认 open-set index 路径存在。默认路径：

```text
artifacts/p2_07/mdsc_policy_v2/
mdsc_core30_open_set_eval.index.jsonl
```

如果本地文件名不同，请传入 `--open-index <actual path>`。

Then:

```powershell
python scripts/prepare_p6_protocol.py

python scripts/run_p6_dev_qualification.py --device cuda
```

仅在开发集 Gate 通过后运行：

```powershell
python scripts/freeze_p6_teacher.py
python scripts/audit_p6_dev_gate.py
```

预期冻结状态：

```text
P6-06 STATUS: FROZEN / READY FOR P6-07
generic_test accessed: NO
```

之后才能添加 P6-07 代码，并使用已冻结的 Teacher 与 thresholds 一次性启用 generic_test。
