# PAPR-SSL 评估协议

## 目的

本协议用于防止 test set 泄漏，并在相同的 waveform、数据集、划分、Mean DR、64D、SCAF、训练预算、threshold 策略和指标下比较 SSL backbone 与精确 hidden-state 索引。本文件定义评估契约，不包含编造的实验结果。

## Development 数据划分

大型公开数据集使用以下四个 development 分区：

| 分区 | 唯一职责 |
|---|---|
| `generic_train` | 在固定训练预算下训练 Mean DR 和 SCAF。 |
| `generic_enrollment` | 为每种 few-shot 条件建立 development prototypes。 |
| `generic_dev_cal` | 选择 score 和 top1-top2 margin thresholds。 |
| `generic_dev_score` | 选择 backbone、精确 hidden-state index、checkpoint 和未来的 DR 方案。 |

每条 utterance 只能属于一个分区。只要数据集元数据允许，划分时应隔离 speaker、recording 和 session。所有 split manifests 及其 hashes 必须在比较实验开始前冻结。

不得使用 `generic_test` 选择模型、backbone、hidden layer、checkpoint、threshold、optimizer、训练预算或超参数。开发期间禁止反复查看 test 结果。

## Backbone 与层的公平比较

每次比较必须固定以下内容：

- 相同的 waveform/window 契约与 sample IDs；
- 相同的 train/enrollment/calibration/scoring 划分；
- 相同的 Mean DR 实现与 64D 输出；
- 相同的 SCAF 实现及 `K=3`、margin `0.2 rad`、scale `30`；
- 相同的 optimizer 策略、step/epoch 预算、early-stop 规则和 seeds；
- 相同的 prototype 构建方法与 open-set 决策协议；
- 相同的 checkpoint 选择指标与评估实现。

每次比较只改变一个独立变量：SSL backbone 或精确的 `outputs.hidden_states` index。真实模型中无效的索引记录为 `INVALID_LAYER`；不得截断、替换或映射到最后一层。

## Threshold 校准

Score 和 top1-top2 margin thresholds 只能使用 `generic_dev_cal` 确定。完整 threshold 策略必须在 `generic_dev_score` 选择及任何 `generic_test` 访问之前冻结。除非另行预先声明跨域校准实验，并在主要 frozen-threshold 结果之外报告，否则不得使用 test-domain labels 重新校准 thresholds。

## 最终 generic test

只有在 backbone、精确 hidden-state index、DR、checkpoint、评估代码和 threshold 策略冻结后，才能启用 `generic_test`。其内部划分如下：

- `test_support`：为声明的 1-shot、5-shot 和 10-shot 条件建立 prototype enrollment；
- `known_query`：来自已 enrollment intents 的 utterances；
- `unknown_query`：来自 enrollment 中未出现 intents 的 utterances。

`test_support` 与 `known_query` 不能包含同一条 utterance。优先按 speaker、recording 和 session 隔离。每种 shot 条件都必须冻结 support manifest。Development thresholds 原样应用；禁止在 `generic_test` 上校准。

## 域内与跨域测试

域内评估使用 Dataset A 训练，并在 Dataset A 独立封存的 test partition 上评估。

跨域评估使用 Dataset A 训练 Teacher，并在 Dataset B 上评估。不得在 Dataset B 上重新训练或微调 Teacher。预先声明的 few-shot 协议可以使用 Dataset B 的 `test_support` 建立测试 prototypes，但在查看 Dataset B labels/results 前必须冻结 threshold 策略。域内与跨域结果应分别报告。

## 三个证据层级

最终结论需要以下三个不同层级的证据：

1. 通用公开基准（Generic public benchmark）。
2. 非典型语音/构音障碍基准（Atypical/dysarthric benchmark）。
3. 目标用户封存测试（Target-user sealed test）。

通用基准上的强结果不能证明某个 PAPR 目标用户档案已经通过。目标用户证据仍需单独通过封存 Gate。

## 指标契约

每个比率都必须同时报告计数和分母。至少报告以下指标：

| 指标 | 定义 |
|---|---|
| Known Macro F1 | 已知 intents 的逐类 F1 非加权平均值。 |
| Correct Accept Rate | 被接受且 intent 正确的 known queries / 全部 known queries。 |
| Wrong Intent Rate | 被接受但 intent 错误的 known queries / 全部 known queries。 |
| Known Reject Rate | 被拒绝的 known queries / 全部 known queries。 |
| Unknown Reject Rate | 被拒绝的 unknown queries / 全部 unknown queries。 |
| FAR | 被错误接受的 unknown queries / 全部 unknown queries。 |
| FRR | 被错误拒绝的 known queries / 全部 known queries。 |
| ACC/DET @ 1% FAR | 在 FAR 不超过 1% 的 operating point 上计算的 accuracy/detection rate。 |
| ACC/DET @ 5% FAR | 在 FAR 不超过 5% 的 operating point 上计算的 accuracy/detection rate。 |
| AUC | 所声明 detection ROC curve 下的面积。 |

必须检查以下 known-query 统计恒等式：

```text
Correct Accept Rate + Wrong Intent Rate + Known Reject Rate = 1
```

Embedding 诊断应包括平均类内 cosine similarity、平均类间 cosine similarity 或 distance，以及完整的 top1-top2 margin 分布。应分别报告 1-shot、5-shot 和 10-shot 结果；若冻结协议有规定，还应提供置信区间或重复 support draws 的结果。

## 可复现性记录

每次已报告的运行都必须留存 config、code revision、不可变的 model revision、split-manifest hashes、seed、checkpoint hash、threshold 值、support-shot manifest、metric 实现版本和环境/依赖锁定信息。Model-selection 表应记录被淘汰的候选项，且不得查看 test 结果。
