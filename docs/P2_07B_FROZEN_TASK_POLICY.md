# P2-07B 冻结任务语义

## 主要结论

原始 MDSC manifest 包含 3,858 个不同的 transcript 字符串，但不得将其视为 3,858 个 wake-word 类别。

MDSC 官方说明指出，每位说话人以不同语速重复录制十个唤醒词各五次，此外还包含非唤醒命令/短语素材。

P2-07A 清单中恰有十个覆盖全部说话人的主要短语。移除明确的 `<p>` 停顿标注后，每个规范唤醒短语包含：

```text
46 speakers × 5 repetitions = 230 utterances
```

and all ten together contain:

```text
10 × 230 = 2,300 wake-word utterances
```

因此，P2-07B 将这十个短语冻结为 MDSC 唤醒词清单。

## 十个规范唤醒词

```text
Hey Siri
你好小布
天猫精灵
小冰小冰
小度小度
小德小德
小溪你好
小爱同学
小艺小艺
灵犀灵犀
```

## 规范化边界

Task-side normalization removes only:

```text
<p> pause annotation
whitespace
Latin case differences
```

For example:

```text
小度<p>小度 -> 小度小度
你<p>好<p>小<p>布 -> 你好小布
小<p>德<p>小<p>德 -> 小德小德
```

P2 阶段不得合并同义词、纠正拼写、按语义规范化命令或改写原始 manifest。

## MDSC 主要任务

```text
speaker-dependent dysarthria wake-up word spotting
```

个性化评估范围：

```text
domain = dysarthria
split  = dev/test
role   = enrollment/eval
```

任务标签：

```text
one of 10 canonical wake words -> positive target
everything else               -> __NON_WAKE__
```

目标 enrollment 语音用于构建说话人专属的唤醒 prototype。非唤醒 enrollment 素材可以作为负例/校准上下文，但绝不能平均进正向唤醒 prototype。

threshold 在 development 数据上选定，并在 test 评估前锁定。

## GSC 任务

GSC remains the generic representation-development benchmark:

```text
gsc_all_35_closed_set
```

using all 35 observed labels and official train/dev/test partitions.

## 可选的 MDSC 诊断视图

The 30 phrases observed across all 46 speakers are retained as:

```text
mdsc_common_30_closed_set
```

这只是表征诊断视图，并非最终主要任务，也不能替代二分类个性化 WWS 协议。

## 对后续阶段的要求

P3/P4 可使用 GSC 和/或 common-30 视图进行受控表征对比。MDSC 最终资格评估必须使用个性化 10-wake-word 视图，并将非唤醒评估语音计入 false-alarm trial。
