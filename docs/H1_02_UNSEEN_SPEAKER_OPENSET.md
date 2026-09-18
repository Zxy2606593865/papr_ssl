# H1-02 — Unseen-Speaker Personalized Open-set Baseline

## 当前在干什么？

H1-01 已经回答：

> 新用户只注册 1～2 条语音时，当前 256D + Prototype 本身能不能区分已注册短语？

答案很好：

```text
1-shot Global F1 = 95.49%
1-shot + DTW F1  = 95.89%

2-shot Global F1 = 96.89%
2-shot + DTW F1  = 97.13%
```

所以现在进入第二个问题：

> 如果用户说了一句“根本没注册过”的短语，系统能不能拒绝它，而不是硬认成某个已注册意图？

这就是 Open-set。

## 为什么这一阶段还不做复杂 Head？

因为先要建立一个最简单、干净的基线：

```text
top1 score >= threshold → ACCEPT
top1 score <  threshold → REJECT
```

如果这个简单基线已经很强，后面的 LOO / Shrinkage / C-W-U Head 必须证明自己真的比它更好。

如果简单基线很弱，我们也能清楚知道问题到底出在 rejection，而不是 representation。

## 怎么构造 Known / Unknown？

每个 DEV speaker 有 30 个 Core30 phrase。

每次 episode：

```text
20 类 → 当前用户已注册
10 类 → 当前用户未注册
```

对于已注册类：

```text
1/2 条 → Enrollment
剩余    → Known Query
```

对于未注册类：

```text
全部录音 → Unknown Query
```

注意：

```text
Unknown ≠ 永久的第31类
Unknown = 当前用户没有注册这个短语
```

下一次 episode，同一个 phrase 完全可以变成 Known。

## 为什么用 3 个 DEV speaker 校准、1 个 DEV speaker评分？

我们不能在“当前要评分的新用户”自己的 query 上挑 threshold。

所以做 leave-one-speaker-out：

```text
Speaker A/B/C → 选 threshold
Speaker D     → 真正评分
```

然后轮换 4 次。

这样 threshold 没看过当前被评分 speaker 的 query。

## Threshold 怎么选？

保持最简单：

```text
只看最终 top1 score
```

在 calibration speakers 上选择：

```text
Unknown FAR <= 10%
```

时 Correct Accept 最高的 threshold。

这里 FAR 指：

```text
unknown query 被错误 ACCEPT 的比例
```

## 这一步看什么？

四个核心指标：

```text
CA = Correct Accept
WI = Wrong Intent
KR = Known Reject
UR = Unknown Reject
```

Known query 应满足：

```text
CA + WI + KR = 1
```

当前还没有 CONFIRM 状态。

## 运行

H1-01 的 DTW 距离已经算好，所以这一步直接：

```powershell
python scripts/run_h1_02_unseen_speaker_openset.py
python scripts/audit_h1_02_unseen_speaker_openset.py
```

## 这一步为什么重要？

H1-01 验证“认得已注册短语”。

H1-02 验证“也知道什么没注册”。

两者都成立以后，我们才真正有资格开始设计：

```text
H2 LOO + Shrinkage
H3 Temporal Medoid
H4 Evidence Fusion
H5 C/W/U Decision
```

也就是正式的 Personalized Task Head。
