# H5-01 — C/W/U Decision Head

## 当前在干什么？

前面已经解决了两个问题：

```text
Global + DTW
→ “最像哪个注册短语？”
```

而 H5 第一次开始解决：

```text
这个 top1 结果到底能不能相信？
```

我们把每条 Query 的真实情况分成三种：

```text
C = Correct
    是已注册短语，而且当前 top1 认对了

W = Wrong
    是已注册短语，但当前 top1 认错了

U = Unknown
    这个短语根本没注册
```

这就是 C/W/U Decision Head。

## 为什么现在做它？

H1-02 的 2-shot + DTW 已经做到：

```text
CA ≈ 96%
WI ≈ 1%
UR ≈ 93%
```

说明“识别哪个短语”已经很强。

剩下最大的实际风险是：

```text
Unknown 被系统自信地当成某个已注册意图。
```

因此 Head 的重点不再是继续把分类 F1 从 97.1% 挤到 97.2%，
而是学会：

> 什么时候该相信 top1，什么时候不该自动执行。

## Head 吃什么？

它不碰原始音频，也不改 WavLM。

输入只是已经算好的证据：

```text
fixed Global+DTW top1 score
top1-top2 margin
Global top1 / margin
DTW top1 / margin
Global 与 DTW 是否同意
Query 与模板时长差
1-shot / 2-shot
Query 与 support 的 cosine mean/min
```

然后一个非常小的线性 3-class Head：

```text
evidence
   ↓
Linear
   ↓
C / W / U decision scores
```

## 为什么不用 H4 learned ranker？

H4-01：

```text
1-shot 有小收益
2-shot 没有稳定收益
```

所以它没有被 PROMOTE。

H5 保持更简单、已经验证稳定的：

```text
0.5 × Global + 0.5 × DTW
```

作为 ranking 主线。

## 数据怎么分？

这次严格不拿 DEV 调 Head：

```text
28 TRAIN speakers
→ 拟合 C/W/U Head

6 TRAIN speakers
→ 只选择 acceptance threshold

4 unseen DEV speakers
→ 最终开发评价
```

这样 DEV speaker 对 Head 来说完全没见过。

## 为什么还要和旧 baseline 比？

旧 baseline：

```text
top1 score >= threshold → ACCEPT
否则 → REJECT
```

新 Head：

```text
先判断 C/W/U
只有“像 C”且 C-score 足够高
→ ACCEPT
否则先不自动接受
```

两者都用同一批 6 个 TRAIN calibration speakers，
并且都在：

```text
Unknown FAR <= 10%
```

约束下选择 threshold。

这样比较是公平的。

## H5-01 还没有 CONFIRM

现在只有：

```text
自动接受
vs
不自动接受
```

H5-01 先验证 C/W/U Head 是否真的比简单 threshold 更可靠。

如果有效，下一步 H6 才正式拆成：

```text
ACCEPT
CONFIRM
REJECT
```

## 运行

```powershell
python scripts/run_h5_01_cwu_decision_head.py
python scripts/audit_h5_01_cwu_decision_head.py
```

第一次运行会重新构建 TRAIN/CAL/DEV 的 DTW evidence，因此可能需要一些时间。

重点看：

```text
BASE:
CA / WI / KR / UR

C/W/U:
CA / WI / KR / UR

DELTA:
dCA / dWI / dKR / dUR
```

理想结果不是单纯“CA最高”，而是：

```text
WI 不增加或下降
Unknown Reject 上升
同时 CA 损失尽量小
```

这才说明 Decision Head 真正有价值。
