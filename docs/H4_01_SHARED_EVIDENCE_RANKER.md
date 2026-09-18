# H4-01 — Shared Evidence Ranker

## 当前在干什么？

这一步是**第一次真正训练 Task Head**。

前面 H4-00B 只是把 Backbone/Neck 的输出准备好：

```text
Audio
→ Frozen WavLM[15]
→ Frozen Attention/256D
→ Global + Temporal Features
```

现在 H4-01 开始学习：

> 当 Global 相似度和 DTW 给出证据时，应该怎样组合这些证据来判断“哪个候选短语更可信”？

## 为什么不用一个大的神经网络？

因为当前数据只有 34 个训练 speaker，而且 Backbone/Neck 已经很强。

因此先用一个非常小的共享线性 Logistic Ranker：

```text
候选短语证据
├─ Global cosine
├─ Global margin
├─ DTW similarity
├─ DTW margin
├─ 1/2-shot
├─ 时长差
└─ 是否有第二模板
        ↓
Shared Linear Logistic Head
        ↓
candidate score
```

它不是 30 类固定 Softmax。

所有用户、所有短语都共享同一组参数，所以以后新增短语不需要重新训练一个新的分类输出层。

## 为什么要和 fixed λ 比？

我们现在的 DTW baseline 是：

```text
0.5 × Global + 0.5 × DTW
```

这个 0.5 是人工固定规则。

H4-01 要回答：

> 让模型从 34 个 TRAIN speaker 自己学习 Global 和 DTW 的权重，是否比固定 0.5 更好？

如果不能超过固定融合，就没有理由把复杂度加进正式系统。

## 数据关系

```text
34 TRAIN speakers
→ 构造 1/2-shot personalized episode
→ 训练共享 Head

4 DEV speakers
→ 完全 unseen
→ 只评价
```

Backbone/Neck 完全冻结。

## 这一步还不做什么？

```text
不做 Unknown Reject
不做 ACCEPT / CONFIRM / REJECT
不使用 Unknown Exposure
不使用 generic_test
```

现在只解决：

```text
已注册短语候选之间，哪一个最可信？
```

如果 H4-01 有稳定收益，下一步 H5 才把这个 ranking evidence 接到：

```text
C = 正确
W = 错分
U = 未知
```

的最终决策 Head。

## 运行

```powershell
python scripts/run_h4_01_shared_evidence_ranker.py
python scripts/audit_h4_01_shared_evidence_ranker.py
```

第一次运行会计算训练 episode 中需要的 DTW，因此会比前面只读 NPZ 的实验慢一些。

重点看：

```text
GLOBAL F1
FIXED-DTW F1
LEARNED F1
dLearned-Fixed
better/equal/worse
```

如果 Learned 没有稳定超过 Fixed-DTW，则 H4-01 作为负消融关闭，继续保留固定融合进入 H5。
