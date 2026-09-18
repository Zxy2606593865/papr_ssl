# H1-01 — Unseen-Speaker Personalized Closed-set Baseline

## 当前环节在干什么？

H0 已经确认：

```text
MDSC TRAIN = 34 speakers
MDSC DEV   = 4 completely unseen speakers
```

而且 DEV 的每一个 `speaker × Core30 phrase` 都至少有：

```text
2-shot support + 1 query
```

所以 H1-01 第一次真正按照“用户”建立注册库。

例如某个 DEV speaker：

```text
用户 A
├─ 我要喝水
│  ├─ 1/2 条作为注册
│  └─ 其余作为 Query
├─ 我要吃饭
│  ├─ 1/2 条作为注册
│  └─ 其余作为 Query
└─ ...共 30 类
```

## 为什么现在先做 Closed-set？

因为我们要先回答最基础的问题：

> 当一个训练时完全没见过的人只给 1～2 条注册语音时，当前 256D 表征到底能不能区分他的 30 个已注册短语？

如果连这个都不行，此时去设计复杂 Rejector 没有意义。

因此这一阶段：

```text
NO Unknown
NO Rejector
NO 新 Head
```

只测“30 类里面能不能选对”。

## 比较什么？

### Baseline A：Global 256D

```text
同用户 N-shot support
→ mean prototype
→ cosine similarity
→ 30-way top1
```

### Baseline B：Global + DTW

```text
Global Top-3
→ 只在 Top-3 上和同用户 support 做 DTW
→ 固定 lambda=0.50 融合
→ top1
```

DTW 的作用仍然是补充语速、局部拉长、时间错位等全局 embedding 容易压掉的信息。

## 为什么做 20 次 support selection？

因为 1-shot/2-shot 结果会受到“恰好选了哪一条注册录音”影响。

所以重复 20 次：

```text
不同 support
→ 剩余录音做 query
```

看方法是否稳定。

这些重复划分共享同一 DEV 数据，因此属于开发稳定性证据，不是 20 个独立测试集。

## 指标

- Macro-F1
- Accuracy
- Global Recall@3
- DTW 相对 Global 的增益
- 20 次里 DTW better/equal/worse 的次数

## 运行

先预计算 DEV speaker 内部 DTW：

```powershell
python scripts/precompute_h1_01_user_dtw.py
```

然后运行 H1-01：

```powershell
python scripts/run_h1_01_unseen_speaker_closedset.py
```

最后审计：

```powershell
python scripts/audit_h1_01_unseen_speaker_closedset.py
```

## 这一步成功后意味着什么？

如果 1-shot / 2-shot 有合理性能，我们第一次获得：

```text
真正 unseen-speaker personalized few-shot result
```

如果 DTW 仍稳定提升，则说明之前的 DTW 结论不仅在类别级模拟中成立，在真正的新用户个性化协议中也成立。

下一步才进入 H1-02：同用户 open-set personalized baseline。
