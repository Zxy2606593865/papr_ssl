# H4-00 — Shared Evidence Head Feature Readiness

## 当前在干什么？

H2-00 的结论已经很明确：

```text
1/2-shot：可以完整覆盖 30 类，但不能可靠估计 user×phrase 方差。
3/4-shot：可以估计方差，但只能覆盖约 1/3 的 speaker×phrase。
5-shot：不可用。
```

所以 **LOO + Shrinkage 不能作为 MDSC 正式 30 类 1/2-shot 主线**。

同时，Temporal Medoid 也需要至少 3 条以上注册音频才有“选择代表模板”的意义：

```text
1-shot：只有 1 个模板，无从选择。
2-shot：两条都是模板，也没有 medoid 选择问题。
```

因此在当前公共数据上：

```text
H2 完整个性化方差 → 留作高-shot/真实儿童数据验证
H3 Temporal Medoid    → 留作高-shot/真实儿童数据验证
```

公共数据主线直接进入 H4：

```text
Shared Evidence Fusion / C-W-U Decision
```

## 为什么 H4-00 还要先审计一次？

H4 应该在 34 个 TRAIN speaker 上拟合，然后在 4 个 unseen DEV speaker 上评价。

这要求 TRAIN 的每条 Core30 音频都有：

```text
256D global embedding
+
temporal feature
```

H1-01/02 使用的是 DEV 442 条现成 embedding，因此不能默认 TRAIN 3756 条也已经完整导出。

H4-00 就检查：

> 现有 artifacts 是否已经覆盖全部 Core30 TRAIN/DEV 的 global + temporal 特征？

如果全部存在，下一步直接训练 H4-01。

如果缺失，只补导出缺的 frozen features，不重训 Backbone/Neck。

## 运行

```powershell
python scripts/audit_h4_feature_readiness.py
```

重点看：

```text
global TRAIN ready
global DEV ready
temporal TRAIN ready
temporal DEV ready
H4 full feature fit ready
```

这一步不训练模型，也不访问 generic_test。
