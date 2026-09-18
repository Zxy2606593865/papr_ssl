# H2-00 — Enrollment Statistics Readiness Audit

## 当前在干什么？

H1-02 已经证明，最简单的个性化 Open-set baseline 很强。

下一步原计划是 H2：

```text
LOO + Shrinkage
```

也就是利用“这个用户说这个短语时到底稳不稳定”来改善拒识。

但这里有一个统计学问题：

```text
1-shot → 只有1条注册音频，根本无法估计用户自己的波动。
2-shot → 两个 Leave-One-Out cosine 是对称的，几乎没有可用的类内方差信息。
3-shot+ → 才开始有资格估计 user × phrase 的离散程度。
```

所以不能为了按开发文档顺序推进，就硬在 1/2-shot 上声称做了“个性化方差”。

H2-00 只做一件事：

> 审计 MDSC 到底有多少 `speaker × phrase` 能提供 3/4/5-shot support，同时还留至少 1 条独立 query。

## 为什么要这么干？

如果 3-shot+ 足够多：

```text
可以在 MDSC 上真正验证 LOO + Shrinkage。
```

如果不足：

```text
H2 的完整个性化不确定性实验应留到真实儿童多次注册数据；
MDSC 下一步直接进入 Shared Evidence Fusion / C-W-U Head。
```

这能避免做一个“数学上看起来完整，但数据根本不支持”的实验。

## 运行

```powershell
python scripts/audit_h2_enrollment_stats_readiness.py
```

输出：

```text
artifacts/h2_00_enrollment_stats_readiness/
    summary.json
    readiness_summary.csv
    speaker_phrase_counts.csv
```

## 判读重点

看 DEV 的：

```text
support_n = 3
full_30intent_speakers
eligible_speaker_phrase_pairs
eligible_pair_fraction
```

如果 `full_30intent_speakers = 0`，说明 3-shot 不能作为完整 Core30 unseen-speaker benchmark。

即使只有部分 pair 可用，也只能作为局部统计消融，不能替代当前正式 1/2-shot benchmark。
