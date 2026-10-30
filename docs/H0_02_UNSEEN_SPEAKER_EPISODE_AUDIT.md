# H0-02 — 未见说话人个性化 episode 审计

H0-01 的结果：

```text
train speakers = 34
dev speakers = 4
train/dev overlap = 0
```

这说明官方 split 按说话人完全隔离。

按照“support 必须来自 train、query 必须来自 dev”的常规逻辑，这似乎无法进行。但对于**新用户的 few-shot 个性化**，说话人隔离实际上是理想设置。

正确的协议如下：

```text
HEAD-FIT:
    TRAIN speaker
    ├─ support from that speaker
    └─ different recordings as query

UNSEEN-SPEAKER DEVELOPMENT:
    DEV speaker (never used to fit shared Head parameters)
    ├─ reserve N recordings as personal support
    └─ use remaining recordings as query

FINAL TEST:
    unseen TEST speaker
    ├─ N labeled support recordings are allowed
    └─ remaining query recordings are scored
```

这是 few-shot 个性化的标准逻辑形式：测试用户可以提供少量带标签的 support 样本，而不会因此成为训练说话人。

因此，H0-02 审计每个说话人/短语是否至少有：

```text
N support recordings + 1 independent query recording
```

for N in:

```text
1 / 2 / 5 / 10 / 15
```

运行：

```powershell
python scripts/audit_mdsc_unseen_speaker_episodes.py
```

不会加载音频、embedding 或 TEST 样本。

由于 H0-01 未发现明确的 session/day 元数据，此处即使审计通过，也只能说明支持**用户级 / 未见说话人个性化**，不能说明支持跨 session 个性化。
