# H6-01 — ACCEPT / CONFIRM / REJECT

## 当前在干什么？

H5 已经证明 C/W/U Head 有一个很明显的效果：

```text
Unknown 更不容易被误接受
Wrong Intent 也下降
```

但代价是：

```text
一部分原本能识别的 Known 被“保守地挡住了”
```

H5 把这些全部算成 Reject，所以 CA 下降约 2～3 个百分点。

真实产品没必要这么二元。

因此 H6 加入中间状态：

```text
ACCEPT
CONFIRM
REJECT
```

## 三个状态分别什么意思？

### ACCEPT

系统证据非常明确：

```text
“我认为就是这个已注册短语，可以自动执行。”
```

### REJECT

系统很确定它不像当前注册集合：

```text
“这个语音不应该映射成任何已注册意图。”
```

### CONFIRM

介于两者之间：

```text
“我有候选，但没有足够把握自动执行。”
```

例如前端可以显示：

```text
你是想说“我要喝水”吗？
```

或者让照护者/按钮确认。

## 为什么 CONFIRM 很重要？

H5：

```text
CA下降
UR大幅上升
```

说明 Head 其实学会了“谨慎”，但 H5 只有接受/不接受两个出口。

H6 把“不确定”和“明确未知”拆开：

```text
不确定 → CONFIRM
明确未知 → REJECT
```

这样才是真正适合 Agent 的 Task Head。

## 阈值怎么定？

仍然只用 6 个 TRAIN calibration speakers。

ACCEPT：

```text
Unknown 自动接受率 <= 10%
```

的前提下，让 Correct Accept 尽量高。

REJECT：

```text
Known 自动拒绝率 <= 5%
```

的前提下，让 Unknown Reject 尽量高。

剩余全部：

```text
CONFIRM
```

这两个 10% / 5% 是当前开发 operating point，不是产品承诺。
后续真实儿童数据应画完整 risk-coverage 曲线重新选。

## 运行

```powershell
python scripts/run_h6_01_three_state_policy.py
python scripts/audit_h6_01_three_state_policy.py
```

## 重点看什么？

Known：

```text
CorrectAccept
WrongAccept
Confirm
Reject
```

四项相加必须 = 1。

Unknown：

```text
Accept
Confirm
Reject
```

三项相加必须 = 1。

最重要的是：

```text
WrongAccept 尽量低
Unknown Accept 尽量低
Known Reject 不要太高
不确定样本允许进入 CONFIRM
```

CONFIRM 不算自动识别成功。

如果这一阶段表现合理，下一步就是把这套策略封装成真实推理接口：

```text
enroll(user_id, ...)
predict(user_id, audio)
→ ACCEPT / CONFIRM / REJECT
→ intent_id / canonical_text / candidates
```
