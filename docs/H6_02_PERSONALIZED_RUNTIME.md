# H6-02 — Personalized Runtime Core

## 当前环节在干什么？

这一步**不再训练模型**。

我们把已经验证过的研究组件正式封装成一个“运行时”：

```text
User Enrollment Memory
        +
Query Features
        ↓
Global Prototype
        ↓
Top-3 DTW
        ↓
C/W/U Decision Head
        ↓
H6 Policy
        ↓
ACCEPT / CONFIRM / REJECT
        ↓
intent_id / canonical_text
```

也就是说，从这一阶段开始，我们不是继续“造模型”，而是在把模型变成可以被 Agent / 前端调用的软件模块。

## 为什么 H6-02 先用 Cached Feature，而不是马上接麦克风？

这是故意的。

现在我们已经科学验证过：

```text
WavLM[15]
→ Attention 256D
→ Temporal 256D
```

如果此时同时加入“真实 WAV 读取、重采样、WavLM 原始推理、注册文件存储、Task Head”
五件新东西，一旦结果不对，很难定位到底是哪一层出错。

所以 H6-02 先固定：

```text
输入 = 已验证的 256D Global + Temporal Feature
```

只测试新的软件部分：

```text
Enrollment Memory
Runtime
JSON output
三态决策
```

它 PASS 以后，再加 Raw WAV Adapter。

这符合我们一直采用的原则：

> 一次只新增一个变量。

## Runtime 接口

核心类：

```python
UserMemory
PersonalizedRuntime
```

注册：

```python
memory.enroll(
    intent_id="drink_water",
    canonical_text="我要喝水",
    global_embedding=...,
    temporal_sequence=...,
)
```

推理：

```python
result = runtime.predict_feature(
    memory=memory,
    query_global=...,
    query_temporal=...,
)
```

输出：

```json
{
  "status": "ACCEPT",
  "intent_id": "drink_water",
  "canonical_text": "我要喝水",
  "shot": 2,
  "decision_scores": {
    "C": 0.8,
    "W": 0.1,
    "U": 0.1
  },
  "scores_are_calibrated_probabilities": false,
  "candidates": []
}
```

注意：

```text
decision_scores ≠ calibrated probabilities
```

因此正式接口里没有把它们命名为 `confidence_probability`。

## REJECT 的输出

REJECT 时：

```json
{
  "status": "REJECT",
  "intent_id": null,
  "canonical_text": null
}
```

不会强行给 Agent 一个文本意图。

## CONFIRM 的输出

CONFIRM 时保留 top1：

```json
{
  "status": "CONFIRM",
  "intent_id": "...",
  "canonical_text": "..."
}
```

这样前端可以直接问：

```text
“你是想说‘我要喝水’吗？”
```

## 用户身份

当前系统假设：

```text
user_id 已由 App / 教师 / 登录账号 / 用户档案选择
```

当前 Head **不负责 speaker authentication（说话人身份认证）**。

这必须在论文和 Demo 中说清楚。

## H6-02 Demo

为了不碰 generic_test，Demo 从一个 unseen DEV speaker 构造：

```text
20 registered phrases
2-shot enrollment
10 unregistered phrases
```

然后走真正的：

```text
save memory
→ reload memory
→ predict
→ JSON
```

## 运行

```powershell
python scripts/run_h6_02_cached_runtime_demo.py
python scripts/audit_h6_02_runtime.py
```

如果之前跑过：

```powershell
python scripts/run_h6_02_cached_runtime_demo.py --overwrite
```

## H6-02 PASS 后

下一步才增加：

```text
H6-03 Raw WAV Feature Adapter

WAV
→ 16 kHz preprocessing
→ frozen WavLM[15]
→ frozen Attention/256D
→ H6-02 Runtime
```

而且 H6-03 必须先做：

```text
Raw WAV重新提取的特征
vs
P5历史 cached feature
```

的 parity check。

只有 parity PASS，才允许用真实儿童 WAV 做演示。
