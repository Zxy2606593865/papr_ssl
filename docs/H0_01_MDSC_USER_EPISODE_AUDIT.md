# H0-01 — MDSC 用户级 Personalized Episode 可行性审计

## 目标

在实现新的 Personalized Task Head 之前，先回答一个协议问题：

> 现有 MDSC 数据能否按照真实 `speaker_id` 构造“同用户注册 → 同用户查询”的 personalized episode？

这个阶段 **不训练任何新模型**。

## 为什么必须先做

历史 15-shot 流程是按 phrase 从整个 train 集合抽样，并不是按具体用户建立注册库。
因此历史结果只能描述为 Core30 类别级 few-shot enrollment simulation。

H0-01 必须检查：

```text
speaker_id
    ├─ TRAIN support
    └─ DEV query
```

是否在数据层面真实存在。

## 冻结 protocol

对用户 u 和短语 c：

```text
Support:
    speaker = u
    phrase = c
    split = train

Known Query:
    speaker = u
    phrase = c
    split = dev

Clean Unknown Query:
    speaker = u
    split = dev
    phrase ∉ Core30
```

Shot level：

```text
1 / 2 / 5 / 10 / 15
```

一个用户至少有两个可注册 intent，才计为可以构造多类别 personalized episode。

## 运行

在项目根目录解压文件后：

```powershell
python scripts/audit_mdsc_user_personalization.py
```

默认输入：

```text
artifacts/p2_04/manifests/mdsc.jsonl
artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl
```

默认输出：

```text
artifacts/h0_01_mdsc_user_audit/
```

包括：

```text
summary.json
speaker_summary.csv
speaker_phrase_core30.csv
shot_feasibility.csv
session_metadata_audit.json
H0_01_MDSC_USER_EPISODE_AUDIT.md
```

## 关键输出

终端会打印：

```text
train/dev speaker overlap
1/2/5/10/15-shot 可构造 episode 的 speaker 数
eligible speaker x phrase 数
known DEV query coverage
有 clean unknown 的 speaker 数
显式 session/day metadata 是否存在
```

## H0-01 判读规则

### 情况 A：train/dev speaker overlap = 0

MDSC 当前 split 无法构造 train-support → dev-query 的真实 user-level baseline。

H1 在 MDSC 上 BLOCKED。

### 情况 B：overlap > 0，且某 shot 有 episode speakers

可以进入 H1，对该 shot 运行真正 user-aware Global + DTW baseline。

### 情况 C：有 clean outside-Core30 unknown

可以进一步构造同用户 open-set episode。

### 情况 D：没有明确 session/day metadata

仍可做 user-level baseline，但不能称为 cross-session personalization。

## 科研边界

这个 audit 只回答“数据结构是否允许构造 user-level episode”。

它不证明：

- 当前模型在真实儿童上有效；
- 当前模型已经完成 personalization；
- 91.45% 是 user-level personalized result；
- MDSC train/dev 等价于跨天/跨 session；
- 新 Head 一定带来性能提升。

真实儿童、多 session 数据仍然是最终 personalized 结论的必要证据。
