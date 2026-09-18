# PAPR 项目记忆

## Project Convention: 双项目边界与编号

**类型**: 架构约束 / 命名规范
**适用范围**: PAPR-SSL、未来 PAPR-Edge-Lite、共享音频工具

### 规范内容

- `papr_ssl` 是 Project 1 当前主工程；Agent 阶段从 `AGENT-A0` 编号，不复用 P/H 编号。
- Project 2 的专用工程尚未建立或纳入当前范围；`whisper/papr` 是用户明确排除的 Legacy 仓库。
- P6 absolute Gate 的结论固定为 `FAIL / BLOCKED`，H7 freeze PASS 不得覆盖该结论。
- 模型数据只通过 WAV、manifest、JSON 和 target export 等显式文件契约跨工程传递。
- 当前冻结 Teacher 表征是 WavLM-large `hidden_states[15]` + Attention DR；Global 为 256D，Temporal 为 `T' × 256D`。64D 仅指未来 Student embedding 目标。

### 示例

**正确**: 将 Runtime 已确认的 `canonical_text` 交给未来 Orchestrator。
**错误**: 让 Agent 直接读取 embedding、prototype 或更改拒识判定。

### 发现来源

2026-09-18 PAPR 双项目工程整理与用户范围修正。

## Decision Record: 基线仓库范围

**日期**: 2026-09-18
**问题**: 工作区包含多个 Git 仓库，哪些仓库属于本次正式整理范围。

### 选项分析

| 选项 | 优势 | 劣势 | 复杂度 |
| --- | --- | --- | --- |
| 仅整理 `papr_ssl` 与 `papr_audio_toolkit` | 严格服从用户范围，避免污染 Legacy 历史 | Project 2 代码入口仍待建立 | 低 |
| 同时整理 `whisper/papr` | 可一次覆盖更多历史材料 | 违反用户明确排除要求 | 高 |

### 决策

**选择**: 仅整理 `papr_ssl` 与 `papr_audio_toolkit`。

**理由**: 用户明确指出 `whisper/papr` 不用管。
**Trade-offs**: 本次只记录 Project 2 尚无纳入范围的专用仓库，不对 Legacy 仓库作任何状态承诺。

### 影响范围

- `docs/PAPR_REPOSITORY_AUDIT.md`
- `docs/PAPR_PROJECT_MAP.md`
- `docs/DEVELOPMENT_STATUS.md`
- Git 清理、测试、提交与 push 范围

### 撤销条件

仅当用户后续明确将 `whisper/papr` 或新的 PAPR-Edge-Lite 仓库重新纳入范围时重新评估。

## Glossary Entry: Teacher Target

**规范名称**: P7 Teacher Target
**别名/变体**: Teacher embedding、Teacher export target
**定义**: 当前冻结 Teacher 输出契约，包含 Global 256D float32 normalized embedding 与 Temporal `T' × 256D` float32 features；Temporal 使用共享 projection，downsample=3。
**适用范围**: Project 2 的 P7 Teacher Target Export 及其 manifest。
**禁止使用**: 不得把历史 64D Teacher 对比实验输出或未来 Student 64D embedding 称为当前 P7 Teacher Target。
**来源**: 2026-09-18 用户基线收尾修正规范。

## Glossary Entry: Head V2

**规范名称**: Head V2 设计方案
**别名/变体**: V2 Head、personalized head v2
**定义**: 已完成设计但尚未实现、尚未完成新用户/跨会话验证的后续方案；不等同于当前已完成的 H6 Runtime。
**适用范围**: Project 1 个性化判定头的后续实现与验证。
**禁止使用**: 不得将 Head V2 表述为已实现、已冻结或已经过跨会话验证。
**来源**: 2026-09-18 用户基线收尾修正规范。
