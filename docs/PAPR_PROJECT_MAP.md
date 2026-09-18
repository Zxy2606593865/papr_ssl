# PAPR 双项目地图

## 总原则

PAPR 当前有两条独立主线。Project 1 负责服务端个性化特殊语音理解与未来 Agent 编排；Project 2 负责边缘端 Student/KD 与部署。共享数据和工具通过 manifest/WAV/JSON 等显式文件契约连接，不通过跨仓库内部 import 连接。

```text
papr_audio_toolkit（共享数据工程）
        │ WAV / manifest / intent mapping / export
        ├──────────────────────────┐
        ▼                          ▼
Project 1                      Project 2
papr_ssl                       PAPR-Edge-Lite（尚无纳入范围的专用仓库）
Speech Runtime                 Teacher target export（待开发）
→ Backend/Frontend             → Student/KD（待开发）
→ Agent（下一阶段）             → Edge runtime（待开发）
```

## Project 1：PAPR-SSL + Agent

### 目标链路

```text
特殊语音
→ PAPR-SSL
→ ACCEPT / CONFIRM / REJECT
→ intent_id
→ canonical_text
→ Backend Orchestrator
→ Agent / Coze
→ 对话 / 业务执行
→ TTS
→ 前端
```

### 当前模块与实际位置

| 模块 | 当前位置 | 状态 |
| --- | --- | --- |
| Speech Runtime | `papr_ssl/src/papr_ssl/inference/h6_personalized_runtime.py`、`h6_raw_wav_adapter.py` | 已完成并冻结研究配置 |
| Enrollment Memory | `UserMemory`、`IntentMemory`、`EnrollmentExample`（同上） | 已完成 |
| Prototype | `PersonalizedRuntime._candidate_evidence`（同上） | 已完成 |
| Top-3 DTW | `PersonalizedRuntime(top_k=3)` 与 `_dtw_distance`（同上） | 已完成 |
| C/W/U | `CWUDecisionHead`（同上） | 已完成 |
| ACCEPT / CONFIRM / REJECT | `CWUDecisionHead.decide`、`PersonalizedRuntime.predict_feature` | 已完成 |
| `intent_id → canonical_text` | `UserMemory.enroll/load/save` 与 Runtime 输出 | 已完成 |
| Raw WAV Adapter | `papr_ssl/src/papr_ssl/inference/h6_raw_wav_adapter.py` | 已完成 |
| 王灏真实数据 Demo | `papr_ssl/artifacts/demo_04a_wanghao_runtime_v2/` 与对应 scripts/docs | 已完成；artifact 不入 Git |
| Web Backend | `papr_ssl/demo/demo05_web/app.py` | 已完成 Demo 版本 |
| Web Frontend | `papr_ssl/demo/demo05_web/static/` | Demo-05B/C v9 已完成 |
| TTS | `papr_ssl/demo/demo05_web/static/app.js` 的浏览器语音合成 | Demo 已完成；不是服务端 TTS |
| Backend Orchestrator | 尚无独立模块 | 下一阶段 |
| Agent Adapter / Coze | 尚无实现；仅文档/界面文案提及 | 下一阶段，编号从 `AGENT-A0` 开始 |

### 不可跨越的边界

- Agent 只接收 Runtime 已确认的 `canonical_text` 和必要业务元数据。
- Agent 不读取 embedding、prototype、DTW 序列或 C/W/U 内部特征。
- `REJECT` 不进入内容猜测；`CONFIRM` 必须由产品交互确认。
- Agent 不修改 ACCEPT / CONFIRM / REJECT 判定。
- Teacher/P/H 历史阶段保持冻结；Agent 使用 `AGENT-A0/A1/...` 独立编号。

## Project 2：PAPR-Edge-Lite

### 目标链路

```text
PCEN / LogMel
→ BC-ResNet Student
→ Temporal Module
→ Pooling
→ 64D Embedding
→ Prototype Similarity
→ optional DTW
→ ACCEPT / CONFIRM / REJECT
→ intent_id / canonical_text
```

### 当前模块与实际位置

| 模块 | 当前位置 | 状态 |
| --- | --- | --- |
| 冻结 Teacher 来源 | `papr_ssl` 的冻结 Teacher 主线 | 已存在；导出契约尚未定义 |
| Project 2 专用仓库 | 工作区未发现顶层 `PAPR-Edge-Lite` 仓库 | 尚未建立/未纳入本轮 |
| Teacher target contract/export | 尚无纳入范围的实现 | 未实现；下一阶段 |
| Student training | 尚无纳入范围的实现 | 未实现 |
| KD | 尚无源码 | 未实现 |
| PCEN / LogMel | 尚无源码 | 未实现 |
| BC-ResNet | 尚无源码 | 未实现 |
| Temporal module / Pooling | 尚无源码 | 未实现 |
| 64D Student embedding | 尚无源码/配置 | 未实现 |
| Prototype / optional DTW | 可复用行为契约尚待定义 | Edge 版本未实现 |
| Edge runtime | 尚无纳入范围的实现 | 未实现 |
| INT8 / TFLite / ESP32-S3 | 尚无纳入范围的实现 | 未实现 |

`whisper/papr` 是用户明确排除的 Legacy 仓库，不作为本轮 Project 2 开发入口，也不做任何整理、测试或 Git 操作。

Project 2 后续顺序保持：

```text
P7 Teacher Target Export
→ P8+ Student / KD
→ PAPR-Edge-Lite Runtime
→ Edge Deployment
```

## Shared：共享数据、契约与工具

| 共享内容 | 权威位置 | 共享方式 |
| --- | --- | --- |
| 王灏原始音频 | `papr_audio_toolkit/data/raw/wanghao/` | 本地只读，不复制进代码仓库 |
| 切分音频 | `papr_audio_toolkit/data/segments/wanghao/` | 派生数据，不提交 |
| Segment manifest | `papr_audio_toolkit/data/manifests/wanghao_segments.jsonl` | JSONL 文件契约，可版本化的小型元数据 |
| Enrollment/Query export | `papr_audio_toolkit/data/exports/` | 通过目录/manifest 交付，不跨仓库 import |
| intent mapping | Audio Toolkit 的人工确认/导出 manifest | `intent_id` 与 `canonical_text` 成对冻结 |
| Public data protocol | `papr_ssl/configs/data/`、`papr_ssl/src/papr_ssl/data/` | 配置、schema、审计脚本 |
| Evaluation utility | 各项目保留与自身模型耦合的评估入口 | 公共指标名称对齐，不复制模型内部代码 |
| Project 2 Teacher target | 未来由 Project 1/冻结 Teacher 导出 | 明确版本、维度、dtype、hash 和 manifest；不直接 import Teacher 内部实现 |

## 阶段编号约束

以下编号不得重排：

- P0 baseline freeze
- P1 Teacher checkpoint integration
- P2 public data engineering
- P3 frozen SSL + Mean DR + SCAF
- P4 backbone/layer selection
- P5 Mean vs Attention DR
- P6 Teacher E0/E0b qualification/freeze
- P7 target export
- P8+ Student/KD

H 系列保持原编号。历史 P6 absolute Gate 的结论保持 `FAIL / BLOCKED`，不得因 H7 freeze 或工程整理改写为 PASS。
