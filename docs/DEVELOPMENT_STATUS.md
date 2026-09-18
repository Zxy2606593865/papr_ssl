# PAPR 当前开发状态

状态日期：2026-09-18

## Project 1：PAPR-SSL + Agent

| 能力 | 状态 | 入口 |
| --- | --- | --- |
| Teacher 主线 | 基本冻结 | `docs/H7_00_FINAL_FREEZE.md`、`configs/teacher/` |
| Speech Runtime | 已完成/基本冻结 | `src/papr_ssl/inference/` |
| Enrollment / Prototype / Top-3 DTW | 已完成 | `h6_personalized_runtime.py` |
| Open-set C/W/U | 已完成 | `CWUDecisionHead` |
| ACCEPT / CONFIRM / REJECT | 已完成 | H6 Runtime |
| `intent_id → canonical_text` | 已完成 | Enrollment Memory / Runtime 输出 |
| Demo Backend | 已完成 | `demo/demo05_web/app.py` |
| Demo Frontend | Demo-05B/C v9 已完成 | `demo/demo05_web/static/` |
| TTS Demo | 已完成 | 浏览器 Web Speech API |
| Backend Orchestrator | 待开发 | 下一阶段 |
| Agent Adapter / Coze | 待开发 | 从 `AGENT-A0` 编号 |

P6 absolute Gate 的历史结论仍为 `FAIL / BLOCKED`。H7 的 freeze/audit PASS 仅表示最终配置封存通过，不改变 P6 科研结论。

## Project 2：PAPR-Edge-Lite

| 能力 | 状态 | 入口 |
| --- | --- | --- |
| 冻结 Teacher 来源 | 已存在/研究基线 | `papr_ssl` 当前 Teacher 主线 |
| Project 2 专用仓库 | 未发现纳入范围的仓库 | `PAPR-Edge-Lite` 尚未建立/确认 |
| Teacher Target Export | 下一阶段/待开发 | `P7` |
| Student | 待开发 | 尚无纳入范围的实现 |
| KD | 待开发 | `P8+` |
| PCEN / LogMel | 待开发 | 尚无实现 |
| BC-ResNet / Temporal / Pooling | 待开发 | 尚无实现 |
| 64D Student Embedding | 待开发 | 尚无实现 |
| Edge Runtime / INT8 / TFLite | 待开发 | 尚无纳入范围的实现 |

`whisper/papr` 已按用户要求排除，不是本轮开发入口。

## Shared Data

- 王灏原始数据：`../papr_audio_toolkit/data/raw/wanghao/`，37 个 WAV，保持不可变。
- 切分数据：`../papr_audio_toolkit/data/segments/wanghao/`，派生数据不提交。
- 主 manifest：`../papr_audio_toolkit/data/manifests/wanghao_segments.jsonl`，901 行。
- Demo 导出：`../papr_audio_toolkit/data/exports/wanghao_demo` 与 `wanghao_demo_v2`；v1 保留科研基线，v2 为人工纠正工程集。
- PAPR-SSL 的 datasets/artifacts 继续本地保存并由 Git 忽略；本轮不复制、不移动、不删除。

## 下一步开发

### Project 1

1. `AGENT-A0`：冻结 Runtime → Orchestrator 的输入/输出契约。
2. `AGENT-A1`：实现 Agent Adapter，明确 ACCEPT/CONFIRM/REJECT 三态路由。
3. `AGENT-A2`：接入 Coze/Agent，并保持 Agent 不接触 embedding 与拒识判定。

### Project 2

1. P7：定义并导出 Teacher target contract（64D、dtype、版本、hash、manifest）。
2. P8+：实现 PCEN/LogMel、BC-ResNet Student、Temporal/Pooling 与 KD。
3. 在离线契约和指标通过后，再进入量化与 Edge deployment。

## 开发入口

- 双项目地图：[PAPR_PROJECT_MAP.md](PAPR_PROJECT_MAP.md)
- 仓库审计：[PAPR_REPOSITORY_AUDIT.md](PAPR_REPOSITORY_AUDIT.md)
- 本次基线报告：[PAPR_BASELINE_REPORT.md](PAPR_BASELINE_REPORT.md)
- Project 1 根 README：`../README.md`
- Shared Audio Toolkit：工作区兄弟目录 `papr_audio_toolkit/`
- Project 2：待确认/建立独立 `PAPR-Edge-Lite` 工程；本轮不使用 `whisper/papr`
