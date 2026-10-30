# PAPR-SSL

**PAPR-SSL 不是通用 ASR 系统。** 本仓库是面向服务端的个性化非典型短语理解项目，基于冻结的 SSL Teacher 研究路线和个性化运行时构建。

当前工作区包含两条独立的开发路线：

- **Project 1 — PAPR-SSL + Agent**：本仓库维护基本冻结的 WavLM-large `hidden_states[15]` Teacher 表征、采用 Attention DR 得到的 256D 全局与时序输出、Enrollment Memory、Prototype/Top-3 DTW、C/W/U 决策、三态运行时、`intent_id → canonical_text` 映射、Web Demo 和浏览器 TTS。Backend Orchestrator 与 Agent Adapter 是下一阶段工作。
- **Project 2 — PAPR-Edge-Lite**：当前工作区尚无纳入范围的专用仓库。Teacher target 导出、Student/KD、PCEN/LogMel、BC-ResNet 和边缘部署均尚未实现。Legacy `..\whisper\papr` 仓库明确不在本次范围内。

请先阅读：

- [当前开发状态](docs/DEVELOPMENT_STATUS.md)
- [双项目地图](docs/PAPR_PROJECT_MAP.md)
- [仓库审计](docs/PAPR_REPOSITORY_AUDIT.md)

## 当前冻结的 Teacher 表征

```text
microsoft/wavlm-large @ c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
→ outputs.hidden_states[15]
→ Attention DR
├─ Global: float32 256D, L2 normalized
└─ Temporal: T' × 256D, shared projection, downsample = 3
```

这是当前 Project-1 的表征基线，也是未来 P7 Teacher Target Export 的来源。P7 尚未实现。未来 Student 的 embedding target 为 64D；Teacher 256D 到 Student 64D 的降维与对齐属于 P8+ KD 设计。

## 历史 Teacher 对比契约

```text
same float32 waveform [B,16000]
                 │
       ┌─────────┼─────────────┐
       │         │             │
   Wav2Vec2    WavLM    W2v-BERT 2.0
       │         │             │
       └─────────┼─────────────┘
                 ↓
 exact outputs.hidden_states index
                 ↓
          shared Mean DR
                 ↓
      unit-normalized 64D
                 ↓
            shared SCAF
```

上图描述较早的 P1-P4 backbone/Mean-DR 对比契约，仍作为研究历史保留，但其 64D 输出不是当前 P7 Teacher target 契约。公平对比的边界是公共 waveform 输入和 `SSLBackboneOutput`；不同模型的内部 frontend 不要求一致。Wav2Vec2 和 WavLM 接收原始 `input_values`；W2v-BERT 2.0 则先通过自己的 feature extractor 将 waveform 转换为模型专用的声学 `input_features`。

## 隐藏层索引

`hidden_layer` 始终是 `outputs.hidden_states` 中精确的零起始索引。对当前 Hugging Face 模型，索引 `0` 通常是初始 embedding 输出，索引 `1..N` 是后续 encoder 输出。PAPR-SSL 不会隐式执行 `+1` 或 `-1`，不会截断非法索引，也不会回退到最后一层。真实模型的层数与索引必须在 T-Integration 中核实。

## 当前实现

- Wav2Vec2、WavLM 和 W2v-BERT 2.0 封装，支持离线依赖注入
- 共享 `SSLBackboneOutput`：float32 `[B,T,D]`、bool `[B,T]`、精确层索引
- 共享 masked Mean DR，输入维度 `D` 可配置，输出固定为 64D
- 共享 Sub-center ArcFace 配置：`K=3`、margin `0.2`、scale `30`
- Teacher 与固定窗口契约
- 离线 layer-sweep 扩展
- 冻结的 Attention-DR 运行时路径，包含 Enrollment Memory 和 Prototype 检索
- Top-3 DTW 证据、C/W/U 决策和 ACCEPT / CONFIRM / REJECT 输出
- `intent_id → canonical_text` 映射
- Raw-WAV adapter、王灏真实数据 Demo、Web backend/frontend 和浏览器 TTS

Backend Orchestrator、Agent Adapter/Coze 集成、Student/KD、PCEN/LogMel、BC-ResNet、PTQ/QAT、TFLite 和 ESP32 部署尚未实现。

## 配置状态

三个历史对比配置使用相同的 head、输出维度、SCAF 和 seed，仅 backbone 配置不同；其 Hugging Face revision 均固定为不可变的 commit hash。当前 H7 表征基线为上文所述的 WavLM-large/Attention-DR 256D 路径。

## 评估规则

**不得使用 test set 进行模型选择。** Backbone、hidden layer、checkpoint、DR 方案、threshold 和超参数只能依据 [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) 中的 development protocol 选择。roadmap 阶段 T9 前，`generic_test` 始终保持封存。

## 离线开发

```powershell
python -m compileall src tests
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

测试使用模拟模型和模拟 W2v-BERT feature extractor，不会安装软件包、加载数据集、下载 checkpoint 或训练模型。
