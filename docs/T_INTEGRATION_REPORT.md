# PAPR-SSL Teacher 集成报告

**Document:** `docs/T_INTEGRATION_REPORT.md`

**Project:** PAPR-SSL v1.0

**阶段：** P1 — Teacher 集成

**Status:** **PASS**
**Date:** 2026-09-10

---

## 1. 目的

本文档记录 PAPR-SSL **P1 — Teacher 集成**的完成情况和验收依据。

P1 的目标是在开展大规模公开数据实验、Teacher 选择、representation head 训练、SCAF 训练、knowledge distillation 或 Student 开发前，建立可复现且可供工程运行的 Teacher 侧 SSL backbone 集成层。

P1 验证三个预训练 SSL backbone：

- `facebook/wav2vec2-base`
- `microsoft/wavlm-large`
- `facebook/w2v-bert-2.0`

本报告范围仅限于**真实 checkpoint 集成和工程验证**。不包括 P2+ 数据集工程、Teacher 训练、Mean DR/SCAF 优化、Attention DR 对比、KD 或 Student 评估。

---

## 2. P1 验收标准

仅当满足以下所有条件时，P1 才算完成：

1. 已建立专用且可复现的 PAPR-SSL runtime 环境。
2. 三个真实预训练 SSL checkpoint 均可加载。
3. 三个模型均可完成真实只读 forward pass。
4. 所有 backbone 均返回通用 `SSLBackboneOutput` 契约。
5. Hidden-state 索引明确且通过验证。
6. Batch 和 mask 行为已确认，不会静默错位。
7. 已在真实 checkpoint 上检查正式候选 layer 的 VALID/INVALID 状态。
8. 已在实际开发机器上测量 CPU/GPU 工程资源用量。
9. Hugging Face revision 已从可变的 `main` 固定为不可变的 40 字符 commit hash。
10. 已提供真实 checkpoint 集成测试，且未混入默认离线单元测试套件。
11. 禁止从一种 backbone 实现静默回退到另一种实现。

以上条件均已满足。

---

## 3. 开发环境

### 3.1 本地开发机器

P1 资源测量数据采集自实际使用的 Windows 开发机器。

- Operating system: Windows 10/11 kernel family (`Windows-10-10.0.26200-SP0` as reported by Python)
- CPU: Intel64 Family 6 Model 183, GenuineIntel
- Logical CPU count: 24
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA runtime reported by PyTorch: 12.8

### 3.2 Python 环境

专用 Conda 环境：

```text
papr_ssl
```

P1 期间记录的主要版本：

```text
Python              3.11.16
torch               2.11.0+cu128
torchaudio          2.11.0+cu128
transformers        5.13.0
numpy               1.26.4
PyYAML              6.0.3
pytest              9.1.1
soundfile           0.14.0
safetensors         0.8.0
huggingface_hub     1.30.0
CUDA runtime        12.8
```

P1-10 还记录了：

```text
artifacts/p1_10/pip_freeze.txt
artifacts/p1_10/key_dependencies.json
```

这些文件提供完整的 Python package 快照和关键依赖快照，记录最终 pinned-checkpoint 验证所用的环境。

---

## 4. 通用 SSL Backbone 契约

三个 backbone 均通过通用输出契约集成：

```text
SSLBackboneOutput
├── features:    float32 [B, T, D]
├── frame_mask:  bool    [B, T]
└── hidden_layer: int
```

所选 `hidden_layer` 对应 `outputs.hidden_states` 中的**精确索引**。

P1 用于 smoke 和资源验证的 waveform 级输入契约如下：

```text
batch size      = 1
sample rate     = 16000 Hz
window length   = 1.0 s
samples/window  = 16000
waveform dtype  = float32
```

三个 backbone 可以保留各自的内部 frontend 实现。P1 不强行进行人为的 frontend 统一。

---

## 5. P1-04 — Wav2Vec2 真实集成

### 5.1 模型

```text
kind:        wav2vec2
model_name:  facebook/wav2vec2-base
```

Pinned revision:

```text
0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
```

Configured hidden layer:

```text
12
```

### 5.2 真实 forward 结果

Observed model structure:

```text
num_hidden_layers       = 12
hidden_size             = 768
hidden_states_count     = 13
valid hidden indices    = [0, 12]
```

Real wrapper output:

```text
features    = [1, 49, 768]
frame_mask  = [1, 49]
dtype       = float32 / bool
finite      = True
```

The selected raw hidden state and wrapper output shape match exactly.

### 5.3 Frontend

Wav2Vec2 接收原始 waveform 输入：

```text
input_values = [1, 16000]
```

P1 frontend 对齐检查：

```text
PASS
```

### 5.4 Load report 说明

从 `facebook/wav2vec2-base` 加载 `Wav2Vec2Model` 时，Transformers 会将若干仅用于预训练的参数（如 quantizer/projector 权重）报告为 `UNEXPECTED`。

P1 接受这一情况，因为项目有意使用基础 Wav2Vec2 表征模型，而不是完整的预训练目标组件。

未发现表征路径参数缺失或静默回退。

---

## 6. P1-05 — WavLM 真实集成

### 6.1 模型

```text
kind:        wavlm
model_name:  microsoft/wavlm-large
```

Pinned revision:

```text
c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
```

Configured hidden layer:

```text
24
```

### 6.2 真实 forward 结果

Observed structure:

```text
num_hidden_layers       = 24
hidden_size             = 1024
hidden_states_count     = 25
valid hidden indices    = [0, 24]
```

Real wrapper output:

```text
features    = [1, 49, 1024]
frame_mask  = [1, 49]
dtype       = float32 / bool
finite      = True
```

### 6.3 Frontend

WavLM 接收原始 waveform 输入：

```text
input_values = [1, 16000]
```

Frontend 对齐检查：

```text
PASS
```

---

## 7. P1-06 — W2v-BERT 2.0 真实集成

### 7.1 模型

```text
kind:        w2v_bert2
model_name:  facebook/w2v-bert-2.0
```

Pinned revision:

```text
da985ba0987f70aaeb84a80f2851cfac8c697a7b
```

Configured hidden layer:

```text
24
```

### 7.2 真实 forward 结果

Observed structure:

```text
num_hidden_layers       = 24
hidden_size             = 1024
hidden_states_count     = 25
valid hidden indices    = [0, 24]
```

Real wrapper output:

```text
features    = [1, 49, 1024]
frame_mask  = [1, 49]
dtype       = float32 / bool
finite      = True
```

### 7.3 Frontend

W2v-BERT 2.0 在 Transformer stack 前使用自身的 feature extractor。

Observed model inputs:

```text
input_features = [1, 49, 160]  float32
attention_mask = [1, 49]       int64
```

Wrapper output mask:

```text
frame_mask = [1, 49] bool
```

Frontend 与输出的时间维度已对齐。

P1 frontend alignment check:

```text
PASS
```

---

## 8. P1-07 — Batch / Padding / Mask 验证

当前 P1 public wrapper API 接受固定窗口 waveform，不提供显式 waveform 长度或 waveform-validity mask。

因此，无法可靠推断任意零填充尾部的语义：值为零的尾部既可能是 padding，也可能是有效静音。

因此，P1-07 区分以下情况：

- singleton 验证；
- 真实的多样本完整长度 batch 验证；
- 当前固定窗口契约下标记为 N/A 的真实 padded-batch 语义。

正式状态：

```text
PASS_WITH_PADDED_NA
```

### 8.1 Wav2Vec2

```text
batch waveform     = [3, 16000]
features           = [3, 49, 768]
frame_mask         = [3, 49]
valid frames       = [49, 49, 49]
singleton/batch max diff ≈ 2.62e-4
```

### 8.2 WavLM

```text
batch waveform     = [3, 16000]
features           = [3, 49, 1024]
frame_mask         = [3, 49]
valid frames       = [49, 49, 49]
singleton/batch max diff ≈ 3.21e-4
```

### 8.3 W2v-BERT 2.0

```text
batch waveform     = [3, 16000]
features           = [3, 49, 1024]
frame_mask         = [3, 49]
valid frames       = [49, 49, 49]
singleton/batch max diff ≈ 5.92e-4
```

singleton 与 batch 之间的微小差异属于 GPU 执行路径产生的正常浮点差异，不代表违反契约。

---

## 9. P1-08 — 候选 Layer 有效性

候选 layer 来自正式 Teacher layer-sweep 配置，并由项目的 `load_layer_sweep()` 实现展开。

此步骤仅检查结构有效性。

这**不代表** VALID layer 就是最佳表征 layer。

### 9.1 Wav2Vec2

Observed valid hidden-state range:

```text
[0, 12]
```

Formal candidates:

```text
8   VALID
12  VALID
```

Summary:

```text
VALID   = [8, 12]
INVALID = []
```

### 9.2 WavLM

Observed valid hidden-state range:

```text
[0, 24]
```

Formal candidates:

```text
8   VALID
12  VALID
16  VALID
20  VALID
22  VALID
```

Summary:

```text
VALID   = [8, 12, 16, 20, 22]
INVALID = []
```

### 9.3 W2v-BERT 2.0

Observed valid hidden-state range:

```text
[0, 24]
```

Formal candidates:

```text
8   VALID
12  VALID
16  VALID
20  VALID
22  VALID
```

Summary:

```text
VALID   = [8, 12, 16, 20, 22]
INVALID = []
```

Overall P1-08 result:

```text
PASS
```

这些候选集合可以进入 P4 表征质量对比。

---

## 10. P1-09 — CPU/GPU 资源基准测试

### 10.1 协议

Formal P1-09 benchmark protocol:

```text
batch size     = 1
window         = 1.0 s
sample rate    = 16000 Hz
samples        = 16000
inference      = torch.inference_mode()
warmup runs    = 10
timed runs     = 50
```

CPU 和 CUDA 测量分别在独立进程中执行。

Teacher 集成并未针对最低延迟进行优化；本阶段要求是在实际开发机器上具备工程可运行性。

### 10.2 最终基准汇总

| Backbone | CPU Mean | CPU P95 | CPU Peak RSS | GPU Mean | GPU P95 | GPU Peak Allocated | GPU Peak Reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wav2Vec2 Base | 37.07 ms | 39.45 ms | 1058.4 MB | 7.06 ms | 7.96 ms | 397.3 MB | 456 MB |
| WavLM Large | 106.45 ms | 141.77 ms | 1891.9 MB | 19.70 ms | 24.58 ms | 1261.6 MB | 1282 MB |
| W2v-BERT 2.0 | 201.25 ms | 217.57 ms | 2858.4 MB | 32.84 ms | 34.56 ms | 2231.2 MB | 2246 MB |

### 10.3 加载时间

| Backbone | CPU Load | GPU Load |
|---|---:|---:|
| Wav2Vec2 Base | 2.78 s | 2.75 s |
| WavLM Large | 3.68 s | 3.95 s |
| W2v-BERT 2.0 | 3.69 s | 5.06 s |

### 10.4 CPU 进程内存观察

Wav2Vec2:

```text
baseline RSS              ≈ 495.3 MB
after load RSS            ≈ 649.5 MB
observed peak RSS         ≈ 1058.4 MB
peak increment            ≈ 563.1 MB
```

WavLM:

```text
baseline RSS              ≈ 495.9 MB
after load RSS            ≈ 680.7 MB
observed peak RSS         ≈ 1891.9 MB
peak increment            ≈ 1396.0 MB
```

W2v-BERT 2.0:

```text
baseline RSS              ≈ 496.0 MB
after load RSS            ≈ 622.1 MB
observed peak RSS         ≈ 2858.4 MB
peak increment            ≈ 2362.4 MB
```

CPU 主机内存测量使用 Windows `GetProcessMemoryInfo` / `PROCESS_MEMORY_COUNTERS_EX`，统计进程级原生内存分配，而非仅统计 Python 分配。

### 10.5 GPU 内存观察

Wav2Vec2:

```text
resident allocated after load ≈ 360.3 MB
forward peak allocated        ≈ 397.3 MB
forward peak reserved         ≈ 456.0 MB
```

WavLM:

```text
resident allocated after load ≈ 1203.4 MB
forward peak allocated        ≈ 1261.6 MB
forward peak reserved         ≈ 1282.0 MB
```

W2v-BERT 2.0:

```text
resident allocated after load ≈ 2214.5 MB
forward peak allocated        ≈ 2231.2 MB
forward peak reserved         ≈ 2246.0 MB
```

### 10.6 工程结论

三个 Teacher backbone 均可在当前开发机器上正常运行。

即使是 P1 中开销最大的 W2v-BERT 2.0，也可在约以下时间内完成 1 秒输入的 forward：

```text
CPU mean ≈ 201 ms
GPU mean ≈ 32.8 ms
```

因此，本地开发不局限于简单 smoke test。该机器足以进行真实集成、调试、有限规模的 feature 提取和 Teacher 侧开发。

对于更大规模的公开数据 sweep、重复 layer 提取或长时间实验，server/GPU 环境可能更高效。这属于工作流考量，不构成 P1 工程阻塞项。

P1-09 result:

```text
PASS
```

---

## 11. P1-10 — 固定不可变 Revision

三个 Hugging Face checkpoint 均从可变 revision：

```text
revision: main
```

固定为不可变的 40 字符 commit hash。

### 11.1 最终固定模型表

| Kind | Model | Immutable revision |
|---|---|---|
| `wav2vec2` | `facebook/wav2vec2-base` | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` |
| `wavlm` | `microsoft/wavlm-large` | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` |
| `w2v_bert2` | `facebook/w2v-bert-2.0` | `da985ba0987f70aaeb84a80f2851cfac8c697a7b` |

固定后的 revision 已写入 Teacher YAML 配置。

### 11.2 固定 Revision 后的真实 smoke 验证

三个模型均从不可变 revision 重新加载，并成功完成真实 CUDA forward pass。

Wav2Vec2:

```text
revision          = 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
hidden states     = 13
features          = [1, 49, 768]
shape match       = True
frontend align    = True
SMOKE RESULT      = PASS
```

WavLM:

```text
revision          = c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
hidden states     = 25
features          = [1, 49, 1024]
shape match       = True
frontend align    = True
SMOKE RESULT      = PASS
```

W2v-BERT 2.0:

```text
revision          = da985ba0987f70aaeb84a80f2851cfac8c697a7b
hidden states     = 25
features          = [1, 49, 1024]
shape match       = True
frontend align    = True
SMOKE RESULT      = PASS
```

因此，固定后的 revision 保持了 P1 前期建立的真实 backbone 契约。

P1-10 result:

```text
PASS
```

---

## 12. P1-11 — 真实 Checkpoint 集成测试

P1-11 增加专用的真实模型集成测试层，不会影响默认离线单元测试套件。

Integration test files:

```text
tests/
├── conftest.py
└── integration/
    ├── __init__.py
    ├── README.md
    └── test_real_ssl_backbones.py
```

### 12.1 默认离线测试套件

默认单元测试不会加载大型 Hugging Face checkpoint。

Command:

```powershell
python -m unittest discover -s tests -v
```

Final result:

```text
Ran 22 tests
OK
```

Status:

```text
22 / 22 PASS
```

### 12.2 显式真实模型测试模式

真实 checkpoint 测试需要显式启用：

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode cache `
  --real-model-device cuda `
  -v
```

Final result:

```text
collected 4 items

test_pinned_real_ssl_backbone_contract[wav2vec2]   PASSED
test_pinned_real_ssl_backbone_contract[wavlm]      PASSED
test_pinned_real_ssl_backbone_contract[w2v_bert2]  PASSED
test_real_model_configs_are_immutably_pinned       PASSED

4 passed in 6.54s
```

Status:

```text
4 / 4 PASS
```

### 12.3 网络/cache 隔离

`--hf-mode cache` 明确使用本地 Hugging Face cache/离线模式。

因此，常规 P1 集成验证不会静默下载大型 checkpoint。

只有在有意请求时，才会显式启用网络访问。

### 12.4 集成测试断言

真实模型测试会验证：

- expected backbone kind;
- expected model identifier;
- expected immutable revision;
- revision is not `main`;
- revision length is 40 characters;
- expected configured hidden layer;
- expected model hidden-state depth;
- expected hidden dimension;
- successful real read-only forward;
- common `SSLBackboneOutput` shape/dtype contract;
- finite output features;
- valid frame mask;
- no silent model-kind fallback.

P1-11 result:

```text
PASS
```

---

## 13. P1 任务状态

| Task | Description | Result |
|---|---|---|
| P1-01 | Independent runtime environment | PASS |
| P1-02 | Dependency/version plan | PASS |
| P1-03 | Shared SSL smoke CLI | PASS |
| P1-04 | Wav2Vec2 real-checkpoint integration | PASS |
| P1-05 | WavLM real-checkpoint integration | PASS |
| P1-06 | W2v-BERT 2.0 real-checkpoint integration | PASS |
| P1-07 | Batch / padding / mask validation | PASS_WITH_PADDED_NA |
| P1-08 | Candidate-layer VALID/INVALID validation | PASS |
| P1-09 | CPU/GPU resource benchmark | PASS |
| P1-10 | Immutable revision pinning + dependency snapshot | PASS |
| P1-11 | Explicit real-model integration tests | PASS |
| P1-12 | Teacher integration report | PASS |

---

## 14. P1 最终 Gate 决策

### 14.1 功能集成

```text
Wav2Vec2      PASS
WavLM         PASS
W2v-BERT 2.0  PASS
```

三个 backbone 均已完成真实 forward pass，并满足通用表征契约。

### 14.2 Hidden-state 索引

```text
Wav2Vec2      13 observed hidden states, valid [0,12]
WavLM         25 observed hidden states, valid [0,24]
W2v-BERT 2.0  25 observed hidden states, valid [0,24]
```

当前 Transformers/model revision 下的 hidden-layer 索引没有歧义。

### 14.3 Mask 对齐

```text
Wav2Vec2      PASS
WavLM         PASS
W2v-BERT 2.0  PASS
```

未发现 temporal mask 与输出错位。

真正的变长 padded-batch 语义不在 P1 固定窗口 wrapper 契约范围内，因此明确记录为 N/A，而不做推断。

### 14.4 可复现性

```text
model revisions pinned      PASS
dependency versions saved   PASS
post-pin real smoke         PASS
```

所有生产候选 checkpoint 均使用不可变 commit hash。

### 14.5 测试隔离

```text
offline unit tests          22/22 PASS
real integration tests       4/4 PASS
real tests explicit opt-in   PASS
cache-only validation        PASS
```

真实模型测试不会影响常规离线测试流程。

### 14.6 工程可运行性

三个模型均可在实际开发机器上加载并成功运行。

Peak GPU allocation for the heaviest model, W2v-BERT 2.0, is approximately:

```text
2231.2 MB
```

and the measured CUDA mean forward latency for a 1-second input is approximately:

```text
32.84 ms
```

P1 不再有工程阻塞项。

---

# P1 FINAL RESULT: PASS

**Teacher 集成已验收并关闭。**

项目可以进入 **P2 — 公共数据工程**。

P2 should build on the frozen P1 contracts and must not silently alter:

- model identifiers;
- pinned model revisions;
- hidden-layer indexing semantics;
- waveform-level input contract;
- `SSLBackboneOutput`;
- default offline-test isolation;
- no-silent-fallback behavior.

Any later change to one of these items should be treated as a contract change and revalidated against the relevant P1 tests.
