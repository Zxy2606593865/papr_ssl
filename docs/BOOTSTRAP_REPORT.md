# PAPR-SSL 初始化报告

## 1. 新项目路径

```text
D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl
```

强制预检时该目录尚不存在，因此将其创建为独立项目。本次初始化仅以只读方式参考 `whisper/papr` 下的 Legacy 项目，没有导入、复制、修改或移动该项目，也未对其执行 Git 操作。

## 2. 目录结构

```text
papr_ssl/
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   └── teacher/
│       └── wav2vec2_mean.yaml
├── src/
│   └── papr_ssl/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   └── teacher.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── windowing.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── teacher/
│       │   │   ├── __init__.py
│       │   │   ├── encoder.py
│       │   │   ├── backbones/
│       │   │   │   ├── __init__.py
│       │   │   │   └── wav2vec2.py
│       │   │   └── heads/
│       │   │       ├── __init__.py
│       │   │       └── mean_dr.py
│       │   └── student/
│       │       └── __init__.py
│       └── losses/
│           ├── __init__.py
│           └── scaf.py
├── tests/
│   ├── contracts/
│   │   └── test_teacher_contract.py
│   ├── data/
│   │   └── test_windowing.py
│   ├── losses/
│   │   └── test_scaf.py
│   └── teacher/
│       ├── test_backbone.py
│       ├── test_encoder.py
│       └── test_mean_dr.py
├── scripts/
│   └── README.md
├── datasets/
│   └── README.md
├── artifacts/
│   └── README.md
└── docs/
    └── BOOTSTRAP_REPORT.md
```

仅创建了当前 Phase T0 所需文件，没有生成供未来训练/评估/工具实现使用的空目录。

## 3. 创建的文件

项目包含根目录元数据、一个真实 Teacher YAML、一个 `src/` package、四个实现模块、package exports、六个离线测试模块、本地数据/artifact 边界说明以及本报告。

主要实现文件包括：

- `src/papr_ssl/contracts/teacher.py`
- `src/papr_ssl/data/windowing.py`
- `src/papr_ssl/models/teacher/backbones/wav2vec2.py`
- `src/papr_ssl/models/teacher/heads/mean_dr.py`
- `src/papr_ssl/models/teacher/encoder.py`
- `src/papr_ssl/losses/scaf.py`

## 4. 已实现的算法与组件

### Wav2Vec2 backbone wrapper

- 仅当 model 未通过 dependency injection 提供时，才使用 Hugging Face `AutoModel.from_pretrained`。
- 返回选定的 hidden state `[B,T,D]`，以及布尔型 `[B,T]` frame mask。
- 支持配置 `model_name`、`revision`、`hidden_layer` 和 `freeze_backbone`。
- 若 model config 中提供 hidden-layer 范围，则先据此校验，并在真实 runtime hidden states 上再次校验。
- 默认冻结参数并保持 eval mode。
- 不执行 pooling、不投影到 64D、不分类、不计算 SCAF，也不执行 KD。

### Mean DR 基线

- 沿时间维计算 masked mean。
- 明确拒绝全 mask / 零有效帧样本。
- 使用一个 linear layer 将 `D → 64` 投影。
- 拒绝零投影向量，并返回 L2-normalized embedding。

### Sub-center ArcFace (SCAF)

- 存储形状为 `[C,K,D]` 的可训练 center。
- 对 embedding 和 center 进行归一化。
- 计算 `[B,C,K]` cosine similarity，并仅在每个类别内部沿 `K` 个 subcenter 归约。
- 仅对正确类别应用 angular margin。
- 使用 cosine 截断、稳定平方根和单调 ArcFace fallback。
- 返回 cross-entropy loss，也可返回供测试/指标使用的 `[B,C]` logits。
- Teacher inference 不引用 SCAF。

### TeacherEncoder

- 仅组合 backbone 和 representation head。
- 强制输入窗口为 float32 `[B,16000]`、有限值且范围在 `[-1,1]`。
- 返回通过校验的 `TeacherOutput` 契约。
- `encode()` is no-grad; deterministic behavior is verified in eval mode with a
  fake backbone.

### Window protocol

- 定义 16 kHz 下的 float32 `[B,16000]` waveform batch。
- 强制值为有限数且位于 `[-1,1]`，并要求 `sample_id` 非空且唯一。
- 暂未定义 recording、utterance、view、audio I/O 或 manifest 格式。

## 5. 明确尚未实现的内容

- WavLM, HuBERT, and Whisper Teacher
- Attention DR
- Knowledge distillation and `kd.py`
- BC-ResNet or any Student network
- Adapter/personalization algorithm
- Prototype or open-set evaluation implementation
- PTQ, QAT, INT8, TFLite, host, or ESP32 runtime
- Formal training/evaluation CLI
- Model download, feature cache, enrollment, or experiment artifacts
- Legacy PAPR package compatibility/imports

不存在名为 `wavlm.py`、`hubert.py`、`attention_dr.py`、`kd.py`、`bc_resnet.py`、`adapter.py` 或 `quantization.py` 的文件。

## 6. Teacher 契约

Input:

```text
waveform: torch.float32 [B,16000]
sample rate: 16000 Hz
finite and in [-1,1]
```

Output:

```text
TeacherOutput.embedding: torch.float32 [B,64]
TeacherOutput.frame_count: integer [B]
```

每行 embedding 都必须是有限值并经过 L2 单位归一化。每个 frame count 都必须为正数。输出契约不依赖 Wav2Vec2、SCAF 或 DR。

## 7. 测试结果

### Static compilation

```text
python -m compileall src tests
PASS
```

最终 compile 运行完成，没有警告。

### Offline unit tests

使用现有 `whisper_v0` Python 环境执行，没有安装依赖，也没有使用其 Legacy 项目 package：

| Area | Tests | Result |
|---|---:|---|
| Teacher contract | 1 | PASS |
| Window protocol | 2 | PASS |
| SCAF shape and correct-class margin | 2 | PASS |
| Wav2Vec2 config/injected fake model | 2 | PASS |
| Teacher composition/determinism | 1 | PASS |
| Mean DR masked mean/all-mask failure | 2 | PASS |
| **Total** | **10** | **10/10 PASS** |

现有本地 interpreter 成功解析 `pyproject.toml` 和 `configs/teacher/wav2vec2_mean.yaml`：**PASS**。

静态边界审计：

- Import of Legacy `papr` package under `src/`, `tests/`, or configs: none.
- Absolute Windows runtime path or `os.getcwd()` under runtime/config/tests:
  none.
- Dataset, artifact, checkpoint, Hugging Face cache, log, and local-learning
  ignore rules: verified.

## 8. 依赖要求

Proposed in `pyproject.toml`:

- Python `>=3.10`
- PyTorch `>=2.1`
- Transformers `>=4.40`
- PyYAML `>=6.0`

`unittest` 使用 Python 标准库。上述依赖为版本建议，并非 lock file。没有安装或升级任何 package。

## 9. 网络与模型下载状态

- Network used only to read current official Transformers API documentation.
- Wav2Vec2 model/checkpoint downloads: **not attempted**.
- Hugging Face cache writes: **none**.
- Real Wav2Vec2 forward smoke test: **deferred integration test**.
- All unit tests use dependency-injected fake models/backbones.

## 10. Git 状态

新项目初始化了独立 Git repository，分支为 `main`。没有暂存、提交、推送或变基任何文件。

预期的初始化状态是项目内容均未跟踪：

```text
?? .gitignore
?? README.md
?? artifacts/
?? configs/
?? datasets/
?? docs/
?? pyproject.toml
?? scripts/
?? src/
?? tests/
```

Local `.learnings/` and generated Python caches are ignored.

## 11. 已知风险

1. The real configured checkpoint has not been downloaded or executed, so its
   actual output frame count/dimension and processor mask policy still need an
   integration gate.
2. Frame-mask conversion uses the Hugging Face Wav2Vec2 model helper
   `_get_feature_vector_attention_mask` for padded batches. It is a private API
   and must be rechecked when Transformers is pinned or upgraded.
3. `revision: main` is not immutable. After the first successful real smoke
   test, pin a verified model commit hash for reproducibility.
4. The dependency proposal is intentionally broad and has not been resolved
   into a lock file or isolated environment.
5. SCAF mathematical unit tests pass, but optimization behavior and E0/E0b
   qualification are untested until a real dataset/training protocol exists.
6. The in-memory Teacher contract is frozen; the cross-framework NPZ/JSON
   teacher-target serialization contract remains a later, separately tested
   deliverable.

## 12. 建议的下一步

按以下顺序推进：

```text
real Wav2Vec2 checkpoint smoke test
→ confirm processor/mask policy and real [B,T,D]
→ pin checkpoint revision and dependency versions
→ Teacher Mean DR training
→ E0 baseline qualification
```

在真实 Teacher Mean DR 基线及其 64D target 契约通过资格确认前，不要开始 KD。

建议的初始 commit message（未执行）：

```text
feat: bootstrap PAPR-SSL Wav2Vec2 teacher baseline
```
