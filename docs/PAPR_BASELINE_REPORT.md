# PAPR 双项目工程与 Git 基线报告

报告日期：2026-09-18（Asia/Shanghai）

> 收尾更新：原报告记录的是 remote 尚未提供时的基线快照。后续已按用户提供的 GitHub URL 配置两个 `origin`，并以 unrelated-history merge 保留远端初始化 LICENSE；最终 push 状态见 `PAPR_BASELINE_FINALIZATION_REPORT.md`。

## 1. 交付结论

本次完成了工作区内两个获准仓库的正式工程整理与本地 Git 基线：

- `papr_ssl`：Project 1（PAPR-SSL + 未来 Agent）主工程，并保存冻结 Teacher 研究主线。
- `papr_audio_toolkit`：共享音频审计、切分、聚类、标注、manifest 与数据导出工具链。

`whisper/papr` 已按用户明确要求排除：不修改、不清理、不测试、不提交、不 push，也不作为本轮 Project 2 权威入口。

两个纳入范围的仓库均未配置 remote，因此本轮只建立本地 commit，没有猜测、创建或修改远端，也没有执行 push。

## 2. 检查范围与初始 Git 状态

工作区根目录：`D:\02_开发项目\02_AI视觉与机器人\02_语音识别`

| 路径 | Git | 初始分支/HEAD | 初始状态 | 本轮处理 |
| --- | --- | --- | --- | --- |
| `papr_ssl` | 独立仓库 | `main` / `cfac59d` | 6 个已跟踪修改，384 个未跟踪文件 | 审计、整理、测试、提交 |
| `papr_audio_toolkit` | 独立仓库 | `main` / unborn | 88 个未跟踪文件，无历史 commit | 审计、整理、测试、首个提交 |
| `whisper/papr` | 独立 Legacy 仓库 | 初始只读发现为 `main` / `2845e66` | 原有 dirty 状态 | 用户排除；保持原样 |

工作区根目录本身不是 Git 仓库。详细审计见 [PAPR_REPOSITORY_AUDIT.md](PAPR_REPOSITORY_AUDIT.md)。

## 3. 项目边界与开发入口

### Project 1：PAPR-SSL + Agent

- 根目录：`papr_ssl/`
- Runtime：`src/papr_ssl/inference/h6_personalized_runtime.py`
- Raw WAV：`src/papr_ssl/inference/h6_raw_wav_adapter.py`
- Demo Backend：`demo/demo05_web/app.py`
- Demo Frontend/TTS：`demo/demo05_web/static/`
- Teacher Backbone / Representation：WavLM-large `hidden_states[15]` → Attention DR；Global 为 256D float32 L2-normalized embedding，Temporal 为 `T' × 256D`、shared projection、downsample=3；状态为基本冻结。
- 当前已具备 Enrollment、Prototype、Top-3 DTW、C/W/U、ACCEPT/CONFIRM/REJECT、`intent_id` 与 `canonical_text`。
- H6 Speech Runtime 已完成，是当前 Project 1 工程基线。
- Head V2 已完成设计，但尚未实现，也尚未完成新用户/跨会话验证。
- Backend Orchestrator 与 Agent Adapter 尚未实现；后续编号从 `AGENT-A0` 开始。

### Project 2：PAPR-Edge-Lite

当前工作区没有纳入范围的专用 Project 2 仓库。冻结 Teacher 来源位于 `papr_ssl`。P7 应导出 Global 256D float32 normalized embedding 与 Temporal `T' × 256D` features，并记录完整模型/投影/来源/hash/manifest 元数据；该导出尚未实现。64D 是未来 Student embedding 目标，Teacher 256D → Student 64D 的 alignment/projection 属 P8+ KD 设计。PCEN/LogMel、BC-ResNet Student、Temporal/Pooling、KD、量化与 Edge Runtime 都尚未实现。

后续建议在用户确认后建立独立 `PAPR-Edge-Lite` 工程；不得自动把已排除的 `whisper/papr` 设为入口。

完整地图见 [PAPR_PROJECT_MAP.md](PAPR_PROJECT_MAP.md)，当前状态见 [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md)。

## 4. 工程整理结果

### 已执行

- 补齐双项目职责、边界、入口、共享数据契约与下一阶段说明。
- 更新两个仓库的 README，使其与实际 Demo/Runtime/数据流程一致。
- 增强 `.gitignore`，精确隔离本地环境、缓存、日志、模型、数组、raw/segments/exports、artifact、dataset 与 cleanup inventory。
- 安全清理两个纳入范围仓库中的 34 个 Python/pytest 缓存目录。
- 删除 `papr_audio_toolkit/--workdir` 这一 0 字节命令残留。
- 保留三组已确认重复实现；它们仍被历史静态测试或脚本路径引用，当前不具备无风险删除条件。
- 将过时 Demo-05 v6/v7 视觉契约标记为历史跳过；保留 v9 当前契约，不修改运行逻辑。
- 将 Audio Toolkit 两个基于注释子串的跨项目 import 误报改为 AST import 检查。

### 明确保留且未移动/未删除

- `papr_ssl/artifacts/`（约 12.0 GB）、`datasets/`（约 6.86 GB）、`.conda/`（约 167 MB）。
- `papr_audio_toolkit/data/raw/wanghao/`（37 个原始 WAV，约 184 MB）、segments、exports 与 artifacts。
- 所有 checkpoint、embedding/frame cache、实验结果、Unknown Exposure v1-v5、manifest 与历史报告。
- `docs/论文/` 本地参考资料；继续忽略，不写入 Git 历史。
- cleanup inventory CSV；体积较大且可再生成，继续忽略。

### 有意纳入的小型数据

- `papr_ssl/demo/demo05_web/static/audio/presets/` 的 6 个 Demo 预置 WAV，总计约 0.425 MiB，用于可运行的前端演示。
- `papr_audio_toolkit/data/manifests/wanghao_segments.jsonl`，901 行，仅包含可版本化的工程元数据，不含原始音频。

## 5. `.gitignore`、大文件与敏感信息

### `.gitignore`

`papr_ssl` 新增/确认忽略：

- `.conda/`、`.ai-memory/*/`、日志与缓存；
- `*.onnx`、`*.tflite`、`*.npy`、`*.npz`；
- `PROJECT_CLEANUP_INVENTORY.csv`、`PROJECT_CLEANUP_DUPLICATES.csv`；
- `docs/论文/`、datasets、artifacts、checkpoint 与 Hugging Face cache。

`papr_audio_toolkit` 新增/确认忽略：

- IDE、mypy/ruff/pytest、环境、日志、临时目录和本地会话记录；
- 模型/数组导出格式；
- raw、segments、exports 和 artifacts 内容，同时保留目录 `.gitkeep` 与 `data/manifests/` 可跟踪。

### 大文件检查

- `papr_ssl` 本地扫描发现 105 个 ≥10 MiB 文件，均位于受保护的数据、artifact、checkpoint/cache 等范围，未暂存。
- `papr_audio_toolkit` 的 5 个 ≥10 MiB 原始 WAV 未暂存。
- 两次功能基线提交的 staged 文件均无 ≥5 MiB 单文件；PAPR-SSL 暂存总量约 2.001 MiB，Audio Toolkit 约 0.776 MiB。

### 敏感信息检查

- 未发现 `.env`、私钥、AWS key、GitHub token、OpenAI key、HF token、Bearer token 或明显的密码/客户端密钥文件。
- 每次 commit 前均对 staged 文件名与内容执行高置信规则扫描，只输出命中文件名；结果为 `NONE`。
- `demo/demo05_web/static/app.js` 的 `token` 是异步运行序号，不是凭证。

## 6. 验证结果

### PAPR-SSL

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
D:\anaconda3\envs\papr_ssl\python.exe -m unittest discover -s tests -v
```

结果：`Ran 342 tests`，`OK (skipped=5)`。

5 个跳过均为显式历史 UI 契约：Demo-05 v6 的 1 个旧布局断言与 v7 的 4 个旧样式断言；当前 v9 契约测试通过。未通过修改 Runtime、模型或评分规则来换取测试通过。

### PAPR Audio Toolkit

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
D:\anaconda3\envs\papr_ssl\python.exe -m unittest discover -s tests -v
```

结果：`Ran 34 tests`，`OK`。

### 未执行项

- 未运行需要下载模型、加载 checkpoint、读取大规模真实数据或 sealed generic-test 的实际 ML 推理/训练/集成流程。
- 未进行浏览器人工交互回归；本轮只修改工程组织、文档、忽略规则和静态测试契约，没有修改 Demo 运行逻辑。

## 7. Git 提交

| 仓库 | Commit | 说明 |
| --- | --- | --- |
| `papr_ssl` | `0aba732` — `建立 PAPR-SSL 研究与运行基线` | 纳入研究/训练/审计脚本、Runtime、Demo、配置与测试；未纳入数据和产物 |
| `papr_audio_toolkit` | `ded07b3` — `建立音频工具链工程基线` | 建立首个可复现代码/文档/manifest 基线；raw、segments、exports、artifacts 保持本地 |

本报告及工程治理文档位于随后的一次 PAPR-SSL 文档提交中；由于 commit 不能在自身内容中可靠写入自身 hash，应以 `git log -1 --oneline` 的当前结果为准，并在交付消息中记录该 hash。

## 8. Push 与最终状态

- `papr_ssl`：无 remote，未 push。
- `papr_audio_toolkit`：无 remote，未 push。
- 未创建、猜测或覆盖任何 remote。
- 功能代码与可版本化工程资料均进入本地 Git；数据、artifact、checkpoint、环境和生成型 inventory 继续由 `.gitignore` 隔离。

## 9. 科研结论保护

- P/H 阶段编号未重排。
- 历史 P6 absolute Gate 仍为 `FAIL / BLOCKED`。
- K=2 exact calibration 仍为 659,344 个候选对、0 个可行解。
- H7 freeze/audit PASS 只表示配置封存通过，不改变 P6 absolute Gate 结论。
- 未修改模型架构、训练目标、已冻结配置、Runtime 决策、intent ID 或实验结果。

## 10. 下一步建议

1. 完成本次基线收尾的普通 `git push -u origin main`；不要 force push。
2. Project 1 下一阶段从 `AGENT-A0` 开始冻结 Runtime → Orchestrator 契约，再实现三态路由和 Agent Adapter。
3. Project 2 下一阶段从 P7 Teacher Target Export 开始，按 Global 256D / Temporal `T' × 256D` 契约冻结元数据，再进入 P8+ Student/KD。
