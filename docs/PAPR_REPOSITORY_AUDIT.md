# PAPR 双项目仓库与目录审计

审计日期：2026-09-18（Asia/Shanghai）

> 本文保留工程整理开始时的审计快照。基线收尾阶段已按用户提供的 URL 配置两个 `origin`，并以 unrelated-history merge 保留远端初始化 LICENSE；当前 remote、commit 与 push 结果以 `PAPR_BASELINE_FINALIZATION_REPORT.md` 为准。

## 1. 审计结论

工作区 `D:\02_开发项目\02_AI视觉与机器人\02_语音识别` 本身不是 Git 仓库。初始扫描发现 3 个彼此独立的仓库；用户随后明确要求 `whisper/papr` 不纳入本次管理。因此本次正式整理范围是 2 个仓库：

| 仓库 | 当前职责 | 分支 | HEAD | remote | 审计前状态 |
| --- | --- | --- | --- | --- | --- |
| `papr_ssl` | Project 1 主工程；同时保存已冻结 Teacher 研究主线 | `main` | `cfac59d` | 未配置 | 6 个已跟踪修改，384 个未跟踪文件 |
| `papr_audio_toolkit` | 共享音频数据处理、标注与导出工程 | `main` | 尚无提交 | 未配置 | 88 个未跟踪文件 |
| `whisper/papr` | Legacy 历史仓库 | `main` | `2845e66` | 未配置 | **排除：不修改、不清理、不测试、不提交、不 push** |

两个纳入范围的仓库都未配置 remote，因此当前不能执行正常 push。不得猜测远端地址，也不得创建“看起来正确”的 remote。

## 2. 实际目录与职责

### `papr_ssl`

- `src/papr_ssl/models/teacher/`：Wav2Vec2、WavLM、W2v-BERT 2.0 Teacher 主干及表示头。
- `src/papr_ssl/training/`：冻结 Teacher 的训练、评估、选择和运行清单。
- `src/papr_ssl/inference/`：H6 个性化 Runtime 与 Raw WAV Adapter。
- `demo/demo05_web/`：Demo-05B/C Web 后端、前端和浏览器 TTS 展示。
- `scripts/`：P2-P6、H0-H7、Demo-04/05 的审计、训练、冻结和演示入口。
- `tests/`：离线单元、静态契约和少量集成测试。
- `artifacts/`：约 12.0 GB 的实验产物、checkpoint、embedding/frame cache 和评估结果；受保护，不纳入 Git。
- `datasets/`：约 6.86 GB 的公开数据及归档；受保护，不纳入 Git。
- `.conda/`：约 167 MB 的本地 Python 环境；本地开发依赖，不纳入 Git。

`papr_ssl` 的现有 Git 基线只有 52 个已跟踪文件，当前工作区的大量源码、测试、文档和 Demo 尚未进入历史。本轮不得把“未跟踪”等同于“可删除”。

### `papr_audio_toolkit`

- `src/papr_audio_toolkit/audio/`：WAV 审计、归一化、重采样。
- `src/papr_audio_toolkit/segmentation/`：Energy VAD 与批量切分。
- `src/papr_audio_toolkit/embedding/`：WavLM embedding 提取。
- `src/papr_audio_toolkit/clustering/`：mutual-kNN 与聚类。
- `src/papr_audio_toolkit/annotation/`：segment manifest 数据结构。
- `scripts/`：Demo-01 至 Demo-03C 的审计、切分、聚类、人工标注和导出入口。
- `data/raw/wanghao/`：37 个真实原始音频，约 184 MB；不可修改、不可提交。
- `data/segments/wanghao/`：902 个文件，约 60 MB；派生音频，不提交。
- `data/manifests/wanghao_segments.jsonl`：901 行工程 manifest；属于可复现的小型元数据。
- `data/exports/`：90 个导出文件，约 5 MB；数据产物，不提交。
- `artifacts/`：276 个处理中间文件，约 19.7 MB；可再生成，不提交。

该仓库已从最初 bootstrap 演进到 Demo-03C，但尚无首个 commit；旧 README 的“仅骨架”描述已经过期。

### `whisper/papr`（排除范围）

初始只读扫描确认它是独立 Legacy 仓库，并含 Project 2 相关历史文档/占位目录。收到用户“这个仓库不用管”的指示后，本轮不再分析或处理其中任何文件；此前对其 README 与 `.gitignore` 的两处未提交编辑已精确撤回。它不作为本轮 Project 2 的权威代码入口。

## 3. Git 历史与远端

### `papr_ssl`

- 最近提交：`cfac59d chore: establish PAPR-SSL baseline`（2026-09-09）。
- 无 remote。
- 已跟踪修改包括三份 Teacher revision 固定、mask 语义/批处理修正和对应测试；属于审计前已有科研工程改动，本轮不改其逻辑。
- 未跟踪内容主要是 P2-P6/H/Demo 后续源码、脚本、测试和文档。

### `papr_audio_toolkit`

- `main` 为 unborn branch，尚无 commit。
- 无 remote。
- 全部工程文件目前未跟踪；原始音频、segments、exports 和 artifacts 已被忽略。

### `whisper/papr`（排除范围）

- 只记录初始发现：`main`、HEAD `2845e66`、无 remote。
- 不解释、不修复、不提交其现有 dirty status。

## 4. 大文件、checkpoint、数据与 artifacts

| 仓库 | 主要大文件/目录 | 处理建议 |
| --- | --- | --- |
| `papr_ssl` | `datasets/public/mdsc/*.zip`；`artifacts/p5_frame_cache/*.pt`；SSL `.safetensors` shards；105 个文件大于等于 10 MiB | 保留本地，继续由 `.gitignore` 排除；不得提交或删除 |
| `papr_audio_toolkit` | 5 个原始王灏 WAV 大于等于 10 MiB | 原始数据不可变；不提交、不复制、不删除 |
| `whisper/papr` | 初始扫描发现数据、artifact 和历史 DOCX 大文件 | 排除范围；本轮不处理 |

排除仓库中的任何大文件、manifest、删除或未跟踪状态均保持原样。

## 5. 敏感信息审计

- 未发现 `.env`、`.env.*`、API key、HF token、OpenAI key、Coze key、Bearer token 或密码赋值文件。
- `papr_ssl/demo/demo05_web/static/app.js` 中的 `token` 是前端异步运行序号（`const token = ++runId`），不是认证凭证。
- 文档中的 `Coze`、`secret`、`password` 和测试中的 `token` 均为说明文字或词法测试命中。
- 仍须在每次 commit 前对 staged diff 再做一次敏感信息扫描；扫描时只报告位置，不回显值。

## 6. 缓存、生成文件与明显临时项

可以安全清理：

- 两个纳入范围仓库代码区的 `__pycache__/`、`*.pyc` 和 `.pytest_cache/`；
- `papr_audio_toolkit/--workdir`（0 字节的命令残留）；
- 文档 QA 生成目录应由 `.gitignore` 排除，但因其中可能含交付验证材料，本轮仅忽略，不删除。

必须保留：

- 所有 `artifacts/`、`datasets/`、raw WAV、manifest、checkpoint 和历史报告；
- `papr_ssl` 的 Unknown Exposure v1-v5 历史；
- 用途不明确的 smoke artifacts、DOCX、报价文档和个人临时目录。

## 7. 重复与废弃候选

已确认三组重复：

1. `papr_ssl/training/teacher/p5_frame_data.py` 与 `src/papr_ssl/training/teacher/p5_frame_data.py` 完全相同；静态测试/历史路径仍引用根路径副本，暂不删除。
2. `papr_ssl/training/teacher/p5_dr.py` 与 `src/papr_ssl/training/teacher/p5_dr.py` 完全相同；同上。
3. `papr_ssl/inference/h6_personalized_runtime.py` 与 `src/papr_ssl/inference/h6_personalized_runtime.py` 完全相同；当前 Demo 使用 `src` 包路径，历史脚本/测试仍需继续核对，暂不删除。

Legacy 仓库中的相似实现属于排除范围，本轮不做重复分析、合并或移动。

## 8. `.gitignore` 审计建议

- `papr_ssl`：补充 `.conda/`、通用日志、额外模型导出格式、NumPy 大数组、生成型 cleanup inventory 和本地会话记忆目录。
- `papr_audio_toolkit`：保留 manifests 可跟踪；补充 `.env.*`、IDE/cache/log/model/local memory 规则；继续精确忽略 raw/segments/exports，而不是机械忽略全部 `data/`。
- `whisper/papr`：排除范围，不提出或执行 `.gitignore` 修改。

## 9. 风险与处理原则

1. 两个纳入范围的仓库均未配置 remote：可以形成本地 commit，但不能 push。
2. `papr_ssl` 存在大量审计前变更：提交必须使用显式路径清单，禁止 `git add .`。
3. 不修改 Teacher 结果、训练配置、Runtime 判定、intent_id、P/H 阶段编号或 P6 结论。
4. 历史 P6 absolute Gate 保持 `FAIL / BLOCKED`；H7 freeze PASS 不等于 P6 absolute Gate PASS。
5. Project 2 尚未实现，文档不得把目录占位写成已完成代码。

## 资料来源

- `README.md`、`.gitignore`、`pyproject.toml`（三个仓库）。
- `git status --short --branch --untracked-files=all`、`git log -10`、`git remote -v`、`git ls-files`。
- `src/papr_ssl/inference/h6_personalized_runtime.py`、`demo/demo05_web/app.py`。
- `whisper/papr` 仅用于初始发现；后续按用户要求排除。
- `papr_audio_toolkit/docs/DEMO_03C_DATASET_CORRECTION.md`。
- `PROJECT_CLEANUP_DRY_RUN.md` 与 `PROJECT_CLEANUP_DUPLICATES.csv`。
