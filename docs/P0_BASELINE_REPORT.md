# PAPR-SSL P0 基线报告

Date: 2026-09-09 (Asia/Shanghai)

P0 冻结现有的 Teacher-first 框架。本阶段不下载或运行真实 SSL checkpoint，不启动 P1，不修改模型数学定义，也不添加新功能。

## 1. 仓库状态

- Repository：已初始化的 Git worktree
- 分支：`main`
- P0 之前的历史：没有 commit（`HEAD` 不存在）
- P0 之前已跟踪的文件：无
- 基线作用：首个正式项目 commit
- Legacy `whisper/papr` 项目：未修改，也未执行 Git 操作
- Python package import：PASS（`papr_ssl 0.1.0` 和公开 Teacher API）

暂存前，整个项目以未跟踪的顶层路径出现。Git 忽略了 `.learnings/` 和 Python cache。基线暂存审计排除了本地数据、artifacts、模型权重、HF cache、临时文件以及本地 `docs/论文/` 参考资料库。

## 2. 编译检查

Command:

```text
python -m compileall src tests scripts
```

结果：**PASS**。`src/`、`tests/` 和现有 `scripts/` 目录均已处理，没有语法错误或警告。

## 3. 测试结果

项目 README 规定使用标准库 unittest discovery；审计环境未安装 pytest。测试通过将本地 `src/` 目录加入进程 import path 执行，全程未联网，也未加载模型。

```text
tests collected: 22
passed: 22
failed: 0
errors: 0
skipped: 0
result: PASS
```

这与此前记录的 22/22 基线一致。没有删除、跳过、放宽或替换任何测试。

## 4. Git 清洁度

以下内容会被忽略：

- `datasets/*` except `datasets/README.md`
- `artifacts/*` except `artifacts/README.md`
- checkpoint/model formats: `.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`
- `checkpoints/`, `hf_cache/`, `.cache/`, `huggingface/`,
  `transformers_cache/`
- Python caches, pytest/mypy/ruff/hypothesis/tox/nox caches
- `.venv/`, `venv/`, `env/`
- `.env` files and common shell-history files
- logs, outputs, temporary directories/files, and IDE caches
- `docs/论文/`, which is a local reference corpus rather than baseline source

暂存前清单中没有音频、NumPy array、checkpoint 或模型权重文件。`docs/论文/` 下有三份超过 1 MB 的本地参考 PDF（最大约 4.83 MB）；整个目录均被忽略，不会纳入 commit。

基线中不包含数据集、生成的 artifact、checkpoint、HF cache、模型二进制文件、临时测试输出或 IDE cache。

## 5. 敏感文件检查

文件名和文本模式审计结果：

```text
Hugging Face token candidates: 0
API/access/secret/password assignment candidates: 0
Windows user-profile path candidates: 0
.env / credential / secret / history files: 0
```

扫描仅报告匹配文件名，没有输出机密值。README 中的本地 Legacy 项目路径用于引用项目，并非 Windows 用户配置文件路径或运行时凭据。`.env` 和 `.env.*` 均被忽略；未来可以有意跟踪 `.env.example`。

## 6. 环境快照

用于 import 和离线测试的环境：

| Item | Value |
|---|---|
| OS | Windows 10, build 26200, 64-bit |
| Python | 3.10.20, Anaconda, MSC v.1942 64-bit |
| Python executable | `D:\anaconda3\envs\whisper_v0\python.exe` |
| pip | 26.1.2 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.13.0 |
| NumPy | 1.23.3 |
| PyYAML | 6.0.3 |
| pytest | not installed |
| CUDA available | True |
| CUDA runtime reported by PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA device count | 1 |

这里只记录 CUDA 是否可用；离线单元测试没有运行真实 SSL 模型。部分 Transformers import 路径会产生已有的 SciPy/NumPy 兼容性警告。P0 不调整依赖来处理该问题。

## 7. 文件变更

P0 的功能性变更仅限于：

- `.gitignore`: hardens exclusions for secrets, caches, model binaries,
  temporary files, and the local reference corpus.
- `docs/P0_BASELINE_REPORT.md`: records this verified baseline.

由于仓库此前没有 commit，首个基线 commit 也收录了所有现有项目源码、config、离线测试、Markdown 报告、README 文件和 `pyproject.toml`，但不包含被忽略的本地内容。暂存 diff 检查发现文件末尾有多余空行后，39 个现有文本文件被机械地规范为恰好以一个换行符结尾。P0 未更改任何算法、Backbone API、Mean DR、SCAF、TeacherOutput、embedding dimension、配置值或测试逻辑。

## 8. Git Commit

- Branch: `main`
- Commit message: `chore: establish PAPR-SSL baseline`
- Commit role: initial root commit containing this report and the audited source
- Commit identity: resolve the commit containing this report with
  `git rev-parse HEAD`; the exact hash is reported in the task handoff because a
  commit cannot embed its own final hash without changing that hash.

P0 不包含 push、rebase、reset、依赖安装或历史重写。

## 9. Gate 检查清单

- [PASS] compileall
- [PASS] offline tests — 22/22, 0 skipped
- [PASS] repository hygiene
- [PASS] sensitive file check
- [PASS] environment snapshot
- [PASS] baseline commit — initial `main` commit described above

**P0 GATE: PASS**

## 10. 剩余风险

1. 尚未加载真实 Wav2Vec2、WavLM 或 W2v-BERT 2.0 checkpoint；实际 shape、mask、hidden-state 索引、内存和吞吐量仍未知。
2. Teacher config 仍使用 `revision: main`，这是明确标注为未固定的开发 revision。
3. `pyproject.toml` 记录的是依赖下限建议，不是独立的 PAPR-SSL lock file；环境固定属于 P1 范围。
4. 审计使用的 `whisper_v0` 环境存在 NumPy/SciPy 兼容性警告，且未安装 pytest。Unittest 基线结果仍然有效。
5. 基于 regex 的机密扫描可降低常见风险，但不能替代未来 CI pipeline 中的专用 secret scanner。
6. 被忽略的本地 `docs/论文/` 资料库没有由此 Git 基线备份或版本控制。

## 11. 建议的下一步

P0 完成后暂停。经人工明确确认后，再开始 **P1 T-Integration**：制定隔离环境方案，然后有计划地对三个真实 SSL checkpoint 执行 smoke test；不启动 Teacher 训练、Attention DR、KD 或 Student 开发。
