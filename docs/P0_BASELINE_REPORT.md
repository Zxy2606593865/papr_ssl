# PAPR-SSL P0 Baseline Report

Date: 2026-09-09 (Asia/Shanghai)

P0 freezes the existing Teacher-first framework. It does not download or run a
real SSL checkpoint, start P1, modify model mathematics, or add a new feature.

## 1. Repository Status

- Repository: initialized Git worktree
- Branch: `main`
- History before P0: no commits (`HEAD` did not exist)
- Tracked files before P0: none
- Baseline role: first formal project commit
- Legacy `whisper/papr` project: not modified and no Git operation performed
- Python package import: PASS (`papr_ssl 0.1.0` and public Teacher APIs)

Before staging, the complete project appeared as untracked top-level paths. Git
ignored `.learnings/` and Python caches. The baseline staging audit excludes
local data, artifacts, model weights, HF caches, temporary files, and the local
`docs/论文/` reference corpus.

## 2. Compile Check

Command:

```text
python -m compileall src tests scripts
```

Result: **PASS**. `src/`, `tests/`, and the existing `scripts/` directory were
processed without syntax errors or warnings.

## 3. Test Results

The project README defines standard-library unittest discovery, and pytest is
not installed in the audited environment. Tests were executed without network
or model loading by adding the local `src/` directory to the process import
path.

```text
tests collected: 22
passed: 22
failed: 0
errors: 0
skipped: 0
result: PASS
```

This matches the previously documented 22/22 baseline. No test was removed,
skipped, relaxed, or replaced.

## 4. Git Hygiene

The following are ignored:

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

Pre-staging inventory found no audio, NumPy array, checkpoint, or model-weight
payload. Three local reference PDFs exceed 1 MB (maximum approximately 4.83 MB)
under `docs/论文/`; the entire directory is ignored and excluded from the commit.

No dataset, generated artifact, checkpoint, HF cache, model binary, temporary
test output, or IDE cache is included in the baseline.

## 5. Sensitive File Check

Filename and text-pattern audits found:

```text
Hugging Face token candidates: 0
API/access/secret/password assignment candidates: 0
Windows user-profile path candidates: 0
.env / credential / secret / history files: 0
```

Scans were filename-only for matches and did not print secret values. The local
Legacy project path in README is a project-reference path, not a Windows user
profile or runtime credential. `.env` and `.env.*` are ignored, while an
intentional `.env.example` may be tracked in the future.

## 6. Environment Snapshot

Environment used for imports and offline tests:

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

CUDA availability is recorded only; offline unit tests did not run real SSL
models. Importing the environment emits an existing SciPy/NumPy compatibility
warning in some Transformers import paths. P0 does not alter dependencies to
address it.

## 7. Files Changed

P0's functional changes are limited to:

- `.gitignore`: hardens exclusions for secrets, caches, model binaries,
  temporary files, and the local reference corpus.
- `docs/P0_BASELINE_REPORT.md`: records this verified baseline.

Because the repository had no previous commit, the first baseline commit also
captures all existing project source, configs, offline tests, Markdown reports,
README files, and `pyproject.toml`. It does not capture ignored local content.
After the staged diff check identified extra blank lines at EOF, 39 existing
text files were mechanically normalized to end with exactly one newline.
No algorithm, Backbone API, Mean DR, SCAF, TeacherOutput, embedding dimension,
configuration value, or test logic changed during P0.

## 8. Git Commit

- Branch: `main`
- Commit message: `chore: establish PAPR-SSL baseline`
- Commit role: initial root commit containing this report and the audited source
- Commit identity: resolve the commit containing this report with
  `git rev-parse HEAD`; the exact hash is reported in the task handoff because a
  commit cannot embed its own final hash without changing that hash.

No push, rebase, reset, dependency installation, or history rewrite is part of
P0.

## 9. Gate Checklist

- [PASS] compileall
- [PASS] offline tests — 22/22, 0 skipped
- [PASS] repository hygiene
- [PASS] sensitive file check
- [PASS] environment snapshot
- [PASS] baseline commit — initial `main` commit described above

**P0 GATE: PASS**

## 10. Remaining Risks

1. No real Wav2Vec2, WavLM, or W2v-BERT 2.0 checkpoint has been loaded; real
   shapes, masks, hidden-state indexing, memory, and throughput remain unknown.
2. Teacher configs still use `revision: main` as explicitly marked unpinned
   development revisions.
3. `pyproject.toml` contains dependency lower-bound proposals, not an isolated
   PAPR-SSL lock file; environment pinning belongs to P1.
4. The audited `whisper_v0` environment has an existing NumPy/SciPy compatibility
   warning and no pytest installation. Unittest baseline results remain valid.
5. Regex-based secret scanning reduces common risks but is not a substitute for
   a dedicated secret scanner in a future CI pipeline.
6. The ignored local `docs/论文/` corpus is not backed up or versioned by this Git
   baseline.

## 11. Recommended Next Step

Stop after P0. After explicit human confirmation, begin **P1 T-Integration**:
create an isolated environment plan, then deliberately smoke-test the three real
SSL checkpoints without starting Teacher training, Attention DR, KD, or Student
development.
