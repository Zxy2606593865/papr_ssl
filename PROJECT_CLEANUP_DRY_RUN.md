# PAPR-SSL Conservative Repository Cleanup — Dry Run

Date: 2026-09-12 (Asia/Shanghai)

Status: **DRY RUN COMPLETE — NO FILES DELETED, MOVED, OR ARCHIVED**

## 1. Scope and Safety Boundary

This audit follows `AUDIT -> CLASSIFY -> DRY RUN`. Cleanup execution is intentionally deferred. The repository contains a large body of uncommitted research work after the single baseline commit, so an untracked file is never treated as disposable evidence.

Protected by default:

- all `datasets/` content, raw audio, ZIP archives, manifests, and metadata;
- all `artifacts/`, checkpoints, model weights, caches, evaluation JSON, repeated-split outputs, and audit evidence;
- the 64D, 128D, 256D, and 512D Teacher experiment lines;
- the Project-1 256D mainline and Project-2 64D future route;
- configs, package source, tests, environment metadata, Git, and documentation;
- Unknown Exposure v1-v5 history, including v5 as the current valid audit;
- anything without affirmative evidence of redundancy.

`generic_test` remains sealed. The final inventory treats all dataset and artifact payloads as metadata-only protected assets: it does not use their contents or hashes for cleanup classification. No model, checkpoint, audio decoder, embedding pipeline, or evaluation entry point was run.

## 2. Inventory Deliverables

- `PROJECT_CLEANUP_INVENTORY.csv`: 137,227 pre-cleanup rows with the required fields `relative_path`, `file_type`, `size`, `modified_time`, `git_tracked`, `git_status`, `python_imported_by`, `referenced_by_scripts`, `referenced_by_docs`, `referenced_by_manifests`, `artifact_role`, and `candidate_status`. It also includes `classification_evidence` and `duplicate_group`.
- `PROJECT_CLEANUP_DUPLICATES.csv`: SHA-256 duplicate groups for non-protected code/document/config/support files. `datasets/` and `artifacts/` are deliberately marked `not_hashed_protected_asset`.

The generated CSV files and this report are audit outputs and are not recursively included in the pre-cleanup inventory.

## 3. Current Directory Statistics

Inventory scope excludes `.git/` internals and includes metadata for ignored files.

| Top-level entry | Files | Bytes | MiB |
|---|---:|---:|---:|
| `artifacts/` | 12,166 | 11,897,965,372 | 11,346.78 |
| `datasets/` | 124,525 | 6,857,688,536 | 6,540.00 |
| `docs/` | 75 | 12,645,769 | 12.06 |
| `scripts/` | 125 | 1,156,735 | 1.10 |
| `src/` | 147 | 723,612 | 0.69 |
| `tests/` | 164 | 598,061 | 0.57 |
| `.learnings/` | 3 | 11,320 | 0.01 |
| `papr_ssl/` | 2 | 7,702 | 0.01 |
| `configs/` | 11 | 6,922 | 0.01 |
| root/support files | 9 | 5,391 | <0.01 |
| **Total** | **137,227** | **18,770,809,420** | **17,901.24** |

Git state at audit time:

- branch: `main`;
- history: one commit, `cfac59d chore: establish PAPR-SSL baseline`;
- tracked files: 52;
- tracked files modified after baseline: 6;
- untracked, non-ignored inventory files: 257;
- ignored inventory files: 136,918;
- no cleanup commit or history rewrite was performed.

## 4. Classification Summary

| Candidate status | Files | Bytes | Decision |
|---|---:|---:|---|
| `KEEP` | 137,003 | 18,761,075,285 | Retain |
| `DELETE_SAFE` | 205 | 1,178,363 | Eligible only after approval/execution gate |
| `ARCHIVE` | 0 | 0 | No safe archive move established |
| `REVIEW_REQUIRED` | 19 | 8,555,772 | Retain pending human review |

Three empty directories are separately eligible for removal and contain zero bytes:

- `artifacts/p6_unknown_exposure_audit/`
- `artifacts/p6_unknown_exposure_audit_v2/`
- `artifacts/p6_unknown_exposure_audit_v3/`

## 5. KEEP Evidence

The exhaustive per-file KEEP classification is in `PROJECT_CLEANUP_INVENTORY.csv`. Major protected totals are:

- `datasets/`: 124,525 files, 6,857,688,536 bytes;
- retained `artifacts/`: 12,148 files, 11,889,409,735 bytes;
- all configs, source, tests, experiment scripts, documents, environment files, and Git metadata;
- `artifacts/p5_frame_cache/`, `artifacts/p6_teacher_dim_sweep/`, `artifacts/p6_teacher_256_15shot/`, and `artifacts/p6_temporal_dtw/`;
- all formal 64D/128D/256D/512D results and Project-2 inputs;
- both MDSC dataset ZIP files, regardless of size or extraction state.

Unknown Exposure v1-v4 are retained, not deleted or moved:

- v1 is referenced by `tests/training/test_p6_unknown_exposure_audit_static.py` and `docs/P6_UNKNOWN_EXPOSURE_DATA_AUDIT.md`;
- v2-v4 are referenced by their versioned audit/review scripts and historical fix documents;
- v4 contains approximately 47.33 MiB of unique historical output;
- moving these scripts would invalidate documented reproduction commands and path-based tests.

## 6. DELETE_SAFE Files and Exact Reasons

Every `DELETE_SAFE` row and its exact path is recorded in `PROJECT_CLEANUP_INVENTORY.csv`; filtering `candidate_status == DELETE_SAFE` yields the complete 205-file list. No other file has this status.

| Exact path set | Files | Bytes | Evidence applying to every file in the set |
|---|---:|---:|---|
| `.pytest_cache/**` | 5 | 933 | Generated pytest bookkeeping; ignored by Git; not a model, result, manifest, protocol, or source; pytest recreates it. |
| `scripts/**/__pycache__/*.pyc` | 16 | 276,124 | Generated CPython bytecode; corresponding source is retained; ignored by Git; Python recreates it. |
| `src/**/__pycache__/*.pyc` | 92 | 481,140 | Generated CPython bytecode; corresponding source is retained; ignored by Git; Python recreates it. |
| `tests/**/__pycache__/*.pyc` | 92 | 420,166 | Generated test bytecode; corresponding tests are retained; ignored by Git; Python recreates it. |
| **Total** | **205** | **1,178,363** | No experiment payload or unique information is included. |

The inventory provides the individual reason on each row rather than relying on filename appearance. No `.log`, backup, temporary export, old-copy Python file, or disposable ZIP was found outside these generated-cache sets.

## 7. ARCHIVE Candidates

None.

No file met a safe archive threshold. In particular, Unknown Exposure v1-v4 retain direct documentation/test references and historical audit value. Moving them would require path edits and could reduce reproducibility, so they remain `KEEP`.

## 8. REVIEW_REQUIRED Files

| Group | Files | Bytes | Why review is required |
|---|---:|---:|---|
| `.vscode/settings.json` | 1 | 135 | Editor configuration is not an editor cache; ownership preference is required. |
| `artifacts/integration/wav2vec2_smoke.json` | 1 | 2,791 | Smoke-labelled integration evidence may document environment compatibility. |
| `artifacts/p2_05/audio_pipeline_smoke.json` | 1 | 2,131 | Smoke-labelled data-pipeline evidence may support protocol history. |
| `artifacts/p6_attentive_stats_ablation_smoke/` | 6 | 4,391,759 | Contains checkpoints/results; protected even though a full run exists. |
| `artifacts/p6_teacher_dim_sweep_smoke/` | 4 | 815,473 | Contains a 64D checkpoint and manifests; Project-2 relevance prevents deletion. |
| `artifacts/p6_wavlm_multilayer_fusion_smoke/` | 6 | 3,343,483 | Contains checkpoints/results for a published negative ablation. |
| **Total** | **19** | **8,555,772** | Retain pending explicit human decision. |

## 9. Duplicate Findings

The protected dataset/artifact payloads were not hashed. Among 45 same-size non-protected candidates, SHA-256 found three duplicate groups:

1. Two copies of `Enhancing Few-shot Keyword Spotting Performance.pdf` under `docs/论文/` (554,860 bytes each). Both stay `KEEP` because the reference corpus is protected and Git/history/reference intent is insufficient to select a canonical copy.
2. `papr_ssl/training/teacher/p5_frame_data.py` and `src/papr_ssl/training/teacher/p5_frame_data.py` are byte-identical (5,735 bytes). Both stay `KEEP`: imports use the package module and a static regression test explicitly references the root-path copy.
3. `papr_ssl/training/teacher/p5_dr.py` and `src/papr_ssl/training/teacher/p5_dr.py` are byte-identical (1,967 bytes). Both stay `KEEP` for the same path-based test/reproducibility reason.

The precise SHA-256 values and paths are in `PROJECT_CLEANUP_DUPLICATES.csv`.

## 10. Estimated Space Recovery

- approved `DELETE_SAFE` files: 1,178,363 bytes (approximately 1.124 MiB);
- three empty directories: 0 bytes;
- `REVIEW_REQUIRED` maximum: 8,555,772 bytes (approximately 8.159 MiB), **not approved for deletion**;
- protected datasets, caches, checkpoints, formal results, and ZIPs: 0 bytes proposed for removal.

The low reclaimable total is expected: almost all disk usage is scientifically meaningful data or reproducibility material.

## 11. Risk Assessment and Dry-Run Decision

| Risk | Level | Control |
|---|---|---|
| Loss of uncommitted research work | High | No untracked source/doc/script is deletable; no Git cleanup/reset used. |
| Breaking experiment paths | High | No scripts moved; root-path duplicates and legacy audit scripts retained. |
| Removing formal artifacts/checkpoints | Critical | All artifact payloads default KEEP or REVIEW_REQUIRED; none DELETE_SAFE. |
| Touching sealed `generic_test` | Critical | No regression/evaluation run; final classification uses metadata-only protection for datasets/artifacts. |
| Cache cleanup changing behavior | Low | Only ignored Python/pytest caches qualify; retained source recreates them. |
| Cosmetic reorganization causing path churn | Medium | `scripts/` remains in place because moving 125 scripts would require broad path/document changes. |

**Dry-run gate:** the 205 cache files and three empty directories have sufficient evidence for a future safe-cleanup step. This report does not authorize the 19 review items, any protected artifact/data, any legacy audit script, or any source/document/config change.

No deletion or archive move was executed in this run.

## 12. Non-Data Regression Verification

- source syntax compilation: **PASS**, 231 Python files compiled in memory with bytecode writes disabled;
- baseline offline unit tests: **PASS**, 22 run, 22 passed, 0 failures, 0 errors, 0 skipped;
- integration/real-model tests: not collected;
- dataset, audio, embedding, open-set, closed-set, and `generic_test` evaluation entry points: not run;
- no dependency installation or environment mutation was performed.
