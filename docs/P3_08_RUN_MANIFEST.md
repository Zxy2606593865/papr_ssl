# P3-08 — 机器可读的 Run Manifest

## 目标

每个已完成的 P3 seed run 都必须生成一份机器可读的来源记录：

```text
run_manifest.json
```

其中记录精确的 task/data identity、SSL revision/layer/cache、Teacher architecture、seed、sampler、optimization budget、metrics/checkpoint artifacts、选择结果、code revision、environment 和封存测试状态。

## 每个 seed 的 run 目录结构

```text
seed_0017/
├── checkpoints/
├── metrics.jsonl
├── checkpoint_hashes.jsonl
├── selected_checkpoint.json
└── run_manifest.json
```

Manifest 记录路径与 SHA-256 hashes，不复制大型二进制 artifacts。

## Experiment fingerprint

`experiment_fingerprint` 排除 seed 和 selected-checkpoint identity，因此用于标识不含 seed 的实验条件：

```text
task view + task-index hash
backbone model_id + exact revision + layer
Mean DR + 64D + SCAF contract
sampler composition
optimization budget
```

Seed 17 / 29 / 43 必须共用一个 experiment fingerprint。

## Run fingerprint

`run_fingerprint` additionally includes:

```text
seed
selected checkpoint SHA-256
```

因此每个 seed run 都有不同的 run identity。

## 必需的完整性检查

出现以下任一情况时，P3-08 失败：

- sampler seed differs from Teacher config seed;
- feature-cache model/revision/layer differs from Teacher config;
- selected checkpoint SHA-256 no longer matches;
- P3-07 selection did not use `generic_dev_score`;
- metric definition is not `prototype_macro_f1`;
- P3-07 reports any generic-test access;
- a run manifest already exists and would be overwritten.

## 机器可读的来源记录

The manifest includes:

```text
task-view and task-index SHA-256
Teacher config file SHA-256
cache manifest/index SHA-256
metrics.jsonl SHA-256
checkpoint_hashes.jsonl SHA-256
selected_checkpoint.json SHA-256
selected checkpoint SHA-256
Git commit/branch/dirty state when available
Python / PyTorch / CUDA / platform / device
```

## Manifest 自身 Hash

The semantic payload is protected by:

```text
manifest_sha256
```

`verify_run_manifest()` 会重新计算该值，并检测后续修改。

## Multi-seed experiment manifest

所有 seed runs 均生成 `run_manifest.json` 后，P3-08 可以创建：

```text
experiment_manifest.json
```

该文件会验证所有 seed runs 是否共享完全相同且不含 seed 的 `experiment_fingerprint`。

借此可以发现以下意外变更：

```text
seed 17 -> layer 24
seed 29 -> layer 18
seed 43 -> layer 24
```

否则这些 run 就不再构成有效的 multi-seed 重复实验。

## P3-08 Gate

满足以下条件时通过：

1. one immutable `run_manifest.json` per completed seed run;
2. exact Teacher config/cache identity agreement;
3. sampler seed equals run seed;
4. data/config/metrics/selection/checkpoint hashes are recorded;
5. selected checkpoint SHA-256 verifies;
6. generic test remains `sealed_not_accessed`;
7. run-manifest self-hash verifies;
8. multi-seed runs share one experiment fingerprint.
