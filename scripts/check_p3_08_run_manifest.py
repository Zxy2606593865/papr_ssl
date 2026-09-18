#!/usr/bin/env python
"""Standalone P3-08 run-manifest smoke check."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from papr_ssl.training.teacher.run_manifest import (
    OptimizationManifest,
    SamplerManifest,
    verify_run_manifest,
    write_run_manifest,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run = root / "seed_0017"
        run.mkdir(parents=True)

        cfg = root / "teacher.yaml"
        cfg.write_text(
            """schema: papr_ssl.teacher_training_config.v1
phase: P3
seed: 17
backbone:
  name: wav2vec2_base
  model_id: facebook/wav2vec2-base
  model_revision: 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
  layer: 12
  frozen: true
  force_eval: true
  use_offline_cache: true
head:
  kind: mean_dr
  embedding_dim: 64
  l2_normalize: true
scaf:
  k: 3
  margin: 0.2
  scale: 30
""",
            encoding="utf-8",
        )

        task = root / "core30.index.jsonl"
        task.write_text(
            '{"dataset":"mdsc","utt_id":"u1","split":"train","task_label":"打开空调"}\n',
            encoding="utf-8",
        )

        cache = root / "cache"
        cache.mkdir()
        (cache / "index.jsonl").write_text(
            '{"dataset":"mdsc","utt_id":"u1"}\n',
            encoding="utf-8",
        )
        (cache / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "papr_ssl.ssl_mean_feature_cache.v1",
                    "identity": {
                        "model_id": "facebook/wav2vec2-base",
                        "model_revision": (
                            "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
                        ),
                        "layer": 12,
                        "pooling": "masked_mean",
                        "feature_dtype": "float32",
                    },
                    "hidden_dim": 768,
                    "sample_count": 1,
                    "shard_count": 1,
                    "index_file": "index.jsonl",
                }
            ),
            encoding="utf-8",
        )

        ckpt = run / "checkpoints" / "epoch_0002.pt"
        ckpt.parent.mkdir()
        ckpt.write_bytes(b"p3-08-smoke-checkpoint")
        ckpt_hash = sha(ckpt)

        row = {
            "epoch": 2,
            "train_loss": 0.12,
            "dev": {
                "generic_dev_score": 0.84,
                "generic_dev_score_name": "prototype_macro_f1",
            },
            "checkpoint": ckpt.as_posix(),
            "checkpoint_sha256": ckpt_hash,
        }
        (run / "metrics.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )
        (run / "checkpoint_hashes.jsonl").write_text(
            json.dumps(
                {
                    "epoch": 2,
                    "checkpoint": ckpt.as_posix(),
                    "sha256": ckpt_hash,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "selected_checkpoint.json").write_text(
            json.dumps(
                {
                    "schema": "papr_ssl.p3_checkpoint_selection.v1",
                    "phase": "P3-07",
                    "run_dir": run.as_posix(),
                    "selection_policy": {
                        "metric": "generic_dev_score",
                        "metric_definition": "prototype_macro_f1",
                        "direction": "maximize",
                        "tie_break": "earliest_epoch",
                        "generic_test_accessed": False,
                    },
                    "selected": {
                        "epoch": 2,
                        "checkpoint": ckpt.as_posix(),
                        "checkpoint_sha256": ckpt_hash,
                        "generic_dev_score": 0.84,
                        "generic_dev_score_name": "prototype_macro_f1",
                        "selection_metric": "generic_dev_score",
                        "tie_break": "earliest_epoch",
                    },
                }
            ),
            encoding="utf-8",
        )

        manifest = write_run_manifest(
            run_dir=run,
            teacher_config_path=cfg,
            task_view="mdsc_core30_exact_phrase",
            task_index_path=task,
            cache_dir=cache,
            sampler=SamplerManifest(
                classes_per_batch=8,
                samples_per_class=4,
                batches_per_epoch=118,
                seed=17,
            ),
            optimization=OptimizationManifest(
                optimizer="AdamW",
                epochs=20,
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip_norm=5.0,
            ),
            repo_root=root,
            device="cpu",
        )
        verify_run_manifest(run / "run_manifest.json")

        print("=" * 100)
        print("PAPR-SSL P3-08 MACHINE-READABLE RUN MANIFEST")
        print("=" * 100)
        print(f"schema:                         {manifest['schema']}")
        print(f"status:                         {manifest['status']}")
        print(
            "experiment fingerprint:         "
            f"{manifest['experiment_fingerprint'][:16]}..."
        )
        print(
            "run fingerprint:                "
            f"{manifest['run_fingerprint'][:16]}..."
        )
        print(
            "teacher seed:                   "
            f"{manifest['teacher_config']['seed']}"
        )
        print(
            "cache revision recorded:        "
            f"{manifest['feature_cache']['identity']['model_revision'][:12]}..."
        )
        print(
            "cache layer recorded:           "
            f"{manifest['feature_cache']['identity']['layer']}"
        )
        print(
            "task-index SHA-256 recorded:     "
            f"{manifest['task']['task_index']['sha256'][:16]}..."
        )
        print(
            "selected checkpoint SHA-256:    "
            f"{manifest['artifacts']['selection']['selected']['checkpoint_sha256'][:16]}..."
        )
        print(
            "metrics SHA-256 recorded:        "
            f"{manifest['artifacts']['metrics']['sha256'][:16]}..."
        )
        print(
            "generic_test policy:             "
            f"{manifest['evaluation_policy']['generic_test']}"
        )
        print("manifest self-hash verified:     YES")
        print("-" * 100)
        print("P3-08 STATUS: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
