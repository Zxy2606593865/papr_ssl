from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.teacher.run_manifest import (
    OptimizationManifest,
    SamplerManifest,
    verify_run_manifest,
    write_experiment_manifest,
    write_run_manifest,
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_run(root: Path, seed: int) -> dict[str, Path]:
    run = root / f"seed_{seed:04d}"
    run.mkdir(parents=True)

    config = root / f"teacher_{seed}.yaml"
    config.write_text(
        f"""schema: papr_ssl.teacher_training_config.v1
phase: P3
seed: {seed}
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

    task_index = root / "task.jsonl"
    if not task_index.exists():
        task_index.write_text(
            '{"dataset":"toy","utt_id":"u1","split":"train","task_label":"A"}\n',
            encoding="utf-8",
        )

    cache_dir = root / "cache"
    cache_dir.mkdir(exist_ok=True)
    _write_json(
        cache_dir / "manifest.json",
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
        },
    )
    (cache_dir / "index.jsonl").write_text(
        '{"dataset":"toy","utt_id":"u1"}\n',
        encoding="utf-8",
    )

    ckpt = run / "checkpoints" / "epoch_0001.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(f"checkpoint-{seed}".encode("utf-8"))
    ckpt_sha = _sha(ckpt)

    metrics_row = {
        "epoch": 1,
        "train_loss": 0.1,
        "dev": {
            "generic_dev_score": 0.8,
            "generic_dev_score_name": "prototype_macro_f1",
        },
        "checkpoint": ckpt.as_posix(),
        "checkpoint_sha256": ckpt_sha,
    }
    (run / "metrics.jsonl").write_text(
        json.dumps(metrics_row) + "\n",
        encoding="utf-8",
    )
    (run / "checkpoint_hashes.jsonl").write_text(
        json.dumps(
            {
                "epoch": 1,
                "checkpoint": ckpt.as_posix(),
                "sha256": ckpt_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _write_json(
        run / "selected_checkpoint.json",
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
                "epoch": 1,
                "checkpoint": ckpt.as_posix(),
                "checkpoint_sha256": ckpt_sha,
                "generic_dev_score": 0.8,
                "generic_dev_score_name": "prototype_macro_f1",
                "selection_metric": "generic_dev_score",
                "tie_break": "earliest_epoch",
            },
        },
    )

    return {
        "run": run,
        "config": config,
        "task_index": task_index,
        "cache": cache_dir,
    }


class RunManifestTest(unittest.TestCase):
    def test_run_manifest_records_full_provenance_and_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _prepare_run(root, 17)
            manifest = write_run_manifest(
                run_dir=paths["run"],
                teacher_config_path=paths["config"],
                task_view="toy_core",
                task_index_path=paths["task_index"],
                cache_dir=paths["cache"],
                sampler=SamplerManifest(2, 2, 5, 17),
                optimization=OptimizationManifest(
                    "AdamW", 20, 1e-3, 1e-4, 5.0
                ),
                repo_root=root,
                device="cpu",
            )
            self.assertEqual(
                manifest["evaluation_policy"]["generic_test"],
                "sealed_not_accessed",
            )
            self.assertEqual(
                manifest["feature_cache"]["identity"]["layer"],
                12,
            )
            verified = verify_run_manifest(
                paths["run"] / "run_manifest.json"
            )
            self.assertEqual(
                verified["run_fingerprint"],
                manifest["run_fingerprint"],
            )

    def test_seed_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _prepare_run(root, 17)
            with self.assertRaises(ValueError):
                write_run_manifest(
                    run_dir=paths["run"],
                    teacher_config_path=paths["config"],
                    task_view="toy_core",
                    task_index_path=paths["task_index"],
                    cache_dir=paths["cache"],
                    sampler=SamplerManifest(2, 2, 5, 29),
                    optimization=OptimizationManifest(
                        "AdamW", 20, 1e-3, 1e-4, 5.0
                    ),
                    repo_root=root,
                )

    def test_cache_identity_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _prepare_run(root, 17)
            cache_manifest = json.loads(
                (paths["cache"] / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            cache_manifest["identity"]["layer"] = 6
            _write_json(paths["cache"] / "manifest.json", cache_manifest)

            with self.assertRaises(RuntimeError):
                write_run_manifest(
                    run_dir=paths["run"],
                    teacher_config_path=paths["config"],
                    task_view="toy_core",
                    task_index_path=paths["task_index"],
                    cache_dir=paths["cache"],
                    sampler=SamplerManifest(2, 2, 5, 17),
                    optimization=OptimizationManifest(
                        "AdamW", 20, 1e-3, 1e-4, 5.0
                    ),
                    repo_root=root,
                )

    def test_selected_checkpoint_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _prepare_run(root, 17)
            ckpt = paths["run"] / "checkpoints" / "epoch_0001.pt"
            ckpt.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                write_run_manifest(
                    run_dir=paths["run"],
                    teacher_config_path=paths["config"],
                    task_view="toy_core",
                    task_index_path=paths["task_index"],
                    cache_dir=paths["cache"],
                    sampler=SamplerManifest(2, 2, 5, 17),
                    optimization=OptimizationManifest(
                        "AdamW", 20, 1e-3, 1e-4, 5.0
                    ),
                    repo_root=root,
                )

    def test_experiment_manifest_requires_same_nonseed_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for seed in (17, 29, 43):
                paths = _prepare_run(root, seed)
                write_run_manifest(
                    run_dir=paths["run"],
                    teacher_config_path=paths["config"],
                    task_view="toy_core",
                    task_index_path=paths["task_index"],
                    cache_dir=paths["cache"],
                    sampler=SamplerManifest(2, 2, 5, seed),
                    optimization=OptimizationManifest(
                        "AdamW", 20, 1e-3, 1e-4, 5.0
                    ),
                    repo_root=root,
                )
            exp = write_experiment_manifest(
                experiment_dir=root,
                seeds=(17, 29, 43),
            )
            self.assertEqual(exp["run_count"], 3)
            self.assertEqual(exp["generic_test"], "sealed_not_accessed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
