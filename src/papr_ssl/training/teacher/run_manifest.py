"""P3-08 machine-readable run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

from papr_ssl.training.teacher_training_config import load_teacher_training_config
from papr_ssl.training.teacher.train import checkpoint_sha256


RUN_SCHEMA = "papr_ssl.teacher_run_manifest.v1"
EXPERIMENT_SCHEMA = "papr_ssl.teacher_experiment_manifest.v1"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(data: Mapping[str, Any]) -> str:
    blob = json.dumps(
        dict(data),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)

    def _run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "available": True,
            "commit": _run("rev-parse", "HEAD"),
            "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run("status", "--porcelain")),
        }
    except Exception:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
        }


def _environment(device: str | None = None) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
    }


def _artifact_ref(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return {
        "path": p.as_posix(),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
    }


def _validate_selected_checkpoint(
    selected_path: str | Path,
) -> dict[str, Any]:
    selected_path = Path(selected_path)
    payload = _read_json(selected_path)

    if payload.get("schema") != "papr_ssl.p3_checkpoint_selection.v1":
        raise ValueError("unsupported P3-07 selection schema")

    policy = payload.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("missing selection_policy")
    if policy.get("metric") != "generic_dev_score":
        raise ValueError("P3-08 requires generic_dev_score selection")
    if policy.get("metric_definition") != "prototype_macro_f1":
        raise ValueError(
            "P3-08 requires generic_dev_score=prototype_macro_f1"
        )
    if policy.get("generic_test_accessed") is not False:
        raise RuntimeError("sealed-test violation recorded by P3-07")

    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("missing selected checkpoint")

    checkpoint = Path(str(selected["checkpoint"]))
    expected_sha = str(selected["checkpoint_sha256"])
    actual_sha = checkpoint_sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            "selected checkpoint SHA-256 mismatch: "
            f"expected={expected_sha}, actual={actual_sha}"
        )

    return {
        "selection_file": _artifact_ref(selected_path),
        "policy": dict(policy),
        "selected": dict(selected),
    }


def _validate_cache(
    cache_dir: str | Path,
    *,
    expected_model_id: str,
    expected_revision: str,
    expected_layer: int,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    index_path = cache_dir / "index.jsonl"
    manifest = _read_json(manifest_path)

    if manifest.get("schema") != "papr_ssl.ssl_mean_feature_cache.v1":
        raise ValueError("unsupported feature-cache schema")

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("cache manifest missing identity")

    expected = {
        "model_id": expected_model_id,
        "model_revision": expected_revision,
        "layer": int(expected_layer),
        "pooling": "masked_mean",
        "feature_dtype": "float32",
    }
    if dict(identity) != expected:
        raise RuntimeError(
            "run/cache identity mismatch: "
            f"expected={expected}, actual={dict(identity)}"
        )

    return {
        "directory": cache_dir.as_posix(),
        "identity": dict(identity),
        "hidden_dim": int(manifest["hidden_dim"]),
        "sample_count": int(manifest["sample_count"]),
        "shard_count": int(manifest["shard_count"]),
        "manifest": _artifact_ref(manifest_path),
        "index": _artifact_ref(index_path),
    }


@dataclass(frozen=True)
class SamplerManifest:
    classes_per_batch: int
    samples_per_class: int
    batches_per_epoch: int
    seed: int

    def validate(self) -> None:
        if self.classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be > 0")
        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class must be > 0")
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be > 0")
        if self.seed < 0:
            raise ValueError("sampler seed must be >= 0")


@dataclass(frozen=True)
class OptimizationManifest:
    optimizer: str
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float

    def validate(self) -> None:
        if self.optimizer != "AdamW":
            raise ValueError("P3-08 baseline optimizer must be AdamW")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")


def write_run_manifest(
    *,
    run_dir: str | Path,
    teacher_config_path: str | Path,
    task_view: str,
    task_index_path: str | Path,
    cache_dir: str | Path,
    sampler: SamplerManifest,
    optimization: OptimizationManifest,
    selected_checkpoint_path: str | Path | None = None,
    repo_root: str | Path = ".",
    device: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_path = run_dir / "run_manifest.json"
    if output_path.exists():
        raise FileExistsError(f"run manifest already exists: {output_path}")

    teacher_cfg = load_teacher_training_config(teacher_config_path)
    sampler.validate()
    optimization.validate()

    if sampler.seed != teacher_cfg.seed:
        raise ValueError(
            "sampler seed and Teacher config seed must match for this run"
        )
    if not task_view.strip():
        raise ValueError("task_view must be non-empty")

    metrics_path = run_dir / "metrics.jsonl"
    hashes_path = run_dir / "checkpoint_hashes.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not hashes_path.is_file():
        raise FileNotFoundError(hashes_path)

    cache = _validate_cache(
        cache_dir,
        expected_model_id=teacher_cfg.backbone.model_id,
        expected_revision=teacher_cfg.backbone.model_revision,
        expected_layer=teacher_cfg.backbone.layer,
    )

    if selected_checkpoint_path is None:
        selected_checkpoint_path = run_dir / "selected_checkpoint.json"
    selection = _validate_selected_checkpoint(selected_checkpoint_path)

    task_index_ref = _artifact_ref(task_index_path)
    teacher_config_ref = _artifact_ref(teacher_config_path)

    experiment_identity = {
        "task_view": task_view,
        "task_index_sha256": task_index_ref["sha256"],
        "model_id": teacher_cfg.backbone.model_id,
        "model_revision": teacher_cfg.backbone.model_revision,
        "layer": teacher_cfg.backbone.layer,
        "head": teacher_cfg.head.kind,
        "embedding_dim": teacher_cfg.head.embedding_dim,
        "scaf": {
            "k": teacher_cfg.scaf.k,
            "margin": teacher_cfg.scaf.margin,
            "scale": teacher_cfg.scaf.scale,
        },
        "sampler": {
            "classes_per_batch": sampler.classes_per_batch,
            "samples_per_class": sampler.samples_per_class,
            "batches_per_epoch": sampler.batches_per_epoch,
        },
        "optimization": asdict(optimization),
    }
    experiment_fingerprint = canonical_json_sha256(experiment_identity)

    run_identity = {
        "experiment_fingerprint": experiment_fingerprint,
        "seed": teacher_cfg.seed,
        "selected_checkpoint_sha256": (
            selection["selected"]["checkpoint_sha256"]
        ),
    }
    run_fingerprint = canonical_json_sha256(run_identity)

    manifest = {
        "schema": RUN_SCHEMA,
        "phase": "P3-08",
        "status": "COMPLETE",
        "experiment_fingerprint": experiment_fingerprint,
        "run_fingerprint": run_fingerprint,
        "task": {
            "task_view": task_view,
            "task_index": task_index_ref,
        },
        "teacher_config": {
            "file": teacher_config_ref,
            "seed": teacher_cfg.seed,
            "backbone": asdict(teacher_cfg.backbone),
            "head": asdict(teacher_cfg.head),
            "scaf": asdict(teacher_cfg.scaf),
        },
        "feature_cache": cache,
        "sampler": asdict(sampler),
        "optimization": asdict(optimization),
        "artifacts": {
            "metrics": _artifact_ref(metrics_path),
            "checkpoint_hashes": _artifact_ref(hashes_path),
            "selection": selection,
        },
        "evaluation_policy": {
            "generic_dev_score": "prototype_macro_f1",
            "checkpoint_selection": "maximize_generic_dev_score",
            "generic_test": "sealed_not_accessed",
        },
        "code": _git_state(repo_root),
        "environment": _environment(device=device),
    }

    manifest["manifest_sha256"] = canonical_json_sha256(manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_run_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = _read_json(path)
    if manifest.get("schema") != RUN_SCHEMA:
        raise ValueError("unsupported run manifest schema")

    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        raise ValueError("run manifest missing manifest_sha256")

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = canonical_json_sha256(payload)
    if actual != expected:
        raise RuntimeError(
            "run manifest semantic hash mismatch: "
            f"expected={expected}, actual={actual}"
        )

    policy = manifest.get("evaluation_policy", {})
    if policy.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("run manifest does not preserve sealed test policy")

    return manifest


def write_experiment_manifest(
    *,
    experiment_dir: str | Path,
    seeds: Sequence[int] = (17, 29, 43),
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    output_path = experiment_dir / "experiment_manifest.json"
    if output_path.exists():
        raise FileExistsError(
            f"experiment manifest already exists: {output_path}"
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    run_refs: list[dict[str, Any]] = []
    fingerprints: set[str] = set()

    for seed in seeds:
        path = experiment_dir / f"seed_{int(seed):04d}" / "run_manifest.json"
        run = verify_run_manifest(path)
        if int(run["teacher_config"]["seed"]) != int(seed):
            raise RuntimeError(
                f"seed mismatch for {path}: "
                f"{run['teacher_config']['seed']} != {seed}"
            )
        fingerprints.add(str(run["experiment_fingerprint"]))
        run_refs.append(
            {
                "seed": int(seed),
                "run_manifest": _artifact_ref(path),
                "run_fingerprint": run["run_fingerprint"],
                "selected_checkpoint_sha256": (
                    run["artifacts"]["selection"]["selected"][
                        "checkpoint_sha256"
                    ]
                ),
            }
        )

    if len(fingerprints) != 1:
        raise RuntimeError(
            "seed runs do not share one experiment fingerprint; "
            "non-seed configuration changed"
        )

    manifest = {
        "schema": EXPERIMENT_SCHEMA,
        "phase": "P3-08",
        "status": "COMPLETE",
        "experiment_fingerprint": next(iter(fingerprints)),
        "seeds": [int(x) for x in seeds],
        "run_count": len(run_refs),
        "runs": run_refs,
        "generic_test": "sealed_not_accessed",
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)

    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
