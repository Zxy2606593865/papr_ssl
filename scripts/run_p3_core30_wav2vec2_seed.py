#!/usr/bin/env python
"""Run one frozen real P3 seed for Core30 + Wav2Vec2-base layer12.

This is not a new P3 phase. It reuses the already frozen P3-01..P3-08
contracts. Only the seed is allowed to change between 17 / 29 / 43.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from papr_ssl.cache.ssl_feature_cache import SSLCacheIdentity
from papr_ssl.training.feature_data import (
    ClassBalancedSamplerConfig,
    OfflineFeatureTaskDataset,
    build_dev_loader,
    build_train_loader,
    feature_cache_collate,
)
from papr_ssl.training.teacher.evaluate import evaluate_embedding_space
from papr_ssl.training.teacher.run_manifest import (
    OptimizationManifest,
    SamplerManifest,
    write_run_manifest,
)
from papr_ssl.training.teacher.select import select_run_checkpoint
from papr_ssl.training.teacher.train import (
    OptimizationConfig,
    build_project_mean_dr,
    build_project_scaf,
    run_training,
    seed_everything,
)
from papr_ssl.training.teacher_training_config import (
    teacher_training_config_from_dict,
)


MODEL_ID = "facebook/wav2vec2-base"
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
LAYER = 12
FROZEN_SEEDS = (17, 29, 43)
TASK_VIEW = "mdsc_core30_exact_phrase"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def materialize_seed_config(
    *,
    base_config_path: Path,
    experiment_dir: Path,
    seed: int,
) -> Path:
    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("base Teacher config must be a YAML mapping")

    data = dict(data)
    data["seed"] = int(seed)
    cfg = teacher_training_config_from_dict(data)

    if cfg.backbone.model_id != MODEL_ID:
        raise RuntimeError("Teacher config model_id mismatch")
    if cfg.backbone.model_revision != REVISION:
        raise RuntimeError("Teacher config revision mismatch")
    if cfg.backbone.layer != LAYER:
        raise RuntimeError("Teacher config layer mismatch")

    config_path = (
        experiment_dir
        / "configs"
        / f"teacher_seed_{seed:04d}.yaml"
    )
    rendered = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    )

    if config_path.exists():
        existing = config_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise RuntimeError(
                f"existing seed config differs from frozen config: {config_path}"
            )
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(rendered, encoding="utf-8")

    return config_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=FROZEN_SEEDS)
    p.add_argument(
        "--task-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--base-teacher-config",
        type=Path,
        default=Path("configs/p3/teacher/wav2vec2_base.yaml"),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/ssl_feature_cache"),
    )
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(
            "artifacts/p3_runs/"
            "mdsc_core30__wav2vec2_base__layer12"
        ),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--classes-per-batch", type=int, default=8)
    p.add_argument("--samples-per-class", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=128)
    args = p.parse_args()

    seed = int(args.seed)
    run_dir = args.experiment_dir / f"seed_{seed:04d}"

    # Immutable run policy.
    immutable_markers = (
        run_dir / "metrics.jsonl",
        run_dir / "checkpoint_hashes.jsonl",
        run_dir / "selected_checkpoint.json",
        run_dir / "run_manifest.json",
        run_dir / "untrained_projection_baseline.json",
    )
    if any(path.exists() for path in immutable_markers):
        raise FileExistsError(
            f"seed {seed} already has run artifacts in {run_dir}; "
            "do not silently overwrite an existing real run"
        )

    effective_config_path = materialize_seed_config(
        base_config_path=args.base_teacher_config,
        experiment_dir=args.experiment_dir,
        seed=seed,
    )

    identity = SSLCacheIdentity(
        model_id=MODEL_ID,
        model_revision=REVISION,
        layer=LAYER,
    )
    cache_dir = args.cache_root / identity.slug
    if not (cache_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            "real SSL cache is missing; materialize it first:\n"
            f"  {cache_dir}"
        )

    train_ds = OfflineFeatureTaskDataset(
        task_index=args.task_index,
        cache_dir=cache_dir,
        cache_identity=identity,
        split="train",
    )
    dev_ds = OfflineFeatureTaskDataset(
        task_index=args.task_index,
        cache_dir=cache_dir,
        cache_identity=identity,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )

    if len(train_ds) != 3756:
        raise RuntimeError(f"expected 3756 train rows, got {len(train_ds)}")
    if len(dev_ds) != 442:
        raise RuntimeError(f"expected 442 dev rows, got {len(dev_ds)}")
    if len(train_ds.class_to_index) != 30:
        raise RuntimeError(
            f"Core30 expected 30 classes, got {len(train_ds.class_to_index)}"
        )

    sampler_cfg = ClassBalancedSamplerConfig(
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        batches_per_epoch=None,
        seed=seed,
    )
    train_loader, sampler = build_train_loader(
        train_ds,
        sampler_config=sampler_cfg,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    train_ref_loader = DataLoader(
        train_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=feature_cache_collate,
    )
    dev_loader = build_dev_loader(
        dev_ds,
        batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    hidden_dim = int(train_ds.reader.hidden_dim)
    if hidden_dim != 768:
        raise RuntimeError(f"expected Wav2Vec2 cache dim=768, got {hidden_dim}")

    device = torch.device(args.device)

    # Frozen P3-06 ordering: seed BEFORE model/SCAF/sampler-dependent training.
    seed_everything(seed)
    mean_dr = build_project_mean_dr(
        input_dim=hidden_dim,
        embedding_dim=64,
    )
    scaf = build_project_scaf(
        num_classes=30,
        embedding_dim=64,
        k=3,
        margin=0.2,
        scale=30.0,
    )
    mean_dr.to(device)
    scaf.to(device)

    baseline = evaluate_embedding_space(
        mean_dr=mean_dr,
        train_reference_loader=train_ref_loader,
        dev_loader=dev_loader,
        num_classes=30,
        device=device,
    )
    baseline_payload = {
        "schema": "papr_ssl.untrained_projection_baseline.v1",
        "phase": "P3-scientific-gate",
        "task_view": TASK_VIEW,
        "backbone": {
            "model_id": MODEL_ID,
            "model_revision": REVISION,
            "layer": LAYER,
        },
        "seed": seed,
        "head": "mean_dr_random_untrained",
        "embedding_dim": 64,
        "dev": baseline.to_dict(),
        "generic_test_accessed": False,
    }
    write_json(
        run_dir / "untrained_projection_baseline.json",
        baseline_payload,
    )

    optimization = OptimizationConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )

    history = run_training(
        mean_dr=mean_dr,
        scaf=scaf,
        train_loader=train_loader,
        train_reference_loader=train_ref_loader,
        dev_loader=dev_loader,
        num_classes=30,
        seed=seed,
        run_dir=run_dir,
        device=device,
        optimization=optimization,
    )

    selection = select_run_checkpoint(run_dir)
    selected = selection["selected"]
    trained_score = float(selected["generic_dev_score"])
    baseline_score = float(baseline.generic_dev_score)
    delta = trained_score - baseline_score

    comparison = {
        "schema": "papr_ssl.p3_real_baseline_comparison.v1",
        "task_view": TASK_VIEW,
        "seed": seed,
        "baseline_generic_dev_score": baseline_score,
        "trained_selected_generic_dev_score": trained_score,
        "absolute_delta": delta,
        "relative_note": (
            "single-seed diagnostic; scientific conclusion uses "
            "the frozen 17/29/43 repetition"
        ),
        "generic_test_accessed": False,
    }
    write_json(run_dir / "baseline_vs_trained.json", comparison)

    batches_per_epoch = len(sampler)
    write_run_manifest(
        run_dir=run_dir,
        teacher_config_path=effective_config_path,
        task_view=TASK_VIEW,
        task_index_path=args.task_index,
        cache_dir=cache_dir,
        sampler=SamplerManifest(
            classes_per_batch=args.classes_per_batch,
            samples_per_class=args.samples_per_class,
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        ),
        optimization=OptimizationManifest(
            optimizer="AdamW",
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
        ),
        repo_root=Path("."),
        device=str(device),
    )

    last = history[-1]
    print("=" * 108)
    print(
        "PAPR-SSL REAL P3: MDSC-Core30 + "
        f"Wav2Vec2-base layer12 + seed{seed}"
    )
    print("=" * 108)
    print(f"train samples:                 {len(train_ds)}")
    print(f"dev samples:                   {len(dev_ds)}")
    print(f"classes:                       {len(train_ds.class_to_index)}")
    print(f"cached input dim:              {hidden_dim}")
    print(f"train batches/epoch:           {batches_per_epoch}")
    print(f"epochs:                        {args.epochs}")
    print(f"untrained projection dev:      {baseline_score:.6f}")
    print(f"selected trained dev:          {trained_score:.6f}")
    print(f"absolute delta:                {delta:+.6f}")
    print(f"selected epoch:                {selected['epoch']}")
    print(
        "final-epoch dev:               "
        f"{float(last['dev']['generic_dev_score']):.6f}"
    )
    print(
        "selected checkpoint SHA-256:   "
        f"{selected['checkpoint_sha256'][:16]}..."
    )
    print(f"baseline collapse:              {baseline.collapsed}")
    print("generic_test accessed:          NO")
    print("-" * 108)
    print(f"REAL P3 SEED {seed}: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
