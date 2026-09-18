#!/usr/bin/env python
"""First real P3 Teacher run: Core30 + Wav2Vec2 layer12 + seed17."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    load_teacher_training_config,
)


MODEL_ID = "facebook/wav2vec2-base"
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
LAYER = 12
SEED = 17


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--teacher-config",
        type=Path,
        default=Path("configs/p3/teacher/wav2vec2_base.yaml"),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/ssl_feature_cache"),
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "artifacts/p3_runs/"
            "mdsc_core30__wav2vec2_base__layer12/seed_0017"
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

    cfg = load_teacher_training_config(args.teacher_config)
    if cfg.seed != SEED:
        raise RuntimeError(
            f"first real run requires seed={SEED}, config has {cfg.seed}"
        )
    if cfg.backbone.model_id != MODEL_ID:
        raise RuntimeError("teacher config model_id mismatch")
    if cfg.backbone.model_revision != REVISION:
        raise RuntimeError("teacher config revision mismatch")
    if cfg.backbone.layer != LAYER:
        raise RuntimeError("teacher config layer mismatch")

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
    if len(train_ds.class_to_index) != 30:
        raise RuntimeError(
            f"Core30 expected 30 classes, got {len(train_ds.class_to_index)}"
        )

    sampler_cfg = ClassBalancedSamplerConfig(
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        batches_per_epoch=None,
        seed=SEED,
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
    device = torch.device(args.device)

    # IMPORTANT: seed before model/SCAF initialization.
    seed_everything(SEED)
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

    # Scientific baseline: same randomly initialized projection, no training.
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
        "task_view": "mdsc_core30_exact_phrase",
        "backbone": {
            "model_id": MODEL_ID,
            "model_revision": REVISION,
            "layer": LAYER,
        },
        "seed": SEED,
        "head": "mean_dr_random_untrained",
        "embedding_dim": 64,
        "dev": baseline.to_dict(),
        "generic_test_accessed": False,
    }
    baseline_path = args.run_dir / "untrained_projection_baseline.json"
    if baseline_path.exists():
        raise FileExistsError(baseline_path)
    write_json(baseline_path, baseline_payload)

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
        seed=SEED,
        run_dir=args.run_dir,
        device=device,
        optimization=optimization,
    )

    selection = select_run_checkpoint(args.run_dir)
    selected = selection["selected"]
    trained_score = float(selected["generic_dev_score"])
    baseline_score = float(baseline.generic_dev_score)
    delta = trained_score - baseline_score

    comparison = {
        "schema": "papr_ssl.p3_real_baseline_comparison.v1",
        "task_view": "mdsc_core30_exact_phrase",
        "seed": SEED,
        "baseline_generic_dev_score": baseline_score,
        "trained_selected_generic_dev_score": trained_score,
        "absolute_delta": delta,
        "relative_note": (
            "single-seed diagnostic only; P3 scientific gate requires "
            "multi-seed stability before declaring superiority"
        ),
        "generic_test_accessed": False,
    }
    write_json(
        args.run_dir / "baseline_vs_trained.json",
        comparison,
    )

    batches_per_epoch = len(sampler)
    write_run_manifest(
        run_dir=args.run_dir,
        teacher_config_path=args.teacher_config,
        task_view="mdsc_core30_exact_phrase",
        task_index_path=args.task_index,
        cache_dir=cache_dir,
        sampler=SamplerManifest(
            classes_per_batch=args.classes_per_batch,
            samples_per_class=args.samples_per_class,
            batches_per_epoch=batches_per_epoch,
            seed=SEED,
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

    print("=" * 104)
    print("PAPR-SSL REAL P3 RUN 01: MDSC-Core30 + Wav2Vec2-base layer12 + seed17")
    print("=" * 104)
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
        "selected checkpoint SHA-256: "
        f"{selected['checkpoint_sha256'][:16]}..."
    )
    print(f"embedding collapse at baseline:{baseline.collapsed}")
    print("generic_test accessed:         NO")
    print("-" * 104)
    print("REAL P3 RUN 01: COMPLETE")
    print(
        "NOTE: do not declare the P3 scientific gate from seed17 alone; "
        "repeat seeds 29 and 43 next."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
