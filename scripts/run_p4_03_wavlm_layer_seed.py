#!/usr/bin/env python
"""P4-03: run one WavLM hidden_state_index x seed condition."""

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
from papr_ssl.training.teacher.p4_candidates import get_p4_backbone
from papr_ssl.training.teacher.p4_run_manifest import write_p4_run_manifest
from papr_ssl.training.teacher.run_manifest import (
    OptimizationManifest,
    SamplerManifest,
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


FROZEN_SEEDS = (17, 29, 43)
TASK_VIEW = "mdsc_core30_exact_phrase"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def materialize_effective_config(
    *,
    base_config_path: Path,
    run_group_dir: Path,
    layer: int,
    seed: int,
) -> Path:
    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("base Teacher config must be a YAML mapping")

    data = dict(data)
    data["seed"] = seed
    data["backbone"] = dict(data["backbone"])
    data["backbone"]["layer"] = layer

    cfg = teacher_training_config_from_dict(data)
    spec = get_p4_backbone("wavlm_large")
    if cfg.backbone.model_id != spec.model_id:
        raise RuntimeError("model_id changed from P4-01")
    if cfg.backbone.model_revision != spec.model_revision:
        raise RuntimeError("model revision changed from P4-01")
    if cfg.backbone.layer != layer:
        raise RuntimeError("effective layer mismatch")

    path = (
        run_group_dir
        / "configs"
        / f"teacher_layer_{layer:02d}_seed_{seed:04d}.yaml"
    )
    rendered = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"frozen effective config changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
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
        default=Path("configs/p3/teacher/wavlm_large.yaml"),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/ssl_feature_cache"),
    )
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("artifacts/p4_runs/wavlm_large"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    spec = get_p4_backbone("wavlm_large")
    if args.layer not in spec.valid_hidden_state_indices:
        raise ValueError(
            f"--layer must be one of {spec.valid_hidden_state_indices}"
        )

    layer = int(args.layer)
    seed = int(args.seed)
    run_group_dir = args.sweep_root / f"layer_{layer:02d}"
    run_dir = run_group_dir / f"seed_{seed:04d}"

    immutable = (
        run_dir / "metrics.jsonl",
        run_dir / "selected_checkpoint.json",
        run_dir / "run_manifest.json",
    )
    if any(x.exists() for x in immutable):
        raise FileExistsError(
            f"P4 run already exists; refusing overwrite: {run_dir}"
        )

    config_path = materialize_effective_config(
        base_config_path=args.base_teacher_config,
        run_group_dir=run_group_dir,
        layer=layer,
        seed=seed,
    )

    identity = SSLCacheIdentity(
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        layer=layer,
    )
    cache_dir = args.cache_root / identity.slug
    if not (cache_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"missing P4-03 cache for hidden_states[{layer}]: {cache_dir}"
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
    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError("Core30 train/dev count changed")
    if len(train_ds.class_to_index) != 30:
        raise RuntimeError("Core30 class count changed")

    sampler_cfg = ClassBalancedSamplerConfig(
        classes_per_batch=8,
        samples_per_class=4,
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
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=feature_cache_collate,
    )
    dev_loader = build_dev_loader(
        dev_ds,
        batch_size=128,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    hidden_dim = int(train_ds.reader.hidden_dim)
    if hidden_dim != spec.embedding_dim:
        raise RuntimeError(
            f"hidden dim mismatch: {hidden_dim} != {spec.embedding_dim}"
        )

    device = torch.device(args.device)
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
    write_json(
        run_dir / "untrained_projection_baseline.json",
        {
            "schema": "papr_ssl.untrained_projection_baseline.v1",
            "phase": "P4-03",
            "task_view": TASK_VIEW,
            "backbone": {
                "model_id": spec.model_id,
                "model_revision": spec.model_revision,
                "hidden_state_index": layer,
            },
            "seed": seed,
            "dev": baseline.to_dict(),
            "generic_test_accessed": False,
        },
    )

    optimization = OptimizationConfig(
        epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=5.0,
    )
    run_training(
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

    write_json(
        run_dir / "p4_result.json",
        {
            "schema": "papr_ssl.p4_03_wavlm_run_result.v1",
            "phase": "P4-03",
            "backbone": "wavlm_large",
            "hidden_state_index": layer,
            "seed": seed,
            "selection_metric": "generic_dev_score",
            "selection_metric_definition": "prototype_macro_f1",
            "selected_epoch": int(selected["epoch"]),
            "selected_generic_dev_score": float(
                selected["generic_dev_score"]
            ),
            "untrained_projection_generic_dev_score": float(
                baseline.generic_dev_score
            ),
            "selected_checkpoint_sha256": selected[
                "checkpoint_sha256"
            ],
            "generic_test_accessed": False,
        },
    )

    write_p4_run_manifest(
        run_dir=run_dir,
        teacher_config_path=config_path,
        task_view=TASK_VIEW,
        task_index_path=args.task_index,
        cache_dir=cache_dir,
        sampler=SamplerManifest(
            classes_per_batch=8,
            samples_per_class=4,
            batches_per_epoch=len(sampler),
            seed=seed,
        ),
        optimization=OptimizationManifest(
            optimizer="AdamW",
            epochs=20,
            learning_rate=1e-3,
            weight_decay=1e-4,
            grad_clip_norm=5.0,
        ),
        repo_root=Path("."),
        device=str(device),
    )

    print("=" * 104)
    print(f"P4-03 WAVLM hidden_states[{layer}] seed={seed}")
    print("=" * 104)
    print(f"train/dev:                   {len(train_ds)} / {len(dev_ds)}")
    print(f"cached input dim:            {hidden_dim}")
    print(f"untrained projection dev:    {baseline.generic_dev_score:.6f}")
    print(
        "selected generic_dev_score: "
        f"{float(selected['generic_dev_score']):.6f}"
    )
    print(f"selected epoch:              {selected['epoch']}")
    print(
        "selected checkpoint SHA:    "
        f"{selected['checkpoint_sha256'][:16]}..."
    )
    print("generic_test accessed:       NO")
    print("-" * 104)
    print("P4-03 SINGLE RUN: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
