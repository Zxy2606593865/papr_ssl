#!/usr/bin/env python
"""Train one P5 Mean-DR or Attention-DR condition.

Frozen:
- WavLM-large hidden_states[15] frame cache
- Core30 train/dev
- 64D L2 embedding
- SCAF: K=3, m=0.2 rad, s=30
- 20 epochs
- AdamW lr=1e-3, weight_decay=1e-4
- grad clip=5
- 8-way x 4-shot
- dev-only checkpoint selection by prototype Macro-F1
- seeds 17,29,43
- generic_test sealed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from papr_ssl.training.teacher.p5_frame_data import (
    FrameCacheDataset,
    ClassBalancedBatchSampler,
    frame_collate,
)
from papr_ssl.training.teacher.p5_dr import build_dr


SEEDS = (17, 29, 43)
NUM_CLASSES = 30
EMBED_DIM = 64
SCAF_K = 3
SCAF_MARGIN = 0.2
SCAF_SCALE = 30.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SubCenterArcFace(nn.Module):
    """Project-local P5 SCAF implementation frozen to K=3,m=.2,s=30."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = EMBED_DIM,
        k: int = SCAF_K,
        margin: float = SCAF_MARGIN,
        scale: float = SCAF_SCALE,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.k = k
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(
            torch.empty(num_classes, k, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        x = F.normalize(embeddings, p=2, dim=-1)
        w = F.normalize(self.weight, p=2, dim=-1)
        cosine_all = torch.einsum("bd,ckd->bck", x, w)
        cosine = cosine_all.max(dim=-1).values

        target = cosine.gather(1, labels[:, None]).squeeze(1)
        target = target.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        target_margin = torch.cos(torch.acos(target) + self.margin)

        logits = cosine.clone()
        logits.scatter_(1, labels[:, None], target_margin[:, None])
        return logits * self.scale


@torch.inference_mode()
def collect_embeddings(
    dr: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dr.eval()
    zs = []
    ys = []
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        m = batch["frame_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        z = dr(x, m)
        zs.append(z.cpu())
        ys.append(y.cpu())
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def prototype_macro_f1(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    dev_z: torch.Tensor,
    dev_y: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> float:
    prototypes = []
    for c in range(num_classes):
        cls = train_z[train_y == c]
        if cls.numel() == 0:
            raise RuntimeError(f"class {c} absent from train references")
        p = F.normalize(cls.mean(dim=0), p=2, dim=0)
        prototypes.append(p)
    prototypes = torch.stack(prototypes, dim=0)

    scores = F.normalize(dev_z, p=2, dim=-1) @ prototypes.T
    pred = scores.argmax(dim=1)

    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (dev_y == c)).sum())
        fp = int(((pred == c) & (dev_y != c)).sum())
        fn = int(((pred != c) & (dev_y == c)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("mean", "attention"), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--task-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/p5_runs/mean_vs_attention"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    seed_everything(args.seed)
    run_dir = args.run_root / args.method / f"seed_{args.seed:04d}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ds = FrameCacheDataset(
        cache_dir=args.cache_dir,
        task_index=args.task_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.cache_dir,
        task_index=args.task_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )
    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError("Core30 split size changed")
    if len(train_ds.class_to_index) != NUM_CLASSES:
        raise RuntimeError("Core30 class count changed")

    sampler = ClassBalancedBatchSampler(
        train_ds,
        classes_per_batch=8,
        samples_per_class=4,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )
    train_ref_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )

    device = torch.device(args.device)
    dr = build_dr(args.method).to(device)
    scaf = SubCenterArcFace().to(device)

    optimizer = torch.optim.AdamW(
        list(dr.parameters()) + list(scaf.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    metrics = []
    best = None
    for epoch in range(1, 21):
        sampler.set_epoch(epoch)
        dr.train()
        scaf.train()
        losses = []

        for batch in train_loader:
            x = batch["features"].to(device, non_blocking=True)
            m = batch["frame_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            z = dr(x, m)
            logits = scaf(z, y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(dr.parameters()) + list(scaf.parameters()),
                5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_z, train_y = collect_embeddings(dr, train_ref_loader, device)
        dev_z, dev_y = collect_embeddings(dr, dev_loader, device)
        score = prototype_macro_f1(train_z, train_y, dev_z, dev_y)

        checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "schema": "papr_ssl.p5_checkpoint.v1",
                "method": args.method,
                "seed": args.seed,
                "epoch": epoch,
                "generic_dev_score": score,
                "dr_state_dict": dr.state_dict(),
                "scaf_state_dict": scaf.state_dict(),
            },
            checkpoint_path,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / len(losses)),
            "generic_dev_score": score,
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_sha256": sha256(checkpoint_path),
        }
        metrics.append(row)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (
            best is None
            or score > best["generic_dev_score"]
            or (
                score == best["generic_dev_score"]
                and epoch < best["epoch"]
            )
        ):
            best = row

        print(
            f"[{args.method} seed={args.seed}] "
            f"epoch={epoch:02d} loss={row['train_loss']:.6f} "
            f"dev_macro_f1={score:.6f}"
        )

    if best is None:
        raise RuntimeError("no checkpoint selected")

    result = {
        "schema": "papr_ssl.p5_run_result.v1",
        "phase": "P5",
        "method": args.method,
        "seed": args.seed,
        "backbone": "wavlm_large",
        "model_id": "microsoft/wavlm-large",
        "model_revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
        "hidden_state_index": 15,
        "hidden_dim": 1024,
        "embedding_dim": 64,
        "scaf": {"k": 3, "margin_rad": 0.2, "scale": 30.0},
        "epochs": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip_norm": 5.0,
        "sampler": {"classes_per_batch": 8, "samples_per_class": 4},
        "selection_metric": "generic_dev_score",
        "selection_metric_definition": "prototype_macro_f1",
        "selected_epoch": int(best["epoch"]),
        "selected_generic_dev_score": float(best["generic_dev_score"]),
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "generic_test_accessed": False,
    }
    save_json(run_dir / "p5_result.json", result)
    save_json(
        run_dir / "selected_checkpoint.json",
        {"selected": best},
    )

    print("=" * 108)
    print(f"P5 {args.method.upper()} DR seed={args.seed}")
    print("=" * 108)
    print(f"selected epoch:              {best['epoch']}")
    print(f"selected generic_dev_score: {best['generic_dev_score']:.6f}")
    print("generic_test accessed:       NO")
    print("-" * 108)
    print("P5 SINGLE RUN: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
