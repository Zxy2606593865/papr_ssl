#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset, frame_collate

DEFAULT_DIMS = (64, 128, 256, 512)
DEFAULT_SEEDS = (17, 29, 43)


def find_and_replace_embedding_projection(model: nn.Module, embedding_dim: int) -> nn.Module:
    model = copy.deepcopy(model)
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 64:
            candidates.append((name, module))
    if len(candidates) != 1:
        detail = [(n, m.in_features, m.out_features) for n, m in candidates]
        raise RuntimeError(
            f"Expected exactly one 64-output Linear in Attention DR; found {len(candidates)}: {detail}"
        )
    name, old = candidates[0]
    if embedding_dim == 64:
        return model

    new = nn.Linear(old.in_features, embedding_dim, bias=(old.bias is not None))
    nn.init.xavier_uniform_(new.weight)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    model.set_submodule(name, new)
    return model


class SCAFHead(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, k: int = 3,
                 margin_rad: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.k = int(k)
        self.margin_rad = float(margin_rad)
        self.scale = float(scale)
        weight = torch.empty(num_classes, k, embedding_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cos_all = torch.einsum("bd,ckd->bck", z, w)
        cos_cls = cos_all.max(dim=-1).values
        cos_cls = cos_cls.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta_y = torch.acos(cos_cls.gather(1, target[:, None]).squeeze(1))
        target_cos = torch.cos(theta_y + self.margin_rad)
        logits = cos_cls.clone()
        logits.scatter_(1, target[:, None], target_cos[:, None])
        return logits * self.scale


def row_label_index(dataset: FrameCacheDataset, row: dict) -> int:
    if "label_index" in row:
        return int(row["label_index"])
    label = None
    for key in ("label_text", "task_label", "label"):
        if key in row and row[key] not in (None, ""):
            label = str(row[key])
            break
    if label is None:
        raise RuntimeError(f"Cannot recover label from row: {row}")
    return int(dataset.class_to_index[label])


def build_index_by_class(dataset: FrameCacheDataset):
    by_class = defaultdict(list)
    labels = []
    for i, row in enumerate(dataset.rows):
        c = row_label_index(dataset, row)
        labels.append(c)
        by_class[c].append(i)
    classes = sorted(by_class)
    if len(classes) != 30:
        raise RuntimeError(f"Expected 30 Core30 classes, got {len(classes)}")
    return by_class, np.asarray(labels, dtype=np.int64)


def sample_episode_indices(rng: random.Random, by_class: dict[int, list[int]],
                           n_way: int = 8, k_shot: int = 4):
    classes = rng.sample(sorted(by_class), n_way)
    indices, labels = [], []
    for c in classes:
        pool = by_class[c]
        chosen = rng.sample(pool, k_shot) if len(pool) >= k_shot else [rng.choice(pool) for _ in range(k_shot)]
        indices.extend(chosen)
        labels.extend([c] * k_shot)
    return indices, torch.tensor(labels, dtype=torch.long)


def make_batch(dataset: FrameCacheDataset, indices: list[int]):
    return frame_collate([dataset[i] for i in indices])


@torch.inference_mode()
def embed_dataset(model: nn.Module, dataset: FrameCacheDataset,
                  device: torch.device, batch_size: int = 64):
    model.eval()
    all_z, all_labels = [], []
    for start in range(0, len(dataset), batch_size):
        idx = list(range(start, min(len(dataset), start + batch_size)))
        batch = make_batch(dataset, idx)
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        z = F.normalize(model(x, mask).to(torch.float32), dim=-1)
        all_z.append(z.cpu())
        all_labels.extend(row_label_index(dataset, dataset.rows[i]) for i in idx)
    return torch.cat(all_z, dim=0), np.asarray(all_labels, dtype=np.int64)


def prototype_macro_f1(train_z, train_y, dev_z, dev_y, num_classes: int = 30):
    train_z = F.normalize(train_z.to(torch.float32), dim=-1)
    dev_z = F.normalize(dev_z.to(torch.float32), dim=-1)
    protos = []
    for c in range(num_classes):
        mask = torch.from_numpy(train_y == c)
        p = train_z[mask].mean(dim=0, keepdim=True)
        protos.append(F.normalize(p, dim=-1))
    protos = torch.cat(protos, dim=0)
    pred = (dev_z @ protos.T).argmax(dim=1).cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (dev_y == c)))
        fp = int(np.sum((pred == c) & (dev_y != c)))
        fn = int(np.sum((pred != c) & (dev_y == c)))
        den = 2 * tp + fp + fn
        f1s.append((2 * tp / den) if den else 0.0)
    return float(np.mean(f1s))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(*, dim, seed, train_ds, dev_ds, output_dir, device,
              epochs, steps_per_epoch, lr, weight_decay, grad_clip):
    seed_everything(seed)
    dr = find_and_replace_embedding_projection(build_dr("attention"), dim).to(device)
    scaf = SCAFHead(30, dim, k=3, margin_rad=0.2, scale=30.0).to(device)
    params = list(dr.parameters()) + list(scaf.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    by_class, _ = build_index_by_class(train_ds)
    rng = random.Random(seed)

    best_score, best_epoch = -1.0, None
    history = []
    run_dir = output_dir / f"dim_{dim}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    for epoch in range(1, epochs + 1):
        dr.train()
        scaf.train()
        losses = []
        for _ in range(steps_per_epoch):
            indices, y = sample_episode_indices(rng, by_class, 8, 4)
            batch = make_batch(train_ds, indices)
            x = batch["features"].to(device, non_blocking=True)
            mask = batch["frame_mask"].to(device, non_blocking=True)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            z = F.normalize(dr(x, mask), dim=-1)
            loss = F.cross_entropy(scaf(z, y), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss dim={dim} seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_z, train_y = embed_dataset(dr, train_ds, device)
        dev_z, dev_y = embed_dataset(dr, dev_ds, device)
        dev_score = prototype_macro_f1(train_z, train_y, dev_z, dev_y)
        history.append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "generic_dev_score": dev_score,
        })
        print(
            f"[dim={dim:4d} seed={seed:2d}] epoch={epoch:02d}/{epochs} "
            f"loss={np.mean(losses):.5f} dev_macro_f1={dev_score:.6f}"
        )

        if dev_score > best_score:
            best_score = dev_score
            best_epoch = epoch
            torch.save({
                "schema": "papr_ssl.teacher_dim_sweep_checkpoint.v1",
                "method": "attention",
                "embedding_dim": dim,
                "seed": seed,
                "selected_epoch": epoch,
                "generic_dev_score": dev_score,
                "dr_state_dict": dr.state_dict(),
                "scaf_state_dict": scaf.state_dict(),
                "scaf": {"k": 3, "margin_rad": 0.2, "scale": 30.0},
                "generic_test": "sealed_not_accessed",
            }, run_dir / "best.pt")

    result = {
        "embedding_dim": dim,
        "seed": seed,
        "selected_epoch": int(best_epoch),
        "selected_generic_dev_score": float(best_score),
        "history": history,
        "generic_test": "sealed_not_accessed",
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-cache", type=Path,
                   default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"))
    p.add_argument("--core-index", type=Path,
                   default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/p6_teacher_dim_sweep"))
    p.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing sweep: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.dims = [64]
        args.seeds = [17]
        args.epochs = 1
        args.steps_per_epoch = 2

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )

    if len(train_ds) != 3756:
        raise RuntimeError(f"Expected Core30 train=3756, got {len(train_ds)}")
    if len(dev_ds) != 442:
        raise RuntimeError(f"Expected Core30 dev=442, got {len(dev_ds)}")

    probe = build_dr("attention")
    candidates = [
        (n, m.in_features, m.out_features)
        for n, m in probe.named_modules()
        if isinstance(m, nn.Linear) and m.out_features == 64
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Attention DR projection audit failed: {candidates}")

    steps = args.steps_per_epoch or math.ceil(len(train_ds) / 32)

    manifest = {
        "schema": "papr_ssl.teacher_dim_sweep.v1",
        "canonical_64d_policy": {
            "status": "PRESERVE_READ_ONLY",
            "note": "Existing P5 64D Teacher remains canonical for Project 2 hardware. Sweep 64D is reproduction control only."
        },
        "frozen": {
            "backbone": "microsoft/wavlm-large",
            "hidden_state_index": 15,
            "task": "MDSC Core30 train/dev",
            "dr_family": "existing Attention DR; only final projection dimension varies",
            "scaf": {"k": 3, "margin_rad": 0.2, "scale": 30.0},
            "epochs": args.epochs,
            "steps_per_epoch": steps,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "sampler": "8-way x 4-shot",
            "selection_metric": "dev prototype_macro_f1",
        },
        "variable": {"embedding_dims": args.dims, "seeds": args.seeds},
        "attention_projection_audit": candidates,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    device = torch.device(args.device)
    results = []
    for dim in args.dims:
        for seed in args.seeds:
            results.append(train_one(
                dim=dim, seed=seed,
                train_ds=train_ds, dev_ds=dev_ds,
                output_dir=args.output_dir, device=device,
                epochs=args.epochs, steps_per_epoch=steps,
                lr=args.lr, weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
            ))

    (args.output_dir / "raw_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("TEACHER DIMENSION SWEEP TRAINING COMPLETE")
    print(f"dims: {args.dims}")
    print(f"seeds: {args.seeds}")
    print("generic_test: SEALED / NOT ACCESSED")
    print("canonical 64D Teacher: PRESERVED / NOT OVERWRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
