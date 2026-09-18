#!/usr/bin/env python
"""Matched Attention-Mean vs Attentive Statistics Pooling ablation.

Scientific question
-------------------
Does adding the attentive temporal standard deviation improve the current
single-layer WavLM[15] -> Attention -> 256D representation?

Arms
----
attention_mean_256:
    alpha_t = softmax(v^T tanh(W h_t))
    mu = sum_t alpha_t h_t
    z = Linear(mu) -> 256D -> L2

attentive_stats_256:
    same alpha_t
    mu = sum_t alpha_t h_t
    sigma = sqrt(sum_t alpha_t (h_t - mu)^2)
    z = Linear([mu, sigma]) -> 256D -> L2

Only the pooling statistic differs. WavLM remains frozen and the same cached
layer-15 features are used by both arms.
"""

from __future__ import annotations

import argparse
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
from papr_ssl.training.teacher.p5_frame_data import (
    FrameCacheDataset,
    frame_collate,
)

DEFAULT_MODES = ("attention_mean_256", "attentive_stats_256")
DEFAULT_SEEDS = (17, 29, 43)


def infer_attention_hidden_dim():
    ref = build_dr("attention")
    linears = [
        (name, int(m.in_features), int(m.out_features))
        for name, m in ref.named_modules()
        if isinstance(m, nn.Linear)
    ]

    candidates = []
    for _, in_f, out_f in linears:
        if in_f == 1024 and out_f not in (1, 64):
            if any(i2 == out_f and o2 == 1 for _, i2, o2 in linears):
                candidates.append(out_f)

    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely infer attention hidden dim. "
            f"Linears={linears}, candidates={candidates}"
        )
    return candidates[0], linears


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x [B,T,D], mask [B,T]
        logits = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=1)
        alpha = alpha * mask.to(alpha.dtype)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return alpha


class AttentionMean256(nn.Module):
    def __init__(self, input_dim=1024, attn_hidden=128, out_dim=256):
        super().__init__()
        self.attention = AdditiveAttention(input_dim, attn_hidden)
        self.projection = nn.Linear(input_dim, out_dim)

    def forward(self, x, mask):
        x = x.to(torch.float32)
        alpha = self.attention(x, mask)
        mu = torch.sum(alpha.unsqueeze(-1) * x, dim=1)
        z = self.projection(mu)
        return F.normalize(z, dim=-1)


class AttentiveStats256(nn.Module):
    def __init__(self, input_dim=1024, attn_hidden=128, out_dim=256, eps=1e-5):
        super().__init__()
        self.attention = AdditiveAttention(input_dim, attn_hidden)
        self.projection = nn.Linear(input_dim * 2, out_dim)
        self.eps = float(eps)

    def forward(self, x, mask):
        x = x.to(torch.float32)
        alpha = self.attention(x, mask)
        w = alpha.unsqueeze(-1)

        mu = torch.sum(w * x, dim=1)
        second = torch.sum(w * (x * x), dim=1)
        var = (second - mu * mu).clamp_min(self.eps)
        sigma = torch.sqrt(var)

        pooled = torch.cat([mu, sigma], dim=-1)
        z = self.projection(pooled)
        return F.normalize(z, dim=-1)


class SCAFHead(nn.Module):
    def __init__(
        self,
        num_classes=30,
        embedding_dim=256,
        k=3,
        margin_rad=0.2,
        scale=30.0,
    ):
        super().__init__()
        self.margin_rad = float(margin_rad)
        self.scale = float(scale)
        weight = torch.empty(num_classes, k, embedding_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, z, target):
        z = F.normalize(z, dim=-1)
        w = F.normalize(self.weight, dim=-1)

        all_cos = torch.einsum("bd,ckd->bck", z, w)
        cls_cos = all_cos.max(dim=-1).values
        cls_cos = cls_cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(
            cls_cos.gather(1, target[:, None]).squeeze(1)
        )
        target_cos = torch.cos(theta + self.margin_rad)

        logits = cls_cos.clone()
        logits.scatter_(1, target[:, None], target_cos[:, None])
        return logits * self.scale


def build_pooler(mode: str, attn_hidden: int):
    if mode == "attention_mean_256":
        return AttentionMean256(
            input_dim=1024,
            attn_hidden=attn_hidden,
            out_dim=256,
        )
    if mode == "attentive_stats_256":
        return AttentiveStats256(
            input_dim=1024,
            attn_hidden=attn_hidden,
            out_dim=256,
        )
    raise ValueError(mode)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_class_index(dataset):
    out = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        c = int(row["label_index"])
        out[c].append(i)
    if len(out) != 30:
        raise RuntimeError(f"Expected 30 classes, got {len(out)}")
    return out


def sample_episode(rng, by_class, n_way=8, k_shot=4):
    classes = rng.sample(sorted(by_class), n_way)
    indices = []
    for c in classes:
        pool = by_class[c]
        if len(pool) >= k_shot:
            chosen = rng.sample(pool, k_shot)
        else:
            chosen = [rng.choice(pool) for _ in range(k_shot)]
        indices.extend(chosen)
    return indices


def make_batch(dataset, indices):
    return frame_collate([dataset[i] for i in indices])


@torch.inference_mode()
def embed_dataset(model, dataset, device, batch_size=64):
    model.eval()
    zs = []
    ys = []

    for start in range(0, len(dataset), batch_size):
        idx = list(range(start, min(len(dataset), start + batch_size)))
        batch = make_batch(dataset, idx)

        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        z = model(x, mask)

        zs.append(z.cpu())
        ys.extend(int(dataset.rows[i]["label_index"]) for i in idx)

    return (
        torch.cat(zs, dim=0),
        np.asarray(ys, dtype=np.int64),
    )


def prototype_macro_f1(train_z, train_y, dev_z, dev_y):
    train_z = F.normalize(train_z.to(torch.float32), dim=-1)
    dev_z = F.normalize(dev_z.to(torch.float32), dim=-1)

    protos = []
    for c in range(30):
        mask = torch.from_numpy(train_y == c)
        if not bool(mask.any()):
            raise RuntimeError(f"Class {c} missing from train set")
        p = train_z[mask].mean(dim=0, keepdim=True)
        protos.append(F.normalize(p, dim=-1))
    protos = torch.cat(protos, dim=0)

    pred = (dev_z @ protos.T).argmax(dim=1).cpu().numpy()

    f1s = []
    for c in range(30):
        tp = int(np.sum((pred == c) & (dev_y == c)))
        fp = int(np.sum((pred == c) & (dev_y != c)))
        fn = int(np.sum((pred != c) & (dev_y == c)))
        den = 2 * tp + fp + fn
        f1s.append((2 * tp / den) if den else 0.0)
    return float(np.mean(f1s))


def train_one(
    *,
    mode,
    seed,
    train_ds,
    dev_ds,
    out_dir,
    device,
    attn_hidden,
    epochs,
    steps_per_epoch,
):
    seed_all(seed)

    pooler = build_pooler(mode, attn_hidden).to(device)
    scaf = SCAFHead().to(device)

    params = list(pooler.parameters()) + list(scaf.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=1e-3,
        weight_decay=1e-4,
    )

    by_class = build_class_index(train_ds)
    rng = random.Random(seed)

    run_dir = out_dir / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    best = -1.0
    best_epoch = None
    history = []

    for epoch in range(1, epochs + 1):
        pooler.train()
        scaf.train()

        losses = []
        for _ in range(steps_per_epoch):
            indices = sample_episode(rng, by_class, 8, 4)
            batch = make_batch(train_ds, indices)

            x = batch["features"].to(device, non_blocking=True)
            mask = batch["frame_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            z = pooler(x, mask)
            logits = scaf(z, y)
            loss = F.cross_entropy(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss mode={mode} seed={seed} epoch={epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        train_z, train_y = embed_dataset(
            pooler, train_ds, device
        )
        dev_z, dev_y = embed_dataset(
            pooler, dev_ds, device
        )
        score = prototype_macro_f1(
            train_z, train_y, dev_z, dev_y
        )

        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "generic_dev_score": score,
        }
        history.append(record)

        print(
            f"[{mode} seed={seed}] "
            f"epoch={epoch:02d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"dev_macro_f1={score:.6f}"
        )

        if score > best:
            best = score
            best_epoch = epoch

            torch.save(
                {
                    "schema": (
                        "papr_ssl.attentive_stats_ablation_checkpoint.v1"
                    ),
                    "mode": mode,
                    "seed": seed,
                    "selected_epoch": epoch,
                    "generic_dev_score": score,
                    "embedding_dim": 256,
                    "attn_hidden_dim": attn_hidden,
                    "pooler_state_dict": pooler.state_dict(),
                    "scaf_state_dict": scaf.state_dict(),
                    "generic_test": "sealed_not_accessed",
                },
                run_dir / "best.pt",
            )

    result = {
        "mode": mode,
        "seed": seed,
        "selected_epoch": int(best_epoch),
        "selected_generic_dev_score": float(best),
        "history": history,
        "generic_test": "sealed_not_accessed",
    }
    (run_dir / "result.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_attentive_stats_ablation"),
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output dir: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attn_hidden, current_linears = infer_attention_hidden_dim()

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

    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError(
            f"Unexpected Core30 counts "
            f"train={len(train_ds)} dev={len(dev_ds)}"
        )

    if args.smoke:
        args.modes = [
            "attention_mean_256",
            "attentive_stats_256",
        ]
        args.seeds = [17]
        args.epochs = 1
        args.steps_per_epoch = 2

    steps = (
        args.steps_per_epoch
        if args.steps_per_epoch > 0
        else math.ceil(len(train_ds) / 32)
    )

    manifest = {
        "schema": "papr_ssl.attentive_stats_ablation.v1",
        "scientific_question": (
            "Does attentive temporal standard deviation improve "
            "single-layer WavLM[15] 256D phrase representation?"
        ),
        "arms": list(args.modes),
        "frozen": {
            "backbone": "microsoft/wavlm-large",
            "hidden_state_index": 15,
            "frame_cache": str(args.frame_cache),
            "task": "MDSC Core30 train/dev",
            "embedding_dim": 256,
            "attention_hidden_dim": attn_hidden,
            "scaf": {
                "k": 3,
                "margin_rad": 0.2,
                "scale": 30.0,
            },
            "epochs": args.epochs,
            "steps_per_epoch": steps,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 5.0,
            "sampler": "8-way x 4-shot",
            "seeds": list(args.seeds),
            "selection_metric": "dev prototype Macro-F1",
        },
        "difference_only": {
            "attention_mean_256": (
                "weighted mean mu -> Linear(1024,256)"
            ),
            "attentive_stats_256": (
                "[weighted mean mu, weighted std sigma] "
                "-> Linear(2048,256)"
            ),
        },
        "current_attention_linear_audit": current_linears,
        "prior_project1_256d_mean": 0.912667,
        "canonical_64d_teacher": "preserved_not_modified",
        "project1_256d_teacher": "preserved_not_overwritten",
        "generic_test": "sealed_not_accessed",
    }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    device = torch.device(args.device)

    results = []
    for mode in args.modes:
        for seed in args.seeds:
            results.append(
                train_one(
                    mode=mode,
                    seed=seed,
                    train_ds=train_ds,
                    dev_ds=dev_ds,
                    out_dir=args.output_dir,
                    device=device,
                    attn_hidden=attn_hidden,
                    epochs=args.epochs,
                    steps_per_epoch=steps,
                )
            )

    (args.output_dir / "raw_results.json").write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("ATTENTIVE STATISTICS POOLING ABLATION COMPLETE")
    print(f"attention hidden dim: {attn_hidden}")
    print(f"modes:                {args.modes}")
    print(f"seeds:                {args.seeds}")
    print("canonical 64D modified: NO")
    print("Project-1 256D modified: NO")
    print("generic_test accessed:  NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
