#!/usr/bin/env python
"""Train clean single-layer vs learnable multi-layer WavLM fusion ablation.

Arms
----
single15:
    cached WavLM hidden_states[15] -> existing Attention DR -> 256D -> SCAF

weighted13_17:
    softmax-weighted sum of cached hidden_states[13..17]
    -> same Attention DR -> 256D -> SCAF

Only the fusion operator differs.
"""

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
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset

LAYERS = (13, 14, 15, 16, 17)
DEFAULT_MODES = ("single15", "weighted13_17")
DEFAULT_SEEDS = (17, 29, 43)


def replace_projection(model: nn.Module, dim: int = 256) -> nn.Module:
    model = copy.deepcopy(model)
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and m.out_features == 64]
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected exactly one 64-output Linear, got "
            f"{[(n,m.in_features,m.out_features) for n,m in cands]}"
        )
    name, old = cands[0]
    new = nn.Linear(old.in_features, dim, bias=(old.bias is not None))
    nn.init.xavier_uniform_(new.weight)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    model.set_submodule(name, new)
    return model


class LearnableLayerFusion(nn.Module):
    def __init__(self, n_layers: int = 5):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers, dtype=torch.float32))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def forward(self, x):
        # x: [B,L,T,D], normally float16 from cache.
        w = self.weights().to(dtype=x.dtype, device=x.device)
        fused = torch.einsum("l,bltd->btd", w, x)
        return fused.to(torch.float32)


class SCAFHead(nn.Module):
    def __init__(self, num_classes=30, embedding_dim=256, k=3,
                 margin_rad=0.2, scale=30.0):
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
        cls_cos = all_cos.max(dim=-1).values.clamp(-1+1e-7, 1-1e-7)
        theta = torch.acos(cls_cos.gather(1, target[:,None]).squeeze(1))
        target_cos = torch.cos(theta + self.margin_rad)
        logits = cls_cos.clone()
        logits.scatter_(1, target[:,None], target_cos[:,None])
        return logits * self.scale


class MultiLayerDataset:
    def __init__(self, *, cache_dir: Path, base_ds: FrameCacheDataset):
        self.cache_dir = Path(cache_dir)
        self.base_ds = base_ds
        self.rows = base_ds.rows
        self.class_to_index = base_ds.class_to_index

        idx = {}
        with (self.cache_dir / "index.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                idx[r["utt_id"]] = r["cache_relpath"]
        self.by_utt = idx

        missing = [r["utt_id"] for r in self.rows if r["utt_id"] not in idx]
        if missing:
            raise RuntimeError(f"{len(missing)} rows missing from multilayer cache; first={missing[:5]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        obj = torch.load(
            self.cache_dir / self.by_utt[row["utt_id"]],
            map_location="cpu",
            weights_only=False,
        )
        x = obj["features"]  # [5,T,1024]
        if tuple(obj["layers"]) != LAYERS:
            raise RuntimeError(f"Layer mismatch for {row['utt_id']}: {obj['layers']}")
        return {
            "utt_id": row["utt_id"],
            "features": x,
            "label_index": int(row["label_index"]),
        }


def collate(items):
    b = len(items)
    max_t = max(x["features"].shape[1] for x in items)
    dtype = items[0]["features"].dtype
    feat = torch.zeros((b, len(LAYERS), max_t, 1024), dtype=dtype)
    mask = torch.zeros((b, max_t), dtype=torch.bool)
    y = torch.empty((b,), dtype=torch.long)
    utt = []
    for i, item in enumerate(items):
        x = item["features"]
        t = x.shape[1]
        feat[i, :, :t] = x
        mask[i, :t] = True
        y[i] = item["label_index"]
        utt.append(item["utt_id"])
    return {"features": feat, "frame_mask": mask, "labels": y, "utt_ids": utt}


def class_index(ds):
    out = defaultdict(list)
    for i, row in enumerate(ds.rows):
        out[int(row["label_index"])].append(i)
    if len(out) != 30:
        raise RuntimeError(f"Expected 30 classes, got {len(out)}")
    return out


def sample_episode(rng, by_class, n_way=8, k_shot=4):
    classes = rng.sample(sorted(by_class), n_way)
    idx = []
    for c in classes:
        pool = by_class[c]
        chosen = rng.sample(pool, k_shot) if len(pool) >= k_shot else [rng.choice(pool) for _ in range(k_shot)]
        idx.extend(chosen)
    return idx


def make_batch(ds, indices):
    return collate([ds[i] for i in indices])


def choose_layer15(x):
    # layers tuple = [13,14,15,16,17], so layer15 is local index 2.
    return x[:, 2].to(torch.float32)


def fuse_features(mode, x, fusion):
    if mode == "single15":
        return choose_layer15(x)
    if mode == "weighted13_17":
        return fusion(x)
    raise ValueError(mode)


@torch.inference_mode()
def embed_dataset(mode, dr, fusion, ds, device, batch_size=48):
    dr.eval()
    if fusion is not None:
        fusion.eval()
    zs, ys = [], []
    for start in range(0, len(ds), batch_size):
        idx = list(range(start, min(len(ds), start+batch_size)))
        batch = make_batch(ds, idx)
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        h = fuse_features(mode, x, fusion)
        z = F.normalize(dr(h, mask).float(), dim=-1)
        zs.append(z.cpu())
        ys.append(batch["labels"])
    return torch.cat(zs, 0), torch.cat(ys, 0).numpy()


def prototype_macro_f1(train_z, train_y, dev_z, dev_y):
    train_z = F.normalize(train_z.float(), dim=-1)
    dev_z = F.normalize(dev_z.float(), dim=-1)
    protos = []
    for c in range(30):
        mask = torch.from_numpy(train_y == c)
        p = train_z[mask].mean(0, keepdim=True)
        protos.append(F.normalize(p, dim=-1))
    protos = torch.cat(protos, 0)
    pred = (dev_z @ protos.T).argmax(1).numpy()

    f1 = []
    for c in range(30):
        tp = int(((pred == c) & (dev_y == c)).sum())
        fp = int(((pred == c) & (dev_y != c)).sum())
        fn = int(((pred != c) & (dev_y == c)).sum())
        den = 2*tp + fp + fn
        f1.append((2*tp/den) if den else 0.0)
    return float(np.mean(f1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(*, mode, seed, train_ds, dev_ds, out_dir, device,
              epochs, steps_per_epoch):
    seed_all(seed)

    dr = replace_projection(build_dr("attention"), 256).to(device)
    scaf = SCAFHead().to(device)
    fusion = LearnableLayerFusion(5).to(device) if mode == "weighted13_17" else None

    params = list(dr.parameters()) + list(scaf.parameters())
    if fusion is not None:
        params += list(fusion.parameters())

    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    by_class = class_index(train_ds)
    rng = random.Random(seed)

    run_dir = out_dir / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    best = -1.0
    best_epoch = None
    history = []

    for epoch in range(1, epochs+1):
        dr.train()
        scaf.train()
        if fusion is not None:
            fusion.train()

        losses = []
        for _ in range(steps_per_epoch):
            idx = sample_episode(rng, by_class, 8, 4)
            batch = make_batch(train_ds, idx)
            x = batch["features"].to(device, non_blocking=True)
            mask = batch["frame_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            h = fuse_features(mode, x, fusion)
            z = F.normalize(dr(h, mask), dim=-1)
            loss = F.cross_entropy(scaf(z, y), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss mode={mode} seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        train_z, train_y = embed_dataset(mode, dr, fusion, train_ds, device)
        dev_z, dev_y = embed_dataset(mode, dr, fusion, dev_ds, device)
        score = prototype_macro_f1(train_z, train_y, dev_z, dev_y)

        weights = None
        if fusion is not None:
            weights = [float(v) for v in fusion.weights().detach().cpu()]

        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "generic_dev_score": score,
            "fusion_weights_13_17": weights,
        }
        history.append(record)

        weight_text = ""
        if weights is not None:
            weight_text = " weights=" + ",".join(f"{v:.3f}" for v in weights)

        print(
            f"[{mode} seed={seed}] epoch={epoch:02d}/{epochs} "
            f"loss={np.mean(losses):.5f} dev_macro_f1={score:.6f}{weight_text}"
        )

        if score > best:
            best = score
            best_epoch = epoch
            torch.save(
                {
                    "schema": "papr_ssl.wavlm_multilayer_fusion_checkpoint.v1",
                    "mode": mode,
                    "seed": seed,
                    "selected_epoch": epoch,
                    "generic_dev_score": score,
                    "embedding_dim": 256,
                    "layers": list(LAYERS),
                    "fusion_state_dict": None if fusion is None else fusion.state_dict(),
                    "selected_fusion_weights_13_17": weights,
                    "dr_state_dict": dr.state_dict(),
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
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--multilayer-cache",
        type=Path,
        default=Path("artifacts/p6_wavlm_layers13_17_cache"),
    )
    p.add_argument(
        "--frame-cache15",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_wavlm_multilayer_fusion"),
    )
    p.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_manifest = json.loads(
        (args.multilayer_cache / "manifest.json").read_text(encoding="utf-8")
    )
    if cache_manifest["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("Multilayer cache test seal violated")
    if tuple(cache_manifest["layers"]) != LAYERS:
        raise RuntimeError(f"Expected cached layers {LAYERS}, got {cache_manifest['layers']}")

    train_base = FrameCacheDataset(
        cache_dir=args.frame_cache15,
        task_index=args.core_index,
        split="train",
    )
    dev_base = FrameCacheDataset(
        cache_dir=args.frame_cache15,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_base.class_to_index,
    )
    train_ds = MultiLayerDataset(cache_dir=args.multilayer_cache, base_ds=train_base)
    dev_ds = MultiLayerDataset(cache_dir=args.multilayer_cache, base_ds=dev_base)

    if args.smoke:
        args.modes = ["single15", "weighted13_17"]
        args.seeds = [17]
        args.epochs = 1
        args.steps_per_epoch = 2

    steps = args.steps_per_epoch or math.ceil(len(train_ds) / 32)

    manifest = {
        "schema": "papr_ssl.wavlm_multilayer_fusion_experiment.v1",
        "scientific_question": "Does learnable fusion of WavLM layers 13..17 improve over the same-precision layer15 control?",
        "arms": list(args.modes),
        "layers": list(LAYERS),
        "weighted_fusion": "global trainable softmax scalar weights; uniform initialization",
        "embedding_dim": 256,
        "dr": "same existing Attention DR; final projection changed to 256D",
        "scaf": {"k": 3, "margin_rad": 0.2, "scale": 30.0},
        "epochs": args.epochs,
        "steps_per_epoch": steps,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "sampler": "8-way x 4-shot",
        "seeds": list(args.seeds),
        "selection_metric": "dev prototype Macro-F1",
        "cache_storage_dtype": cache_manifest["storage_dtype"],
        "canonical_64d_teacher": "preserved_not_modified",
        "project1_256d_single_layer_reference_mean": 0.912667,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
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
                    epochs=args.epochs,
                    steps_per_epoch=steps,
                )
            )

    (args.output_dir / "raw_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("WAVLM MULTI-LAYER FUSION TRAINING COMPLETE")
    print(f"modes: {args.modes}")
    print(f"seeds: {args.seeds}")
    print("canonical 64D modified: NO")
    print("generic_test accessed:  NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
