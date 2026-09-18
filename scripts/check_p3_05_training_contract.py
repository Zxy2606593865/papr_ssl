#!/usr/bin/env python
"""Standalone P3-05 training contract using project MeanDR + SCAF classes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from papr_ssl.training.feature_data import feature_cache_collate
from papr_ssl.training.teacher.train import (
    OptimizationConfig,
    build_project_mean_dr,
    build_project_scaf,
    run_training,
)


class ToyDataset(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return int(self.ys.shape[0])

    def __getitem__(self, i):
        return {
            "features": self.xs[i],
            "label": int(self.ys[i]),
            "task_label": str(int(self.ys[i])),
            "dataset": "p3_05_smoke",
            "utt_id": f"u{i:04d}",
            "split": "train",
            "sample_hash": f"{i:064x}",
            "frame_count": 1,
        }


def make_loader(xs, ys, batch_size):
    return DataLoader(
        ToyDataset(xs, ys),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=feature_cache_collate,
    )


def main() -> int:
    torch.manual_seed(17)
    input_dim = 8
    num_classes = 4

    centers = torch.eye(num_classes, input_dim) * 4.0
    xs = []
    ys = []
    for c in range(num_classes):
        for _ in range(10):
            xs.append(centers[c] + 0.08 * torch.randn(input_dim))
            ys.append(c)
    xs = torch.stack(xs)
    ys = torch.tensor(ys, dtype=torch.long)

    mean_dr = build_project_mean_dr(
        input_dim=input_dim,
        embedding_dim=64,
    )
    scaf = build_project_scaf(
        num_classes=num_classes,
        embedding_dim=64,
        k=3,
        margin=0.2,
        scale=30.0,
    )

    train_loader = make_loader(xs, ys, 8)
    reference_loader = make_loader(xs, ys, 16)
    dev_loader = make_loader(xs.clone(), ys.clone(), 16)

    with tempfile.TemporaryDirectory() as td:
        history = run_training(
            mean_dr=mean_dr,
            scaf=scaf,
            train_loader=train_loader,
            train_reference_loader=reference_loader,
            dev_loader=dev_loader,
            num_classes=num_classes,
            seed=17,
            run_dir=Path(td) / "smoke_run",
            device=torch.device("cpu"),
            optimization=OptimizationConfig(
                epochs=2,
                learning_rate=1e-2,
                weight_decay=0.0,
                grad_clip_norm=5.0,
            ),
        )

        last = history[-1]
        dev = last["dev"]
        assert len(last["checkpoint_sha256"]) == 64
        assert dev["generic_dev_score_name"] == "prototype_macro_f1"
        assert dev["embedding_norm_max_abs_error"] < 1e-4
        assert dev["collapsed"] is False

        print("=" * 96)
        print("PAPR-SSL P3-05 MEAN DR + SCAF TRAINING CONTRACT")
        print("=" * 96)
        print(f"epochs:                         {len(history)}")
        print(
            "train loss:                     "
            f"{history[0]['train_loss']:.6f} -> "
            f"{history[-1]['train_loss']:.6f}"
        )
        print(
            "generic_dev_score:              "
            f"{dev['generic_dev_score']:.6f}"
        )
        print(
            "generic_dev_score definition:   "
            f"{dev['generic_dev_score_name']}"
        )
        print(
            "64D norm max abs error:          "
            f"{dev['embedding_norm_max_abs_error']:.3e}"
        )
        print(f"embedding collapse:             {dev['collapsed']}")
        print(
            "checkpoint SHA-256 recorded:     "
            f"{last['checkpoint_sha256'][:16]}..."
        )
        print("sealed test accessed:            NO")
        print("-" * 96)
        print("P3-05 STATUS: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
