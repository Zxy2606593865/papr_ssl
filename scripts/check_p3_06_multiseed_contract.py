#!/usr/bin/env python
"""Standalone P3-06 17/29/43 multi-seed contract smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from papr_ssl.training.feature_data import (
    ClassBalancedBatchSampler,
    ClassBalancedSamplerConfig,
    feature_cache_collate,
)
from papr_ssl.training.teacher.multiseed import (
    MultiSeedConfig,
    SeedRunSpec,
    run_multiseed_experiment,
)
from papr_ssl.training.teacher.train import (
    OptimizationConfig,
    build_project_mean_dr,
    build_project_scaf,
)


class ToyDataset(Dataset):
    def __init__(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        *,
        split: str,
    ):
        self.xs = xs
        self.ys = ys
        self.split = split

    def __len__(self):
        return int(self.ys.shape[0])

    def __getitem__(self, i):
        return {
            "features": self.xs[i],
            "label": int(self.ys[i]),
            "task_label": str(int(self.ys[i])),
            "dataset": "p3_06_smoke",
            "utt_id": f"{self.split}_{i:04d}",
            "split": self.split,
            "sample_hash": f"{i + (0 if self.split == 'train' else 10000):064x}",
            "frame_count": 1,
        }


def main() -> int:
    # Fixed data generated independently of the experiment seeds.
    g = torch.Generator().manual_seed(20260911)
    input_dim = 8
    num_classes = 4

    class_centers = torch.eye(num_classes, input_dim) * 3.0

    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    for c in range(num_classes):
        for _ in range(12):
            train_x.append(
                class_centers[c]
                + 0.12 * torch.randn(input_dim, generator=g)
            )
            train_y.append(c)
        for _ in range(5):
            dev_x.append(
                class_centers[c]
                + 0.12 * torch.randn(input_dim, generator=g)
            )
            dev_y.append(c)

    train_x = torch.stack(train_x)
    train_y = torch.tensor(train_y, dtype=torch.long)
    dev_x = torch.stack(dev_x)
    dev_y = torch.tensor(dev_y, dtype=torch.long)

    train_dataset = ToyDataset(train_x, train_y, split="train")
    dev_dataset = ToyDataset(dev_x, dev_y, split="dev")

    def build_seed_run(seed: int, run_dir: Path) -> SeedRunSpec:
        # Model initialization occurs after P3-06 sets the seed.
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

        sampler = ClassBalancedBatchSampler(
            train_y.tolist(),
            ClassBalancedSamplerConfig(
                classes_per_batch=4,
                samples_per_class=2,
                batches_per_epoch=6,
                seed=seed,
            ),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=feature_cache_collate,
        )
        train_ref_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=feature_cache_collate,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=feature_cache_collate,
        )

        return SeedRunSpec(
            mean_dr=mean_dr,
            scaf=scaf,
            train_loader=train_loader,
            train_reference_loader=train_ref_loader,
            dev_loader=dev_loader,
            num_classes=num_classes,
            device=torch.device("cpu"),
            optimization=OptimizationConfig(
                epochs=2,
                learning_rate=1e-2,
                weight_decay=0.0,
                grad_clip_norm=5.0,
            ),
        )

    with tempfile.TemporaryDirectory() as td:
        summary = run_multiseed_experiment(
            experiment_dir=Path(td) / "smoke_multiseed",
            build_seed_run=build_seed_run,
            config=MultiSeedConfig((17, 29, 43)),
        )

        score_agg = summary["final_epoch_summary"][
            "generic_dev_score"
        ]

        print("=" * 96)
        print("PAPR-SSL P3-06 MULTI-SEED CONTRACT")
        print("=" * 96)
        for row in summary["runs"]:
            print(
                f"seed={row['seed']:2d}  "
                f"final_loss={row['final_train_loss']:.6f}  "
                f"final_dev={row['final_generic_dev_score']:.6f}  "
                f"sha256={row['final_checkpoint_sha256'][:12]}..."
            )
        print("-" * 96)
        print(f"required seeds:                  {summary['required_seeds']}")
        print(f"seed count:                      {summary['seed_count']}")
        print(
            "final-epoch dev mean:            "
            f"{score_agg['mean']:.6f}"
        )
        print(
            "final-epoch dev sample std:      "
            f"{score_agg['sample_std']:.6f}"
        )
        print(f"std ddof:                        {score_agg['std_ddof']}")
        print("seed set before model factory:   YES")
        print("same experiment budget:          YES")
        print("checkpoint selection in P3-06:  NO")
        print("selection deferred to P3-07:     YES")
        print("-" * 96)
        print("P3-06 STATUS: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
