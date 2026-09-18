from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from papr_ssl.training.feature_data import feature_cache_collate
from papr_ssl.training.teacher.evaluate import (
    GENERIC_DEV_SCORE_NAME,
    assert_unit_norm,
    cached_mean_dr_forward,
    evaluate_embedding_space,
)
from papr_ssl.training.teacher.train import (
    OptimizationConfig,
    checkpoint_sha256,
    run_training,
)


class TinyMeanDR(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, 64)

    def forward(self, features, frame_mask):
        weights = frame_mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * weights).sum(1) / weights.sum(1)
        z = self.projection(pooled)
        return F.normalize(z, dim=1)


class TinySCAF(nn.Module):
    def __init__(self, num_classes: int, k: int = 3):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, k, 64)
        )
        self.scale = 10.0

    def forward(self, embedding, labels):
        c = F.normalize(self.centers, dim=-1)
        logits = torch.einsum("bd,ckd->bck", embedding, c).max(-1).values
        return F.cross_entropy(self.scale * logits, labels)


class DictDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return {
            "features": self.xs[i],
            "label": int(self.ys[i]),
            "task_label": str(int(self.ys[i])),
            "dataset": "toy",
            "utt_id": f"u{i}",
            "split": "train",
            "sample_hash": f"{i:064x}",
            "frame_count": 1,
        }


def _loader(xs, ys, batch_size=8):
    return DataLoader(
        DictDataset(xs, ys),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=feature_cache_collate,
    )


class TeacherTrainingLoopTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_cached_mean_dr_keeps_64d_unit_norm_contract(self):
        head = TinyMeanDR(6)
        z = cached_mean_dr_forward(
            head,
            torch.randn(5, 6),
        )
        self.assertEqual(tuple(z.shape), (5, 64))
        assert_unit_norm(z)

    def test_dev_metrics_use_train_prototypes_and_freeze_score_name(self):
        # Three clearly separated input classes.
        train_x = torch.cat(
            [
                torch.tensor([[3.0, 0.0, 0.0]]).repeat(5, 1),
                torch.tensor([[0.0, 3.0, 0.0]]).repeat(5, 1),
                torch.tensor([[0.0, 0.0, 3.0]]).repeat(5, 1),
            ]
        )
        y = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
        dev_x = train_x.clone()

        head = TinyMeanDR(3)
        with torch.no_grad():
            head.projection.weight.zero_()
            head.projection.bias.zero_()
            head.projection.weight[0, 0] = 1.0
            head.projection.weight[1, 1] = 1.0
            head.projection.weight[2, 2] = 1.0

        metrics = evaluate_embedding_space(
            mean_dr=head,
            train_reference_loader=_loader(train_x, y),
            dev_loader=_loader(dev_x, y),
            num_classes=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            metrics.generic_dev_score_name,
            GENERIC_DEV_SCORE_NAME,
        )
        self.assertAlmostEqual(metrics.prototype_macro_f1, 1.0)
        self.assertAlmostEqual(metrics.generic_dev_score, 1.0)
        self.assertFalse(metrics.collapsed)
        self.assertLess(
            metrics.embedding_norm_max_abs_error,
            1e-5,
        )

    def test_training_records_loss_dev_metrics_and_checkpoint_hash(self):
        # Synthetic clustered features.
        centers = torch.eye(3, 6) * 3.0
        xs = []
        ys = []
        for c in range(3):
            for _ in range(12):
                xs.append(centers[c] + 0.05 * torch.randn(6))
                ys.append(c)
        xs = torch.stack(xs)
        ys = torch.tensor(ys)

        train_loader = _loader(xs, ys, batch_size=9)
        train_ref = _loader(xs, ys, batch_size=12)
        dev_loader = _loader(xs.clone(), ys.clone(), batch_size=12)

        with tempfile.TemporaryDirectory() as td:
            history = run_training(
                mean_dr=TinyMeanDR(6),
                scaf=TinySCAF(3),
                train_loader=train_loader,
                train_reference_loader=train_ref,
                dev_loader=dev_loader,
                num_classes=3,
                seed=17,
                run_dir=Path(td) / "run",
                device=torch.device("cpu"),
                optimization=OptimizationConfig(
                    epochs=2,
                    learning_rate=1e-2,
                    weight_decay=0.0,
                    grad_clip_norm=5.0,
                ),
            )
            self.assertEqual(len(history), 2)
            for row in history:
                self.assertTrue(torch.isfinite(
                    torch.tensor(row["train_loss"])
                ))
                self.assertIn("generic_dev_score", row["dev"])
                self.assertEqual(len(row["checkpoint_sha256"]), 64)
                ckpt = Path(row["checkpoint"])
                self.assertTrue(ckpt.is_file())
                self.assertEqual(
                    checkpoint_sha256(ckpt),
                    row["checkpoint_sha256"],
                )

            self.assertTrue(
                (Path(td) / "run" / "metrics.jsonl").is_file()
            )
            self.assertTrue(
                (Path(td) / "run" / "checkpoint_hashes.jsonl").is_file()
            )

    def test_existing_run_dir_is_not_silently_overwritten(self):
        xs = torch.randn(6, 4)
        ys = torch.tensor([0, 0, 1, 1, 2, 2])
        loader = _loader(xs, ys, batch_size=3)

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            with self.assertRaises(FileExistsError):
                run_training(
                    mean_dr=TinyMeanDR(4),
                    scaf=TinySCAF(3),
                    train_loader=loader,
                    train_reference_loader=loader,
                    dev_loader=loader,
                    num_classes=3,
                    seed=17,
                    run_dir=run_dir,
                    device=torch.device("cpu"),
                    optimization=OptimizationConfig(epochs=1),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
