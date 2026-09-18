from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from papr_ssl.training.teacher.multiseed import (
    MultiSeedConfig,
    SeedRunSpec,
    aggregate_scalar,
    run_multiseed_experiment,
    summarize_final_epoch,
)
from papr_ssl.training.teacher.train import OptimizationConfig


class MultiSeedConfigTest(unittest.TestCase):
    def test_required_17_29_43_are_enforced(self):
        MultiSeedConfig((17, 29, 43)).validate()
        MultiSeedConfig((17, 29, 43, 101)).validate()

        with self.assertRaises(ValueError):
            MultiSeedConfig((17, 29, 41)).validate()

    def test_duplicate_seed_fails(self):
        with self.assertRaises(ValueError):
            MultiSeedConfig((17, 29, 43, 43)).validate()

    def test_aggregate_uses_sample_std(self):
        agg = aggregate_scalar({17: 1.0, 29: 2.0, 43: 3.0})
        self.assertAlmostEqual(agg["mean"], 2.0)
        self.assertAlmostEqual(agg["sample_std"], 1.0)
        self.assertEqual(agg["std_ddof"], 1)
        self.assertEqual(agg["n"], 3)

    def test_final_epoch_summary_refuses_unequal_training_budgets(self):
        histories = {
            17: [
                {"epoch": 1, "train_loss": 1.0,
                 "dev": {"generic_dev_score": 0.1}},
                {"epoch": 2, "train_loss": 0.8,
                 "dev": {"generic_dev_score": 0.2}},
            ],
            29: [
                {"epoch": 1, "train_loss": 1.1,
                 "dev": {"generic_dev_score": 0.1}},
            ],
            43: [
                {"epoch": 1, "train_loss": 1.2,
                 "dev": {"generic_dev_score": 0.1}},
                {"epoch": 2, "train_loss": 0.9,
                 "dev": {"generic_dev_score": 0.2}},
            ],
        }
        with self.assertRaises(ValueError):
            summarize_final_epoch(histories)


class SeedBeforeFactoryTest(unittest.TestCase):
    def test_seed_is_set_before_factory_construction(self):
        # Patch the expensive training call so this test isolates ordering.
        import papr_ssl.training.teacher.multiseed as ms

        fingerprints = {}
        original = ms.run_training

        def fake_run_training(**kwargs):
            seed = kwargs["seed"]
            weight = kwargs["mean_dr"].weight.detach().clone()
            fingerprints[seed] = weight
            return [
                {
                    "epoch": 1,
                    "train_loss": float(seed) / 100.0,
                    "dev": {"generic_dev_score": float(seed) / 1000.0},
                    "checkpoint": f"seed_{seed}.pt",
                    "checkpoint_sha256": f"{seed:064x}"[-64:],
                }
            ]

        def factory(seed, run_dir):
            # nn.Linear initialization must already see the requested seed.
            return SeedRunSpec(
                mean_dr=nn.Linear(4, 4, bias=False),
                scaf=nn.Linear(4, 4, bias=False),
                train_loader=[],
                train_reference_loader=[],
                dev_loader=[],
                num_classes=3,
                device=torch.device("cpu"),
                optimization=OptimizationConfig(epochs=1),
            )

        ms.run_training = fake_run_training
        try:
            with tempfile.TemporaryDirectory() as td:
                run_multiseed_experiment(
                    experiment_dir=Path(td) / "exp",
                    build_seed_run=factory,
                    config=MultiSeedConfig((17, 29, 43)),
                )
        finally:
            ms.run_training = original

        self.assertFalse(torch.equal(
            fingerprints[17],
            fingerprints[29],
        ))
        self.assertFalse(torch.equal(
            fingerprints[29],
            fingerprints[43],
        ))

        # Reconstruct seed 17 independently: must match exactly.
        torch.manual_seed(17)
        expected_17 = nn.Linear(4, 4, bias=False).weight.detach()
        torch.testing.assert_close(
            fingerprints[17],
            expected_17,
            rtol=0.0,
            atol=0.0,
        )

    def test_summary_has_one_immutable_run_per_seed_and_no_selection(self):
        import papr_ssl.training.teacher.multiseed as ms

        original = ms.run_training

        def fake_run_training(**kwargs):
            seed = kwargs["seed"]
            run_dir = kwargs["run_dir"]
            return [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "dev": {"generic_dev_score": seed / 100.0},
                    "checkpoint": (
                        Path(run_dir) / "checkpoints" / "epoch_0001.pt"
                    ).as_posix(),
                    "checkpoint_sha256": f"{seed:064x}"[-64:],
                }
            ]

        def factory(seed, run_dir):
            return SeedRunSpec(
                mean_dr=nn.Linear(2, 2),
                scaf=nn.Linear(2, 2),
                train_loader=[],
                train_reference_loader=[],
                dev_loader=[],
                num_classes=3,
                device=torch.device("cpu"),
                optimization=OptimizationConfig(epochs=1),
            )

        ms.run_training = fake_run_training
        try:
            with tempfile.TemporaryDirectory() as td:
                exp = Path(td) / "exp"
                summary = run_multiseed_experiment(
                    experiment_dir=exp,
                    build_seed_run=factory,
                    config=MultiSeedConfig((17, 29, 43)),
                )
                self.assertTrue(
                    (exp / "multiseed_summary.json").is_file()
                )
        finally:
            ms.run_training = original

        self.assertEqual(
            [row["seed"] for row in summary["runs"]],
            [17, 29, 43],
        )
        self.assertIn(
            "deferred to P3-07",
            summary["selection_policy"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
