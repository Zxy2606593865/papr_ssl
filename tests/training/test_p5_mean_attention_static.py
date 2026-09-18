from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P5MeanAttentionStaticTest(unittest.TestCase):
    def test_heads_are_exactly_mean_and_attention(self):
        src = Path(
            "papr_ssl/training/teacher/p5_dr.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("class MeanDR", src)
        self.assertIn("class AttentionDR", src)
        self.assertIn("torch.softmax", src)
        self.assertIn("F.normalize", src)

    def test_frozen_training_budget(self):
        src = Path("scripts/train_p5_dr.py").read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("for epoch in range(1, 21)", src)
        self.assertIn("lr=1e-3", src)
        self.assertIn("weight_decay=1e-4", src)
        self.assertIn("classes_per_batch=8", src)
        self.assertIn("samples_per_class=4", src)

    def test_p5_uses_three_frozen_seeds(self):
        src = Path("scripts/train_p5_dr.py").read_text(encoding="utf-8")
        self.assertIn("SEEDS = (17, 29, 43)", src)

    def test_generic_test_is_sealed(self):
        src = Path("scripts/train_p5_dr.py").read_text(encoding="utf-8")
        self.assertIn('"generic_test_accessed": False', src)

    def test_mean_reproduction_gate_exists(self):
        src = Path(
            "scripts/aggregate_p5_mean_vs_attention.py"
        ).read_text(encoding="utf-8")
        self.assertIn("P4_MEAN_REFERENCE = 0.8864723619749779", src)
        self.assertIn("mean_abs_diff_vs_p4 <= 0.02", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
