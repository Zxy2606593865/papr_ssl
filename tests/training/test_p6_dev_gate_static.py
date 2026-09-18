from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P6DevGateStaticTest(unittest.TestCase):
    def test_protocol_selects_median_seed_not_best_seed(self):
        src = Path("scripts/prepare_p6_protocol.py").read_text(
            encoding="utf-8"
        )
        ast.parse(src)
        self.assertIn("return ordered[1]", src)
        self.assertIn("never best seed", src)

    def test_default_enrollment_is_k1(self):
        src = Path("scripts/prepare_p6_protocol.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--enrollment-k", type=int, default=1', src)

    def test_threshold_rule_uses_score_and_margin(self):
        src = Path("scripts/run_p6_dev_qualification.py").read_text(
            encoding="utf-8"
        )
        ast.parse(src)
        self.assertIn('known["score"] >= score_threshold', src)
        self.assertIn('known["margin"] >= margin_threshold', src)

    def test_absolute_gates_are_frozen(self):
        src = Path("scripts/run_p6_dev_qualification.py").read_text(
            encoding="utf-8"
        )
        for text in (
            '"macro_f1_min": 0.80',
            '"correct_accept_min": 0.80',
            '"wrong_intent_max": 0.10',
            '"known_reject_max": 0.20',
            '"unknown_reject_min": 0.85',
        ):
            self.assertIn(text, src)

    def test_e0b_drop_is_frozen(self):
        src = Path("scripts/run_p6_dev_qualification.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("E0B_MAX_DROP = 0.02", src)

    def test_test_is_sealed_before_freeze(self):
        src = Path("scripts/freeze_p6_teacher.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', src)
        self.assertIn('"test_may_tune_model_or_thresholds": False', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
