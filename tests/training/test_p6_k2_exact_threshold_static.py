from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P6K2ExactThresholdStaticTest(unittest.TestCase):
    def test_exact_candidates_use_unique_observed_values(self):
        src = Path(
            "scripts/run_p6_k2_exact_calibration.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("unique = np.unique(values)", src)
        self.assertIn("np.nextafter(unique[0], -np.inf)", src)
        self.assertIn("np.nextafter(unique[-1], np.inf)", src)

    def test_absolute_gates_unchanged(self):
        src = Path(
            "scripts/run_p6_k2_exact_calibration.py"
        ).read_text(encoding="utf-8")
        for token in (
            '"macro_f1_min": 0.80',
            '"correct_accept_min": 0.80',
            '"wrong_intent_max": 0.10',
            '"known_reject_max": 0.20',
            '"unknown_reject_min": 0.85',
        ):
            self.assertIn(token, src)

    def test_generic_test_is_sealed(self):
        src = Path(
            "scripts/export_p6_k2_dev_scores.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_thresholds_are_selected_on_cal_only(self):
        src = Path(
            "scripts/run_p6_k2_exact_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "calibrated = exact_calibrate(known_cal, unknown_cal)",
            src,
        )
        self.assertIn(
            "score_metrics = scalar_metrics(",
            src,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
