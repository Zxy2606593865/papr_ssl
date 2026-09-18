from pathlib import Path
import ast
import unittest


class P615ShotStabilityStaticTest(unittest.TestCase):
    def test_export_is_15shot_mean_prototype(self):
        src = Path(
            "scripts/export_p6_15shot_all_dev_scores.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("SHOT = 15", src)
        self.assertIn("build_mean_prototypes", src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_repeated_split_default_is_20(self):
        src = Path(
            "scripts/run_p6_15shot_split_stability.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn('"--num-splits", type=int, default=20', src)

    def test_known_split_is_stratified(self):
        src = Path(
            "scripts/run_p6_15shot_split_stability.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def stratified_known_split", src)

    def test_exact_calibration_is_repeated(self):
        src = Path(
            "scripts/run_p6_15shot_split_stability.py"
        ).read_text(encoding="utf-8")
        self.assertIn("calibrated = exact_calibrate(k_cal, u_cal)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
