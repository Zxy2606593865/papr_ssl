from pathlib import Path
import ast
import unittest


class P61015ShotStaticTest(unittest.TestCase):
    def test_only_shot_count_changes(self):
        src = Path(
            "scripts/run_p6_10_15_shot_ablation.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("SHOTS = (10, 15)", src)
        self.assertIn("build_mean_prototypes", src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_nested_enrollment(self):
        src = Path(
            "scripts/run_p6_10_15_shot_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ids10.issubset(ids15)", src)

    def test_exact_threshold_search_is_kept(self):
        src = Path(
            "scripts/run_p6_10_15_shot_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("np.unique", src)
        self.assertIn("np.nextafter", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
