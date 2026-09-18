from pathlib import Path
import ast
import unittest


class ClosedSetDTWStaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/run_closedset_dtw_eval.py",
            "scripts/audit_closedset_dtw_eval.py",
        ]:
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_no_unknown_rejector_threshold(self):
        src = Path("scripts/run_closedset_dtw_eval.py").read_text(encoding="utf-8")
        self.assertIn('"unknown_data_used": False', src)
        self.assertIn('"rejector_used": False', src)
        self.assertIn('"threshold_used": False', src)

    def test_test_sealed(self):
        src = Path("scripts/run_closedset_dtw_eval.py").read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_lambda_grid(self):
        src = Path("scripts/run_closedset_dtw_eval.py").read_text(encoding="utf-8")
        self.assertIn("LAMBDA_GRID = (0.50, 0.75, 1.00)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
