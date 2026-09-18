from pathlib import Path
import ast
import unittest


class H501StaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/run_h5_01_cwu_decision_head.py",
            "scripts/audit_h5_01_cwu_decision_head.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_protocol(self):
        src = Path("scripts/run_h5_01_cwu_decision_head.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("REGISTERED_CLASSES = 20", src)
        self.assertIn("TARGET_UNKNOWN_FAR = 0.10", src)
        self.assertIn('EVENT_TO_INDEX = {"C": 0, "W": 1, "U": 2}', src)

    def test_train_cal_dev_split(self):
        src = Path("scripts/run_h5_01_cwu_decision_head.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("cal_count=6", src)
        self.assertIn("Expected 28 fit / 6 calibration TRAIN speakers", src)

    def test_h4_ranker_not_used(self):
        src = Path("scripts/run_h5_01_cwu_decision_head.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"learned_h4_ranker_used": False', src)

    def test_test_sealed(self):
        src = Path("scripts/run_h5_01_cwu_decision_head.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
