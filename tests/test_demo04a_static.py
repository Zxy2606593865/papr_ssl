from pathlib import Path
import ast
import unittest


class Demo04AStaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/run_demo04a_wanghao_runtime.py",
            "scripts/audit_demo04a_wanghao_runtime.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_uses_frozen_h6(self):
        s = Path("scripts/run_demo04a_wanghao_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("RawWavFeatureAdapter", s)
        self.assertIn("PersonalizedRuntime", s)
        self.assertIn("UserMemory", s)

    def test_no_threshold_fit(self):
        s = Path("scripts/run_demo04a_wanghao_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"training_or_threshold_fitting_performed": False', s)

    def test_generic_test_sealed(self):
        s = Path("scripts/run_demo04a_wanghao_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
