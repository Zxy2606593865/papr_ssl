from pathlib import Path
import ast
import unittest


class Demo04BStaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/analyze_demo04a_failures.py").read_text(encoding="utf-8")
        )

    def test_no_fit(self):
        s = Path("scripts/analyze_demo04a_failures.py").read_text(encoding="utf-8")
        self.assertIn('"threshold_fitting_performed": False', s)
        self.assertIn('"training_performed": False', s)

    def test_generic_test_sealed(self):
        s = Path("scripts/analyze_demo04a_failures.py").read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
