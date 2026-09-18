from pathlib import Path
import ast
import unittest


class Demo04CStaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/analyze_demo04c_decision_features.py").read_text(
                encoding="utf-8"
            )
        )

    def test_no_fit(self):
        s = Path("scripts/analyze_demo04c_decision_features.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"threshold_fitting_performed": False', s)
        self.assertIn('"representation_changed": False', s)

    def test_generic_test_sealed(self):
        s = Path("scripts/analyze_demo04c_decision_features.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', s)

    def test_duration_diagnostic(self):
        s = Path("scripts/analyze_demo04c_decision_features.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("duration", s.lower())
        self.assertIn("logit_contribution", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
