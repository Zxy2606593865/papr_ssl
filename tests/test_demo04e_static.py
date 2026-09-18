from pathlib import Path
import ast
import unittest


class Demo04EStaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/audit_demo04e_prefix_temporal.py").read_text(
                encoding="utf-8"
            )
        )

    def test_prefix_metrics_exist(self):
        s = Path("scripts/audit_demo04e_prefix_temporal.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "prefix_dtw_sim",
            "suffix_dtw_sim",
            "prefix_minus_suffix",
            "start_cosine",
            "end_cosine",
            "start_minus_end",
        ):
            self.assertIn(token, s)

    def test_no_fit_or_runtime_change(self):
        s = Path("scripts/audit_demo04e_prefix_temporal.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"threshold_fitting_performed": False', s)
        self.assertIn('"runtime_policy_changed": False', s)
        self.assertIn('"representation_changed": False', s)
        self.assertIn('"generic_test": "sealed_not_accessed"', s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
