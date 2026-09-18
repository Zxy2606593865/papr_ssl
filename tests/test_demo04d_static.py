from pathlib import Path
import ast
import unittest


class Demo04DStaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/audit_demo04d_signed_duration.py").read_text(
                encoding="utf-8"
            )
        )

    def test_signed_not_absolute_only(self):
        s = Path("scripts/audit_demo04d_signed_duration.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("signed_duration_log_ratio", s)
        self.assertIn("query_support_ratio", s)
        self.assertIn("math.log", s)

    def test_no_fit_or_policy_change(self):
        s = Path("scripts/audit_demo04d_signed_duration.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"threshold_fitting_performed": False', s)
        self.assertIn('"runtime_policy_changed": False', s)
        self.assertIn('"generic_test": "sealed_not_accessed"', s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
