from pathlib import Path
import ast
import unittest


class H200StaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/audit_h2_enrollment_stats_readiness.py").read_text(
                encoding="utf-8"
            )
        )

    def test_support_levels(self):
        src = Path("scripts/audit_h2_enrollment_stats_readiness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SUPPORT_LEVELS = (1, 2, 3, 4, 5)", src)

    def test_variance_requires_three(self):
        src = Path("scripts/audit_h2_enrollment_stats_readiness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"class_specific_variance_informative": int(k >= 3)', src)

    def test_sealed_test(self):
        src = Path("scripts/audit_h2_enrollment_stats_readiness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
