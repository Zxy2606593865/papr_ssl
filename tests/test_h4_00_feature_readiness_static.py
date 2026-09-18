from pathlib import Path
import ast
import unittest


class H400StaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/audit_h4_feature_readiness.py").read_text(
                encoding="utf-8"
            )
        )

    def test_sealed(self):
        src = Path("scripts/audit_h4_feature_readiness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_checks_both_modalities(self):
        src = Path("scripts/audit_h4_feature_readiness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("global_train_ready", src)
        self.assertIn("temporal_train_ready", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
