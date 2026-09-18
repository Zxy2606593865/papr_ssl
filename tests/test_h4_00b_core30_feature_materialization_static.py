from pathlib import Path
import ast
import unittest


class H400BStaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/prepare_h4_core30_features.py",
            "scripts/audit_h4_core30_features.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_no_training(self):
        src = Path("scripts/prepare_h4_core30_features.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"trainable_parameters_updated": False', src)
        self.assertIn('"checkpoint_modified": False', src)

    def test_expected_counts(self):
        src = Path("scripts/prepare_h4_core30_features.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("len(train_ds) != 3756", src)
        self.assertIn("len(dev_ds) != 442", src)

    def test_test_sealed(self):
        src = Path("scripts/prepare_h4_core30_features.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
