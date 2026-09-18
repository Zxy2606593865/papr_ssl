from pathlib import Path
import ast
import unittest


class H401StaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/run_h4_01_shared_evidence_ranker.py",
            "scripts/audit_h4_01_shared_evidence_ranker.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_is_first_trainable_head(self):
        src = Path(
            "scripts/run_h4_01_shared_evidence_ranker.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"new_trainable_head_added": True', src)
        self.assertIn('"backbone_neck_updated": False', src)

    def test_train_dev_separation(self):
        src = Path(
            "scripts/run_h4_01_shared_evidence_ranker.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"speakers": 34', src)
        self.assertIn('"speakers": 4', src)

    def test_test_sealed(self):
        src = Path(
            "scripts/run_h4_01_shared_evidence_ranker.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
