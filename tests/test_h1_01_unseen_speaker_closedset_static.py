from pathlib import Path
import ast
import unittest


class H101StaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/precompute_h1_01_user_dtw.py",
            "scripts/run_h1_01_unseen_speaker_closedset.py",
            "scripts/audit_h1_01_unseen_speaker_closedset.py",
        ]:
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_shots(self):
        src = Path(
            "scripts/run_h1_01_unseen_speaker_closedset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("SHOT_LEVELS = (1, 2)", src)

    def test_no_rejector(self):
        src = Path(
            "scripts/run_h1_01_unseen_speaker_closedset.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"rejector_used": False', src)
        self.assertIn('"unknown_data_used": False', src)

    def test_test_sealed(self):
        src = Path(
            "scripts/run_h1_01_unseen_speaker_closedset.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_fixed_lambda(self):
        src = Path(
            "scripts/run_h1_01_unseen_speaker_closedset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("FUSION_LAMBDA = 0.50", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
