from pathlib import Path
import ast
import unittest


class H102StaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/run_h1_02_unseen_speaker_openset.py",
            "scripts/audit_h1_02_unseen_speaker_openset.py",
        ]:
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_dynamic_unknown(self):
        src = Path(
            "scripts/run_h1_02_unseen_speaker_openset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("REGISTERED_CLASSES = 20", src)
        self.assertIn("TARGET_UNKNOWN_FAR = 0.10", src)

    def test_loso_calibration(self):
        src = Path(
            "scripts/run_h1_02_unseen_speaker_openset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if s != heldout", src)

    def test_no_new_head(self):
        src = Path(
            "scripts/run_h1_02_unseen_speaker_openset.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"new_neural_head_added": False', src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
