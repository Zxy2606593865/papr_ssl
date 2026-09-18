from pathlib import Path
import ast
import unittest


class H604StaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/run_h6_04_raw_wav_end_to_end.py",
            "scripts/audit_h6_04_raw_wav_end_to_end.py",
            "scripts/enroll_h6_user_from_wavs.py",
            "scripts/predict_h6_user_wav.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_raw_both_sides(self):
        src = Path(
            "scripts/run_h6_04_raw_wav_end_to_end.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"raw_wav_enrollment": True', src)
        self.assertIn('"raw_wav_query": True', src)

    def test_runtime_parity_gate(self):
        src = Path(
            "scripts/run_h6_04_raw_wav_end_to_end.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"all_predictions_match_cached_runtime"', src)

    def test_generic_test_sealed(self):
        src = Path(
            "scripts/run_h6_04_raw_wav_end_to_end.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
