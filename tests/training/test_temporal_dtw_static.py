from pathlib import Path
import ast
import unittest


class TemporalDTWStaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/build_temporal_dtw_features.py",
            "scripts/precompute_dtw_rerank.py",
            "scripts/run_temporal_dtw_ablation.py",
            "scripts/audit_temporal_dtw_ablation.py",
        ]:
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_no_second_encoder_contract(self):
        src = Path("scripts/build_temporal_dtw_features.py").read_text(encoding="utf-8")
        self.assertIn("no second encoder", src.lower())

    def test_low_cost_rerank(self):
        src = Path("scripts/precompute_dtw_rerank.py").read_text(encoding="utf-8")
        self.assertIn("--top-k", src)
        self.assertIn("default=3", src)
        self.assertIn("default=2", src)

    def test_test_sealed(self):
        for rel in [
            "scripts/build_temporal_dtw_features.py",
            "scripts/precompute_dtw_rerank.py",
            "scripts/run_temporal_dtw_ablation.py",
        ]:
            src = Path(rel).read_text(encoding="utf-8")
            self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_teacher_preserved(self):
        src = Path("scripts/run_temporal_dtw_ablation.py").read_text(encoding="utf-8")
        self.assertIn('"canonical_64d_teacher": "preserved_not_modified"', src)
        self.assertIn('"project1_256d_teacher": "preserved_not_modified"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
