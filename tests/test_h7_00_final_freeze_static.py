from pathlib import Path
import ast
import unittest


class H700FreezeStaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/run_h7_00_final_freeze.py",
            "scripts/audit_h7_00_final_freeze.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_generic_test_not_accessed(self):
        src = Path("scripts/run_h7_00_final_freeze.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "SEALED_NOT_ACCESSED"', src)
        self.assertNotIn("generic_test.wav", src)

    def test_fixed_decisions(self):
        src = Path("scripts/run_h7_00_final_freeze.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"fixed_fusion_lambda": 0.50', src)
        self.assertIn('"h4_learned_shared_ranker_promoted": False', src)
        self.assertIn(
            '"scores_are_calibrated_probabilities": False',
            src,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
