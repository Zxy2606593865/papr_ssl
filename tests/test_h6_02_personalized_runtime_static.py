from pathlib import Path
import ast
import unittest


class H602StaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "papr_ssl/inference/h6_personalized_runtime.py",
            "scripts/run_h6_02_cached_runtime_demo.py",
            "scripts/audit_h6_02_runtime.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_three_state_runtime(self):
        src = Path(
            "papr_ssl/inference/h6_personalized_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"ACCEPT"', src)
        self.assertIn('"CONFIRM"', src)
        self.assertIn('"REJECT"', src)

    def test_reject_does_not_emit_intent(self):
        src = Path(
            "papr_ssl/inference/h6_personalized_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("intent_id = None", src)
        self.assertIn("canonical_text = None", src)

    def test_scores_not_probabilities(self):
        src = Path(
            "papr_ssl/inference/h6_personalized_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"scores_are_calibrated_probabilities": False', src)

    def test_demo_keeps_generic_test_sealed(self):
        src = Path(
            "scripts/run_h6_02_cached_runtime_demo.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
