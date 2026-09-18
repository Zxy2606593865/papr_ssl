from pathlib import Path
import ast
import unittest


class P615ShotConsistencyRejectorStaticTest(unittest.TestCase):
    def test_export_seals_generic_test(self):
        src = Path(
            "scripts/export_p6_15shot_consistency_embeddings.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)
        self.assertIn("SHOT = 15", src)

    def test_uses_full_enrollment_distribution(self):
        src = Path(
            "scripts/run_p6_15shot_consistency_rejector.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("Similarity to all 15 enrollment utterances", src)
        self.assertIn("top3mean", src)
        self.assertIn("bottom3mean", src)

    def test_same_split_policy_as_baseline(self):
        src = Path(
            "scripts/run_p6_15shot_consistency_rejector.py"
        ).read_text(encoding="utf-8")
        self.assertIn('num_splits = int(baseline["num_splits"])', src)
        self.assertIn('base_seed = int(baseline["base_seed"])', src)

    def test_intent_prediction_unchanged(self):
        src = Path(
            "scripts/run_p6_15shot_consistency_rejector.py"
        ).read_text(encoding="utf-8")
        self.assertIn("mean-prototype top1 cosine unchanged", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
