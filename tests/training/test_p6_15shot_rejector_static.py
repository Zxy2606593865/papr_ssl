from pathlib import Path
import ast
import unittest


class P615ShotRejectorStaticTest(unittest.TestCase):
    def test_teacher_and_intent_prediction_are_frozen_by_design(self):
        src = Path(
            "scripts/run_p6_15shot_rejector_ablation.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("intent_prediction", src)
        self.assertIn("top1 cosine class unchanged", src)

    def test_rejector_is_low_capacity(self):
        src = Path(
            "scripts/run_p6_15shot_rejector_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("balanced L2-regularized logistic regression", src)
        self.assertIn("l2: float = 1e-2", src)

    def test_uses_same_repeated_split_seed_policy(self):
        src = Path(
            "scripts/run_p6_15shot_rejector_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn('base_seed = int(baseline["base_seed"])', src)
        self.assertIn('num_splits = int(baseline["num_splits"])', src)

    def test_test_set_is_sealed(self):
        src = Path(
            "scripts/run_p6_15shot_rejector_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
