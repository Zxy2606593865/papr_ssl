from pathlib import Path
import ast
import unittest


class TeacherDimSweepStaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/audit_teacher_dim_sweep.py",
            "scripts/run_teacher_dim_sweep.py",
            "scripts/summarize_teacher_dim_sweep.py",
        ]:
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_canonical_64d_is_preserved(self):
        src = Path("scripts/run_teacher_dim_sweep.py").read_text(encoding="utf-8")
        self.assertIn("PRESERVE_READ_ONLY", src)

    def test_test_is_sealed(self):
        src = Path("scripts/run_teacher_dim_sweep.py").read_text(encoding="utf-8")
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_frozen_dims(self):
        src = Path("scripts/run_teacher_dim_sweep.py").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_DIMS = (64, 128, 256, 512)", src)
        self.assertIn("DEFAULT_SEEDS = (17, 29, 43)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
