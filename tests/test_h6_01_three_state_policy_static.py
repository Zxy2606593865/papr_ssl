from pathlib import Path
import ast
import unittest


class H601StaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "scripts/run_h6_01_three_state_policy.py",
            "scripts/audit_h6_01_three_state_policy.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_three_states(self):
        src = Path("scripts/run_h6_01_three_state_policy.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"CONFIRM"', src)
        self.assertIn('"ACCEPT"', src)
        self.assertIn('"REJECT"', src)

    def test_constraints(self):
        src = Path("scripts/run_h6_01_three_state_policy.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("TARGET_UNKNOWN_ACCEPT = 0.10", src)
        self.assertIn("MAX_KNOWN_REJECT = 0.05", src)

    def test_test_sealed(self):
        src = Path("scripts/run_h6_01_three_state_policy.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
