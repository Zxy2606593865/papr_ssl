from pathlib import Path
import ast
import unittest


class Demo04FStaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/run_demo04f_expanded_unknown.py").read_text(
                encoding="utf-8"
            )
        )

    def test_all_confirmed_unregistered(self):
        s = Path("scripts/run_demo04f_expanded_unknown.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('label_status") != "CONFIRMED"', s)
        self.assertIn("registered_set", s)

    def test_no_fit(self):
        s = Path("scripts/run_demo04f_expanded_unknown.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"threshold_fitting_performed": False', s)
        self.assertIn('"runtime_policy_changed": False', s)
        self.assertIn('"generic_test": "sealed_not_accessed"', s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
