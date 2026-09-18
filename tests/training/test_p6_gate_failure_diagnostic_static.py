from pathlib import Path
import ast
import unittest

class P6GateFailureDiagnosticStaticTest(unittest.TestCase):
    def test_exact_empirical_enumeration(self):
        src = Path("scripts/diagnose_p6_k2_exact_gate.py").read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn("u = np.unique", src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)
        self.assertIn("max_correct_accept_given_unknown_reject_ge_085", src)
        self.assertIn("max_unknown_reject_given_correct_accept_ge_080", src)

if __name__ == "__main__":
    unittest.main(verbosity=2)
