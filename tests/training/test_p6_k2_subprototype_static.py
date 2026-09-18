from pathlib import Path
import ast, unittest
class T(unittest.TestCase):
    def test_runner(self):
        s=Path("scripts/run_p6_k2_subprototype.py").read_text(encoding="utf-8")
        ast.parse(s)
        self.assertIn('max(-1).values',s)
        self.assertIn('"generic_test":"sealed_not_accessed"',s)
if __name__=="__main__": unittest.main()
