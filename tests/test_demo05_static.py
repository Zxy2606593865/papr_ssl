from pathlib import Path
import ast
import unittest


class Demo05StaticTest(unittest.TestCase):
    def test_python_parse(self):
        for rel in (
            "demo/demo05_web/app.py",
            "scripts/prepare_demo05_presets.py",
            "scripts/run_demo05_web.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_ui_uses_canonical_text(self):
        s = Path("demo/demo05_web/app.py").read_text(encoding="utf-8")
        self.assertIn("canonical_text", s)
        self.assertIn("scores_are_calibrated_probabilities", s)

    def test_client_does_not_show_confidence(self):
        html = Path("demo/demo05_web/static/index.html").read_text(encoding="utf-8")
        self.assertNotIn("置信度", html)

    def test_real_runtime_used(self):
        s = Path("demo/demo05_web/app.py").read_text(encoding="utf-8")
        self.assertIn("RawWavFeatureAdapter", s)
        self.assertIn("PersonalizedRuntime", s)
        self.assertIn("wanghao_user_memory.pt", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
