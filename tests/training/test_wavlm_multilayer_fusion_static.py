from pathlib import Path
import ast
import unittest


class MultiLayerFusionStaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/build_wavlm_layers13_17_cache.py",
            "scripts/run_wavlm_multilayer_fusion.py",
            "scripts/summarize_wavlm_multilayer_fusion.py",
            "scripts/audit_wavlm_multilayer_fusion.py",
        ]:
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_layers_frozen(self):
        src = Path("scripts/run_wavlm_multilayer_fusion.py").read_text(encoding="utf-8")
        self.assertIn("LAYERS = (13, 14, 15, 16, 17)", src)

    def test_modes_exist(self):
        src = Path("scripts/run_wavlm_multilayer_fusion.py").read_text(encoding="utf-8")
        self.assertIn('"single15"', src)
        self.assertIn('"weighted13_17"', src)

    def test_test_is_sealed(self):
        for rel in [
            "scripts/build_wavlm_layers13_17_cache.py",
            "scripts/run_wavlm_multilayer_fusion.py",
        ]:
            src = Path(rel).read_text(encoding="utf-8")
            self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_64d_preserved(self):
        src = Path("scripts/run_wavlm_multilayer_fusion.py").read_text(encoding="utf-8")
        self.assertIn('"canonical_64d_teacher": "preserved_not_modified"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
