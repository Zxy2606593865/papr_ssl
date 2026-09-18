from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P402StaticContractTest(unittest.TestCase):
    def test_materializer_covers_all_13_indices_and_reuses_complete_cache(self):
        source = Path(
            "scripts/materialize_p4_02_wav2vec2_all_layers.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('get_p4_backbone("wav2vec2_base")', source)
        self.assertIn('status[layer] == "missing"', source)
        self.assertIn('status[layer] == "complete"', source)

    def test_single_runner_freezes_budget(self):
        source = Path(
            "scripts/run_p4_02_wav2vec2_layer_seed.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("epochs=20", source)
        self.assertIn("learning_rate=1e-3", source)
        self.assertIn("k=3", source)
        self.assertIn("margin=0.2", source)
        self.assertIn("scale=30.0", source)

    def test_orchestrator_reuses_identical_p3_layer12_by_default(self):
        source = Path(
            "scripts/run_p4_02_wav2vec2_sweep.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("reuse_p3", source)
        self.assertIn("--include-layer12-rerun", source)

    def test_p402_does_not_select_best_layer(self):
        source = Path(
            "scripts/audit_p4_02_wav2vec2_sweep.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"best_layer_selected": False', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
