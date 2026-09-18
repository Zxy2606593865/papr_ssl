from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P404StaticContractTest(unittest.TestCase):
    def test_materializer_uses_exact_semantic_waveforms_before_frontend(self):
        source = Path(
            "scripts/materialize_p4_04_w2vbert2_all_layers.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("raw_speech = [", source)
        self.assertIn("padding=True", source)
        self.assertIn("return_attention_mask=True", source)

    def test_materializer_uses_w2vbert_native_inputs(self):
        source = Path(
            "scripts/materialize_p4_04_w2vbert2_all_layers.py"
        ).read_text(encoding="utf-8")
        self.assertIn("input_features=input_features", source)
        self.assertIn(
            "attention_mask=frame_mask.to(torch.long)",
            source,
        )
        self.assertIn("EXPECTED_FRONTEND_DIM = 160", source)
        self.assertIn("EXPECTED_HIDDEN_STATE_COUNT = 25", source)

    def test_single_runner_freezes_p3_budget(self):
        source = Path(
            "scripts/run_p4_04_w2vbert2_layer_seed.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("epochs=20", source)
        self.assertIn("learning_rate=1e-3", source)
        self.assertIn("k=3", source)
        self.assertIn("margin=0.2", source)
        self.assertIn("scale=30.0", source)

    def test_p404_does_not_select_best_layer(self):
        source = Path(
            "scripts/audit_p4_04_w2vbert2_sweep.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"best_layer_selected": False', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
