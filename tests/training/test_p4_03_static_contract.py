from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P403StaticContractTest(unittest.TestCase):
    def test_materializer_uses_native_padding_attention_mask(self):
        source = Path(
            "scripts/materialize_p4_03_wavlm_all_layers.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("attention_mask=sample_mask.to(torch.long)", source)
        self.assertIn("_get_feature_vector_attention_mask", source)
        self.assertIn("EXPECTED_HIDDEN_STATE_COUNT = 25", source)

    def test_materializer_length_sorts_to_reduce_padding(self):
        source = Path(
            "scripts/materialize_p4_03_wavlm_all_layers.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "resolved.sort(key=lambda x: (x[0], x[1].utt_id))",
            source,
        )

    def test_single_runner_freezes_p3_budget(self):
        source = Path(
            "scripts/run_p4_03_wavlm_layer_seed.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("epochs=20", source)
        self.assertIn("learning_rate=1e-3", source)
        self.assertIn("k=3", source)
        self.assertIn("margin=0.2", source)
        self.assertIn("scale=30.0", source)

    def test_p403_does_not_select_best_layer(self):
        source = Path(
            "scripts/audit_p4_03_wavlm_sweep.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"best_layer_selected": False', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
