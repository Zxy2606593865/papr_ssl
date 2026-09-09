from __future__ import annotations

import unittest
from pathlib import Path

from papr_ssl.experiments.layer_sweep import expand_layer_sweep, load_layer_sweep


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LayerSweepTest(unittest.TestCase):
    def test_real_config_expands_without_loading_models(self) -> None:
        candidates = load_layer_sweep(
            PROJECT_ROOT / "configs" / "teacher" / "layer_sweep.yaml"
        )
        self.assertEqual(len(candidates), 12)
        self.assertEqual(
            {candidate.backbone_kind for candidate in candidates},
            {"wav2vec2", "wavlm", "w2v_bert2"},
        )
        self.assertTrue(all(candidate.hidden_layer >= 0 for candidate in candidates))

    def test_duplicate_candidate_fails(self) -> None:
        payload = {
            "backbones": [
                {
                    "kind": "wavlm",
                    "base_config": "configs/teacher/wavlm_mean.yaml",
                    "hidden_layers": [12, 12],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            expand_layer_sweep(payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
