from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P5FrameCacheStaticContractTest(unittest.TestCase):
    def test_exact_p4_winner_is_used(self):
        source = Path(
            "scripts/materialize_p5_wavlm15_frame_cache.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('MODEL_ID = "microsoft/wavlm-large"', source)
        self.assertIn("HIDDEN_STATE_INDEX = 15", source)
        self.assertIn("HIDDEN_DIM = 1024", source)

    def test_test_split_is_not_loaded(self):
        source = Path(
            "scripts/materialize_p5_wavlm15_frame_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn('splits=("train", "dev")', source)
        self.assertIn(
            '"generic_test": "sealed_not_accessed"',
            source,
        )

    def test_frame_cache_remains_float32(self):
        source = Path(
            "scripts/materialize_p5_wavlm15_frame_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"feature_dtype": "float32"', source)

    def test_native_output_mask_is_used(self):
        source = Path(
            "scripts/materialize_p5_wavlm15_frame_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "model._get_feature_vector_attention_mask(",
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
