from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P408StaticContractTest(unittest.TestCase):
    def test_exact_wavlm_mainline_is_frozen(self):
        source = Path(
            "scripts/freeze_p4_08_mainline.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn(
            'EXPECTED_MAINLINE_BACKBONE = "wavlm_large"',
            source,
        )
        self.assertIn("EXPECTED_MAINLINE_LAYER = 15", source)

    def test_other_backbones_are_ablation_only(self):
        source = Path(
            "scripts/freeze_p4_08_mainline.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"role": "ablation_baseline"', source)

    def test_p5_is_mean_vs_attention_only(self):
        source = Path(
            "scripts/freeze_p4_08_mainline.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"Mean DR"', source)
        self.assertIn('"Attention DR"', source)
        self.assertIn(
            '"frame_level_cache_required_for_attention": True',
            source,
        )

    def test_generic_test_remains_sealed(self):
        source = Path(
            "scripts/freeze_p4_08_mainline.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"generic_test": "sealed_not_accessed"',
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
