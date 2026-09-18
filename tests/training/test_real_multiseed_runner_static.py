from __future__ import annotations

import ast
import unittest
from pathlib import Path


class RealMultiSeedRunnerStaticTest(unittest.TestCase):
    def test_runner_is_seed_parameterized_and_frozen_to_required_seeds(self):
        path = Path("scripts/run_p3_core30_wav2vec2_seed.py")
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("FROZEN_SEEDS = (17, 29, 43)", source)
        self.assertIn("--seed", source)
        self.assertIn("seed_everything(seed)", source)

    def test_runner_keeps_20_epoch_default_and_fixed_scaf(self):
        source = Path(
            "scripts/run_p3_core30_wav2vec2_seed.py"
        ).read_text(encoding="utf-8")
        self.assertIn('default=20', source)
        self.assertIn("k=3", source)
        self.assertIn("margin=0.2", source)
        self.assertIn("scale=30.0", source)

    def test_summary_requires_all_three_seeds(self):
        source = Path(
            "scripts/summarize_p3_core30_wav2vec2_multiseed.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("SEEDS = (17, 29, 43)", source)
        self.assertIn("generic_test", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
