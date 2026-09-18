from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P405StaticContractTest(unittest.TestCase):
    def test_aggregate_uses_exact_three_frozen_seeds(self):
        source = Path(
            "scripts/aggregate_p4_05_statistics.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("SEEDS = (17, 29, 43)", source)

    def test_aggregate_uses_sample_std_and_student_t_df2(self):
        source = Path(
            "scripts/aggregate_p4_05_statistics.py"
        ).read_text(encoding="utf-8")
        self.assertIn("stdev(values)", source)
        self.assertIn(
            "T_CRIT_975_DF2 = 4.302652729911275",
            source,
        )

    def test_p405_does_not_rank_or_select(self):
        source = Path(
            "scripts/aggregate_p4_05_statistics.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"best_layer_selected": False', source)
        self.assertIn('"best_backbone_selected": False', source)

    def test_all_three_raw_sources_are_required(self):
        source = Path(
            "scripts/aggregate_p4_05_statistics.py"
        ).read_text(encoding="utf-8")
        self.assertIn("p4_02_raw_results.json", source)
        self.assertIn("p4_03_raw_results.json", source)
        self.assertIn("p4_04_raw_results.json", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
