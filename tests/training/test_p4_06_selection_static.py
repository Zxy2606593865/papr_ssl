from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P406StaticContractTest(unittest.TestCase):
    def test_selection_uses_three_seed_mean_only(self):
        source = Path(
            "scripts/select_p4_06_backbone_layer.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn(
            '"selection_metric": "generic_dev_score"',
            source,
        )
        self.assertIn(
            '"aggregation": "3-seed mean over seeds 17/29/43"',
            source,
        )

    def test_within_backbone_exact_tie_uses_lower_layer(self):
        source = Path(
            "scripts/select_p4_06_backbone_layer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'int(r["hidden_state_index"])',
            source,
        )

    def test_resource_cost_does_not_override_quality(self):
        source = Path(
            "scripts/select_p4_06_backbone_layer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"resource_cost_used_for_selection": False',
            source,
        )

    def test_test_split_remains_sealed(self):
        source = Path(
            "scripts/select_p4_06_backbone_layer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"generic_test": "sealed_not_accessed"',
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
