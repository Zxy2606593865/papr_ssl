from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P407StaticContractTest(unittest.TestCase):
    def test_resource_phase_never_changes_teacher_selection(self):
        source = Path(
            "scripts/profile_p4_07_resource_cost.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn(
            '"teacher_selection_changed_by_resource_cost": False',
            source,
        )
        self.assertIn(
            '"ranking_by_resource_cost_performed": False',
            source,
        )

    def test_benchmark_uses_full_standard_forward(self):
        source = Path(
            "scripts/profile_p4_07_resource_cost.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"standard_hf_full_forward": True', source)
        self.assertIn(
            '"early_exit_or_encoder_truncation_used": False',
            source,
        )

    def test_w2vbert_reports_frontend_in_e2e_latency(self):
        source = Path(
            "scripts/profile_p4_07_resource_cost.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"includes_cpu_feature_extractor": True',
            source,
        )

    def test_test_split_stays_sealed(self):
        source = Path(
            "scripts/profile_p4_07_resource_cost.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"generic_test": "sealed_not_accessed"',
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
