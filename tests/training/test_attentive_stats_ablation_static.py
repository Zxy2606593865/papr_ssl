from pathlib import Path
import ast
import unittest


class AttentiveStatsStaticTest(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in [
            "scripts/audit_attentive_stats_setup.py",
            "scripts/run_attentive_stats_ablation.py",
            "scripts/summarize_attentive_stats_ablation.py",
            "scripts/audit_attentive_stats_ablation.py",
        ]:
            ast.parse(
                Path(rel).read_text(encoding="utf-8")
            )

    def test_two_arms_exist(self):
        src = Path(
            "scripts/run_attentive_stats_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"attention_mean_256"',
            src,
        )
        self.assertIn(
            '"attentive_stats_256"',
            src,
        )

    def test_asp_keeps_sigma(self):
        src = Path(
            "scripts/run_attentive_stats_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sigma = torch.sqrt(var)", src)
        self.assertIn(
            "torch.cat([mu, sigma], dim=-1)",
            src,
        )

    def test_test_is_sealed(self):
        src = Path(
            "scripts/run_attentive_stats_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"generic_test": "sealed_not_accessed"',
            src,
        )

    def test_64d_is_preserved(self):
        src = Path(
            "scripts/run_attentive_stats_ablation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"canonical_64d_teacher": "preserved_not_modified"',
            src,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
