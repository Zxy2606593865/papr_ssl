from pathlib import Path
import ast
import unittest


class P6UnknownExposureAuditStaticTest(unittest.TestCase):
    def test_audit_script_parses(self):
        src = Path(
            "scripts/audit_p6_unknown_exposure_candidates.py"
        ).read_text(encoding="utf-8")
        ast.parse(src)

    def test_only_train_candidates(self):
        src = Path(
            "scripts/audit_p6_unknown_exposure_candidates.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'train_rows = [r for r in full_rows if row_split(r) == "train"]',
            src,
        )

    def test_excludes_known_and_open_dev_phrases(self):
        src = Path(
            "scripts/audit_p6_unknown_exposure_candidates.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if phr in core30:", src)
        self.assertIn("if phr in open_dev_phrases:", src)

    def test_test_not_used_for_selection(self):
        src = Path(
            "scripts/audit_p6_unknown_exposure_candidates.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"generic_test_used_for_selection": False', src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
