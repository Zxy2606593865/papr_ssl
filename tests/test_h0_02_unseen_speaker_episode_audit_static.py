from pathlib import Path
import ast
import unittest


class H002StaticTest(unittest.TestCase):
    def test_parse(self):
        ast.parse(
            Path("scripts/audit_mdsc_unseen_speaker_episodes.py").read_text(
                encoding="utf-8"
            )
        )

    def test_shots_require_query(self):
        src = Path("scripts/audit_mdsc_unseen_speaker_episodes.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("count >= shot + 1", src)

    def test_test_sealed(self):
        src = Path("scripts/audit_mdsc_unseen_speaker_episodes.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"generic_test_accessed": False', src)

    def test_dev_support_is_allowed(self):
        src = Path("scripts/audit_mdsc_unseen_speaker_episodes.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("UNSEEN-SPEAKER DEV", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
