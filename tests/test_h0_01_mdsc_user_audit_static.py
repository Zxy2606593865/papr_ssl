from pathlib import Path
import ast
import unittest


class H001StaticTest(unittest.TestCase):
    def test_script_parses(self):
        ast.parse(
            Path("scripts/audit_mdsc_user_personalization.py").read_text(
                encoding="utf-8"
            )
        )

    def test_shot_levels(self):
        src = Path("scripts/audit_mdsc_user_personalization.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SHOT_LEVELS = (1, 2, 5, 10, 15)", src)

    def test_train_support_dev_query(self):
        src = Path("scripts/audit_mdsc_user_personalization.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"support": "same speaker + same Core30 phrase, TRAIN only"', src)
        self.assertIn('"known_query": "same speaker + registered Core30 phrase, DEV only"', src)

    def test_no_audio_or_embedding_access(self):
        src = Path("scripts/audit_mdsc_user_personalization.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("soundfile", src)
        self.assertNotIn("torch.load", src)
        self.assertIn('"generic_test_audio_accessed": False', src)
        self.assertIn('"generic_test_embeddings_accessed": False', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
