from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.teacher.real_experiment import (
    parse_real_task_row,
    resolve_audio_path,
)


class RealExperimentPathHotfixTest(unittest.TestCase):
    def test_canonical_p2_path_field_is_accepted(self):
        row = parse_real_task_row(
            {
                "dataset": "mdsc",
                "utterance_id": "CF0010_0024",
                "split": "dev",
                "label": "打开空调",
                "path": "Control/dev/wav/CF0010/CF0010_0024.wav",
            }
        )
        self.assertEqual(
            row.audio_relpath,
            "Control/dev/wav/CF0010/CF0010_0024.wav",
        )

    def test_dataset_root_relative_path_resolves_inside_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            wav = root / "Control" / "dev" / "wav" / "u.wav"
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"x")

            resolved = resolve_audio_path(
                mdsc_root=root,
                audio_relpath="Control/dev/wav/u.wav",
            )
            self.assertEqual(resolved, wav.resolve())

    def test_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            root.mkdir()
            with self.assertRaises(ValueError):
                resolve_audio_path(
                    mdsc_root=root,
                    audio_relpath="../outside.wav",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
