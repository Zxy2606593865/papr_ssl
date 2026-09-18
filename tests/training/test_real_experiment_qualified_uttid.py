from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.teacher.real_experiment import (
    RealTaskRow,
    qualified_utt_relpath,
    resolve_task_audio_path,
)


class QualifiedUttIdHotfixTest(unittest.TestCase):
    def test_current_mdsc_qualified_utt_id_decodes_relative_wav_path(self):
        row = RealTaskRow(
            dataset="mdsc",
            utt_id="mdsc:Control/dev/wav/CF0010/CF0010_0001.wav",
            split="dev",
            task_label="A",
        )
        self.assertEqual(
            qualified_utt_relpath(row),
            "Control/dev/wav/CF0010/CF0010_0001.wav",
        )

    def test_qualified_utt_id_resolves_without_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            wav = (
                root
                / "Control"
                / "dev"
                / "wav"
                / "CF0010"
                / "CF0010_0001.wav"
            )
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"x")

            row = RealTaskRow(
                dataset="mdsc",
                utt_id="mdsc:Control/dev/wav/CF0010/CF0010_0001.wav",
                split="dev",
                task_label="A",
            )
            resolved = resolve_task_audio_path(
                row=row,
                mdsc_root=root,
                audio_catalog=None,
            )
            self.assertEqual(resolved, wav.resolve())

    def test_wrong_dataset_prefix_is_not_treated_as_canonical(self):
        row = RealTaskRow(
            dataset="mdsc",
            utt_id="other:Control/dev/wav/u.wav",
            split="dev",
            task_label="A",
        )
        self.assertIsNone(qualified_utt_relpath(row))

    def test_qualified_id_split_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            wav = (
                root
                / "Control"
                / "train"
                / "wav"
                / "S"
                / "u1.wav"
            )
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"x")
            row = RealTaskRow(
                dataset="mdsc",
                utt_id="mdsc:Control/train/wav/S/u1.wav",
                split="dev",
                task_label="A",
            )
            with self.assertRaises(RuntimeError):
                resolve_task_audio_path(
                    row=row,
                    mdsc_root=root,
                    audio_catalog=None,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
