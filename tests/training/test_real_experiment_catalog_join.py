from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.teacher.real_experiment import (
    RealTaskRow,
    build_mdsc_audio_catalog,
    load_real_task_rows,
    resolve_task_audio_path,
)


class TaskViewAudioCatalogJoinTest(unittest.TestCase):
    def test_task_view_does_not_need_path_field(self):
        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "core30.jsonl"
            index.write_text(
                json.dumps(
                    {
                        "dataset": "mdsc",
                        "utt_id": "CF0010_0024",
                        "split": "dev",
                        "task_label": "打开空调",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_real_task_rows(index, splits=("dev",))
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0].audio_relpath)

    def test_catalog_join_resolves_utt_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            wav = root / "Control" / "dev" / "wav" / "CF0010" / "CF0010_0024.wav"
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"x")

            catalog = build_mdsc_audio_catalog(root)
            row = RealTaskRow(
                dataset="mdsc",
                utt_id="CF0010_0024",
                split="dev",
                task_label="打开空调",
                audio_relpath=None,
            )
            resolved = resolve_task_audio_path(
                row=row,
                mdsc_root=root,
                audio_catalog=catalog,
            )
            self.assertEqual(resolved, wav.resolve())

    def test_duplicate_wav_stems_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            for group in ("Control", "Uncontrol"):
                wav = root / group / "train" / "wav" / "S" / "same.wav"
                wav.parent.mkdir(parents=True)
                wav.write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                build_mdsc_audio_catalog(root)

    def test_split_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mdsc"
            wav = root / "Control" / "train" / "wav" / "S" / "u1.wav"
            wav.parent.mkdir(parents=True)
            wav.write_bytes(b"x")
            catalog = build_mdsc_audio_catalog(root)
            row = RealTaskRow(
                dataset="mdsc",
                utt_id="u1",
                split="dev",
                task_label="A",
            )
            with self.assertRaises(RuntimeError):
                resolve_task_audio_path(
                    row=row,
                    mdsc_root=root,
                    audio_catalog=catalog,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
