from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.teacher.real_experiment import (
    load_real_task_rows,
    parse_real_task_row,
    resolve_audio_path,
    summarize_real_rows,
)


class RealExperimentHelpersTest(unittest.TestCase):
    def test_aliases_parse(self):
        row = parse_real_task_row(
            {
                "dataset": "mdsc",
                "utterance_id": "u1",
                "split": "train",
                "task_label": "打开空调",
                "audio_path": "Control/train/wav/u1.wav",
            }
        )
        self.assertEqual(row.utt_id, "u1")
        self.assertEqual(row.audio_relpath, "Control/train/wav/u1.wav")

    def test_real_loader_keeps_train_dev_and_skips_test(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.jsonl"
            rows = [
                {
                    "dataset": "mdsc",
                    "utt_id": "tr",
                    "split": "train",
                    "task_label": "A",
                    "audio_relpath": "a.wav",
                },
                {
                    "dataset": "mdsc",
                    "utt_id": "dv",
                    "split": "dev",
                    "task_label": "B",
                    "audio_relpath": "b.wav",
                },
                {
                    "dataset": "mdsc",
                    "utt_id": "te",
                    "split": "test",
                    "task_label": "C",
                    "audio_relpath": "c.wav",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
                + "\n",
                encoding="utf-8",
            )
            out = load_real_task_rows(path)
            self.assertEqual([x.utt_id for x in out], ["tr", "dv"])
            summary = summarize_real_rows(out)
            self.assertEqual(summary["split_counts"], {"train": 1, "dev": 1})

    def test_test_only_request_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "mdsc",
                        "utt_id": "te",
                        "split": "test",
                        "task_label": "A",
                        "audio_relpath": "a.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_real_task_rows(path, splits=("test",))

    def test_audio_path_cannot_escape_root(self):
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
