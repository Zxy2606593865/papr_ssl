from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.feature_data import _load_task_rows


class P2TaskViewCompatibilityTest(unittest.TestCase):
    def test_dataset_is_derived_from_qualified_utt_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "task.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "utt_id": (
                            "mdsc:Control/dev/wav/CF0010/"
                            "CF0010_0001.wav"
                        ),
                        "split": "dev",
                        "task_label": "打开空调",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _load_task_rows(path)
            self.assertEqual(rows[0].dataset, "mdsc")
            self.assertEqual(
                rows[0].utt_id,
                "mdsc:Control/dev/wav/CF0010/CF0010_0001.wav",
            )

    def test_label_alias_is_accepted_without_rewriting_value(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "task.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "utt_id": "mdsc:Uncontrol/train/wav/S/u1.wav",
                        "split": "train",
                        "label": "打开空调",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _load_task_rows(path)
            self.assertEqual(rows[0].task_label, "打开空调")
            self.assertIsNone(rows[0].sample_hash)

    def test_plain_utt_id_without_dataset_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "task.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "utt_id": "u1",
                        "split": "train",
                        "task_label": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_task_rows(path)

    def test_explicit_dataset_still_supported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "task.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "gsc",
                        "utt_id": "u1",
                        "split": "train",
                        "task_label": "yes",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _load_task_rows(path)
            self.assertEqual(rows[0].dataset, "gsc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
