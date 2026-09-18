from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from papr_ssl.training.teacher.select import (
    GENERIC_DEV_SCORE_NAME,
    select_checkpoint_from_rows,
    select_run_checkpoint,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointSelectionTest(unittest.TestCase):
    def test_selects_max_generic_dev_score_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for epoch, score, accuracy in [
                (1, 0.70, 0.99),
                (2, 0.80, 0.80),
                (3, 0.75, 1.00),
            ]:
                ckpt = root / f"e{epoch}.pt"
                ckpt.write_bytes(f"epoch{epoch}".encode())
                rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": 1.0 / epoch,
                        "dev": {
                            "generic_dev_score": score,
                            "generic_dev_score_name": (
                                GENERIC_DEV_SCORE_NAME
                            ),
                            "prototype_accuracy": accuracy,
                        },
                        "checkpoint": ckpt.as_posix(),
                        "checkpoint_sha256": _sha(ckpt),
                    }
                )

            selected = select_checkpoint_from_rows(rows)
            self.assertEqual(selected.epoch, 2)
            self.assertAlmostEqual(selected.generic_dev_score, 0.80)

    def test_exact_tie_uses_earliest_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for epoch in [2, 5]:
                ckpt = root / f"e{epoch}.pt"
                ckpt.write_bytes(f"epoch{epoch}".encode())
                rows.append(
                    {
                        "epoch": epoch,
                        "dev": {
                            "generic_dev_score": 0.8,
                            "generic_dev_score_name": (
                                GENERIC_DEV_SCORE_NAME
                            ),
                        },
                        "checkpoint": ckpt.as_posix(),
                        "checkpoint_sha256": _sha(ckpt),
                    }
                )
            selected = select_checkpoint_from_rows(rows)
            self.assertEqual(selected.epoch, 2)

    def test_test_related_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "e1.pt"
            ckpt.write_bytes(b"x")
            row = {
                "epoch": 1,
                "dev": {
                    "generic_dev_score": 0.8,
                    "generic_dev_score_name": GENERIC_DEV_SCORE_NAME,
                },
                "generic_test_score": 0.99,
                "checkpoint": ckpt.as_posix(),
                "checkpoint_sha256": _sha(ckpt),
            }
            with self.assertRaises(RuntimeError):
                select_checkpoint_from_rows([row])

    def test_wrong_metric_definition_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "e1.pt"
            ckpt.write_bytes(b"x")
            row = {
                "epoch": 1,
                "dev": {
                    "generic_dev_score": 0.8,
                    "generic_dev_score_name": "prototype_accuracy",
                },
                "checkpoint": ckpt.as_posix(),
                "checkpoint_sha256": _sha(ckpt),
            }
            with self.assertRaises(ValueError):
                select_checkpoint_from_rows([row])

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "e1.pt"
            ckpt.write_bytes(b"x")
            row = {
                "epoch": 1,
                "dev": {
                    "generic_dev_score": 0.8,
                    "generic_dev_score_name": GENERIC_DEV_SCORE_NAME,
                },
                "checkpoint": ckpt.as_posix(),
                "checkpoint_sha256": "0" * 64,
            }
            with self.assertRaises(RuntimeError):
                select_checkpoint_from_rows([row])

    def test_run_selection_writes_pointer_not_best_checkpoint_copy(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "seed_0017"
            run.mkdir()
            ckpt_dir = run / "checkpoints"
            ckpt_dir.mkdir()
            ckpt = ckpt_dir / "epoch_0001.pt"
            ckpt.write_bytes(b"checkpoint")

            row = {
                "epoch": 1,
                "train_loss": 0.1,
                "dev": {
                    "generic_dev_score": 0.8,
                    "generic_dev_score_name": GENERIC_DEV_SCORE_NAME,
                },
                "checkpoint": ckpt.as_posix(),
                "checkpoint_sha256": _sha(ckpt),
            }
            (run / "metrics.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

            payload = select_run_checkpoint(run)
            self.assertTrue(
                (run / "selected_checkpoint.json").is_file()
            )
            self.assertEqual(payload["selected"]["epoch"], 1)
            self.assertFalse((run / "best.pt").exists())
            self.assertFalse(
                payload["selection_policy"]["generic_test_accessed"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
