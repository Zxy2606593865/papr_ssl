#!/usr/bin/env python
"""Standalone P3-07 dev-only checkpoint-selection smoke test."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from papr_ssl.training.teacher.evaluate import GENERIC_DEV_SCORE_NAME
from papr_ssl.training.teacher.select import select_run_checkpoint


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "seed_0017"
        (run / "checkpoints").mkdir(parents=True)

        # Deliberately make accuracy disagree with generic_dev_score so the
        # contract proves that only generic_dev_score controls selection.
        rows = []
        for epoch, dev_score, accuracy in [
            (1, 0.71, 0.99),
            (2, 0.84, 0.90),
            (3, 0.84, 0.95),  # exact score tie: earliest epoch must win
            (4, 0.80, 1.00),
        ]:
            ckpt = run / "checkpoints" / f"epoch_{epoch:04d}.pt"
            ckpt.write_bytes(
                f"p3-07-checkpoint-{epoch}".encode("utf-8")
            )
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "dev": {
                        "generic_dev_score": dev_score,
                        "generic_dev_score_name": GENERIC_DEV_SCORE_NAME,
                        "prototype_accuracy": accuracy,
                    },
                    "checkpoint": ckpt.as_posix(),
                    "checkpoint_sha256": sha(ckpt),
                }
            )

        with (run / "metrics.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        payload = select_run_checkpoint(run)
        selected = payload["selected"]

        assert selected["epoch"] == 2
        assert selected["generic_dev_score"] == 0.84
        assert payload["selection_policy"]["metric"] == "generic_dev_score"
        assert payload["selection_policy"]["generic_test_accessed"] is False
        assert not (run / "best.pt").exists()

        print("=" * 96)
        print("PAPR-SSL P3-07 DEV-ONLY CHECKPOINT SELECTION")
        print("=" * 96)
        print("selection metric:               generic_dev_score")
        print(
            "metric definition:              "
            f"{payload['selection_policy']['metric_definition']}"
        )
        print("direction:                      maximize")
        print("tie break:                      earliest_epoch")
        print(f"selected epoch:                 {selected['epoch']}")
        print(
            "selected generic_dev_score:     "
            f"{selected['generic_dev_score']:.6f}"
        )
        print(
            "selected checkpoint sha256:     "
            f"{selected['checkpoint_sha256'][:16]}..."
        )
        print("prototype_accuracy used to rank:NO")
        print("generic_test accessed:          NO")
        print("best.pt copied:                 NO")
        print("-" * 96)
        print("P3-07 STATUS: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
