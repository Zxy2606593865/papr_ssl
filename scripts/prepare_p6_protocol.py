#!/usr/bin/env python
"""P6-01/02 protocol preparation.

Creates:
- one frozen representative checkpoint per P5 method (median seed score;
  never best-of-3 cherry-picking)
- generic_enrollment from Core30 train, default K=1 per intent
- disjoint generic_dev_cal / generic_dev_score partitions from Core30 dev
- disjoint unknown cal / score partitions from open-set dev

No generic_test row is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset


SEEDS = (17, 29, 43)
METHODS = ("mean", "attention")
SALT = "papr_ssl_p6_protocol_v1"


def stable_key(*parts: str) -> str:
    h = hashlib.sha256()
    h.update(SALT.encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_representative_checkpoint(run_root: Path, method: str) -> dict:
    rows = []
    for seed in SEEDS:
        path = run_root / method / f"seed_{seed:04d}" / "p5_result.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("generic_test_accessed") is not False:
            raise RuntimeError(f"{method} seed={seed}: test access violation")
        rows.append(
            {
                "seed": seed,
                "score": float(obj["selected_generic_dev_score"]),
                "selected_epoch": int(obj["selected_epoch"]),
                "checkpoint": obj["selected_checkpoint"],
                "checkpoint_sha256": obj["selected_checkpoint_sha256"],
            }
        )

    # Robust representative choice: median of the three selected dev scores.
    # This is deliberately NOT max(seed score).
    ordered = sorted(rows, key=lambda x: (x["score"], x["seed"]))
    return ordered[1]


def label_groups(dataset: FrameCacheDataset) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for row in dataset.rows:
        groups.setdefault(row["label_text"], []).append(row["utt_id"])
    return groups


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--open-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/"
            "mdsc_core30_open_set_eval.index.jsonl"
        ),
    )
    p.add_argument(
        "--p5-run-root",
        type=Path,
        default=Path("artifacts/p5_runs/mean_vs_attention"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_dev/protocol"),
    )
    p.add_argument("--enrollment-k", type=int, default=1)
    args = p.parse_args()

    if args.enrollment_k not in (1, 2):
        raise ValueError("P6 supports enrollment K=1 primary or K=2 secondary")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"protocol already exists; refusing overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )
    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError("Core30 train/dev contract changed")
    if len(train_ds.class_to_index) != 30:
        raise RuntimeError("Core30 class count changed")

    representatives = {
        method: select_representative_checkpoint(args.p5_run_root, method)
        for method in METHODS
    }

    train_groups = label_groups(train_ds)
    dev_groups = label_groups(dev_ds)

    enrollment = []
    known_cal = []
    known_score = []

    for label in sorted(train_groups):
        train_ids = sorted(
            train_groups[label],
            key=lambda utt: stable_key("enrollment", label, utt),
        )
        if len(train_ids) < args.enrollment_k:
            raise RuntimeError(f"not enough train samples for {label}")
        for utt_id in train_ids[: args.enrollment_k]:
            enrollment.append(
                {
                    "utt_id": utt_id,
                    "label_text": label,
                    "label_index": train_ds.class_to_index[label],
                    "source_split": "train",
                }
            )

        dev_ids = sorted(
            dev_groups[label],
            key=lambda utt: stable_key("known_dev", label, utt),
        )
        if len(dev_ids) < 2:
            raise RuntimeError(
                f"need at least 2 dev examples for disjoint cal/score: {label}"
            )
        n_cal = len(dev_ids) // 2
        cal_ids = dev_ids[:n_cal]
        score_ids = dev_ids[n_cal:]
        for utt_id in cal_ids:
            known_cal.append(
                {
                    "utt_id": utt_id,
                    "label_text": label,
                    "label_index": train_ds.class_to_index[label],
                    "source_split": "dev",
                }
            )
        for utt_id in score_ids:
            known_score.append(
                {
                    "utt_id": utt_id,
                    "label_text": label,
                    "label_index": train_ds.class_to_index[label],
                    "source_split": "dev",
                }
            )

    open_rows_all = read_jsonl(args.open_index)
    open_dev = [r for r in open_rows_all if r.get("split") == "dev"]
    if len(open_dev) != 1178:
        raise RuntimeError(
            f"expected 1178 open-set dev rows, got {len(open_dev)}"
        )
    if any(r.get("split") == "test" for r in open_dev):
        raise RuntimeError("test row leaked into open dev")

    open_dev = sorted(
        open_dev,
        key=lambda r: stable_key("unknown_dev", r["utt_id"]),
    )
    n_open_cal = len(open_dev) // 2
    unknown_cal = [
        {"utt_id": r["utt_id"], "source_split": "dev"}
        for r in open_dev[:n_open_cal]
    ]
    unknown_score = [
        {"utt_id": r["utt_id"], "source_split": "dev"}
        for r in open_dev[n_open_cal:]
    ]

    write_jsonl(args.output_dir / "generic_enrollment.jsonl", enrollment)
    write_jsonl(args.output_dir / "generic_dev_cal_known.jsonl", known_cal)
    write_jsonl(args.output_dir / "generic_dev_score_known.jsonl", known_score)
    write_jsonl(args.output_dir / "generic_dev_cal_unknown.jsonl", unknown_cal)
    write_jsonl(args.output_dir / "generic_dev_score_unknown.jsonl", unknown_score)

    protocol = {
        "schema": "papr_ssl.p6_generic_protocol.v1",
        "phase": "P6",
        "p6_01": {
            "prototype_policy": "mean_of_enrollment_embeddings_then_l2_norm",
            "enrollment_k": args.enrollment_k,
            "enrollment_selection": "stable_hash_no_embedding_cherry_pick",
            "enrollment_count": len(enrollment),
        },
        "p6_02": {
            "threshold_calibration_split": "generic_dev_cal",
            "thresholds": ["score_threshold", "margin_threshold"],
            "threshold_policy": (
                "cal-only exhaustive quantile-grid; satisfy absolute gates "
                "if feasible; lexicographic maximize macro_f1, correct_accept, "
                "unknown_reject"
            ),
        },
        "p6_03": {
            "score_split": "generic_dev_score",
            "known_score_count": len(known_score),
            "unknown_score_count": len(unknown_score),
        },
        "known_dev_counts": {
            "cal": len(known_cal),
            "score": len(known_score),
            "total": len(known_cal) + len(known_score),
        },
        "unknown_dev_counts": {
            "cal": len(unknown_cal),
            "score": len(unknown_score),
            "total": len(unknown_cal) + len(unknown_score),
        },
        "checkpoint_selection_rule": (
            "representative median selected-dev score across seeds 17/29/43; "
            "never best seed"
        ),
        "representative_checkpoints": representatives,
        "protocol_salt": SALT,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "p6_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6-01 / P6-02 GENERIC PROTOCOL PREPARED")
    print("=" * 108)
    print(f"enrollment K:         {args.enrollment_k}")
    print(f"enrollment samples:   {len(enrollment)}")
    print(f"known dev cal/score:  {len(known_cal)} / {len(known_score)}")
    print(f"unknown dev cal/score:{len(unknown_cal)} / {len(unknown_score)}")
    for method in METHODS:
        r = representatives[method]
        print(
            f"{method:10s} representative seed={r['seed']} "
            f"dev={r['score']:.6f} epoch={r['selected_epoch']}"
        )
    print("generic_test accessed: NO")
    print("-" * 108)
    print("P6 PROTOCOL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
