#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result",
        type=Path,
        default=Path(
            "artifacts/p6_temporal_dtw/evaluation/"
            "temporal_dtw_ablation.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if d["redundancy_control"]["second_encoder"]:
        raise RuntimeError("Unexpected second encoder")
    if d["canonical_64d_teacher"] != "preserved_not_modified":
        raise RuntimeError("64D preservation contract violated")
    if d["project1_256d_teacher"] != "preserved_not_modified":
        raise RuntimeError("256D preservation contract violated")

    print("=" * 108)
    print("TEMPORAL DTW RERANKER AUDIT")
    print("=" * 108)

    for k in (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
    ):
        g = d["global_summary"][k]["mean"]
        t = d["dtw_summary"][k]["mean"]
        delta = d["dtw_minus_global_summary"][k]["mean"]
        print(
            f"{k:20s} global={g:.6f} dtw={t:.6f} delta={delta:+.6f}"
        )

    print("-" * 108)
    print(f"global Gate pass rate: {d['global_gate_pass_rate']:.3f}")
    print(f"DTW Gate pass rate:    {d['dtw_gate_pass_rate']:.3f}")
    print(f"lambda histogram:      {d['lambda_histogram']}")
    print("second encoder added:  NO")
    print("DTW top-k classes:     3")
    print("templates per class:   2")
    print("generic_test accessed: NO")
    print("AUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
