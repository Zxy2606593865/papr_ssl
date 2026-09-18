#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/h6_01_three_state_policy/h6_01_eval.json"),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    assert d["generic_test"] == "sealed_not_accessed"
    assert d["backbone_neck_updated"] is False
    assert d["scores_are_calibrated_probabilities"] is False
    assert d["confirm_counts_as_auto_success"] is False

    print("=" * 108)
    print("H6-01 THREE-STATE POLICY AUDIT")
    print("=" * 108)

    for shot in ("1", "2"):
        m = d["summary"][shot]["metrics"]

        assert abs(m["known_identity_sum"] - 1.0) < 1e-9
        assert abs(m["unknown_identity_sum"] - 1.0) < 1e-9

        print(
            f"{shot}-shot KNOWN   | "
            f"CA={m['correct_accept']:.6f} "
            f"WA={m['wrong_accept']:.6f} "
            f"CONFIRM={m['known_confirm']:.6f} "
            f"REJECT={m['known_reject']:.6f}"
        )
        print(
            f"{shot}-shot UNKNOWN | "
            f"ACCEPT={m['unknown_accept']:.6f} "
            f"CONFIRM={m['unknown_confirm']:.6f} "
            f"REJECT={m['unknown_reject']:.6f}"
        )
        print(
            f"{shot}-shot ACCEPT  | "
            f"coverage={m['accept_coverage_all_queries']:.6f} "
            f"selective_risk={m['selective_risk_on_accept']:.6f}"
        )

    print("-" * 108)
    print("Known partition sum:      PASS")
    print("Unknown partition sum:    PASS")
    print("CONFIRM auto-success:     NO")
    print("generic_test accessed:    NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
