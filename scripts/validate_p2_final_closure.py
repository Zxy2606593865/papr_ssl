#!/usr/bin/env python
"""Final P2 closure validator.

This script does not regenerate data. It validates the already materialized
MDSC task-policy summary and checks the frozen P2 invariants before P3 starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "core30": {
        "count": 5083,
        "class_count": 30,
        "train": 3756,
        "dev": 442,
        "test": 885,
        "control": 2763,
        "dysarthria": 2320,
    },
    "command20": {
        "count": 2783,
        "class_count": 20,
        "train": 2056,
        "dev": 242,
        "test": 485,
        "control": 1513,
        "dysarthria": 1270,
    },
    "open_set": {
        "count": 3533,
        "dev": 1178,
        "test": 2355,
        "control": 1177,
        "dysarthria": 2356,
    },
}


def check(name: str, condition: bool, details: str = "") -> tuple[str, bool, str]:
    return name, bool(condition), details


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/"
            "mdsc_task_policy_v2_summary.json"
        ),
    )
    args = p.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    views = data["views"]

    core = views["mdsc_core30_exact_phrase"]
    cmd = views["mdsc_command20_exact_phrase"]
    open_set = views["mdsc_core30_open_set_eval"]

    checks = [
        check(
            "schema",
            data.get("schema") == "papr_ssl.mdsc_task_policy.v2",
            str(data.get("schema")),
        ),
        check(
            "phase",
            data.get("phase") == "P2-07E",
            str(data.get("phase")),
        ),
        check(
            "gate overall",
            data.get("gate", {}).get("overall") == "PASS",
            str(data.get("gate", {}).get("overall")),
        ),
        check(
            "Core30 exact count",
            core["count"] == EXPECTED["core30"]["count"],
            f"{core['count']} != {EXPECTED['core30']['count']}",
        ),
        check(
            "Core30 class count",
            core["class_count"] == EXPECTED["core30"]["class_count"],
            str(core["class_count"]),
        ),
        check(
            "Core30 split counts",
            core["splits"] == {
                "dev": EXPECTED["core30"]["dev"],
                "test": EXPECTED["core30"]["test"],
                "train": EXPECTED["core30"]["train"],
            },
            str(core["splits"]),
        ),
        check(
            "Core30 domain counts",
            core["domains"] == {
                "control": EXPECTED["core30"]["control"],
                "dysarthria": EXPECTED["core30"]["dysarthria"],
            },
            str(core["domains"]),
        ),
        check(
            "Command20 exact count",
            cmd["count"] == EXPECTED["command20"]["count"],
            f"{cmd['count']} != {EXPECTED['command20']['count']}",
        ),
        check(
            "Command20 class count",
            cmd["class_count"] == EXPECTED["command20"]["class_count"],
            str(cmd["class_count"]),
        ),
        check(
            "Command20 split counts",
            cmd["splits"] == {
                "dev": EXPECTED["command20"]["dev"],
                "test": EXPECTED["command20"]["test"],
                "train": EXPECTED["command20"]["train"],
            },
            str(cmd["splits"]),
        ),
        check(
            "Command20 domain counts",
            cmd["domains"] == {
                "control": EXPECTED["command20"]["control"],
                "dysarthria": EXPECTED["command20"]["dysarthria"],
            },
            str(cmd["domains"]),
        ),
        check(
            "Open-set exact count",
            open_set["count"] == EXPECTED["open_set"]["count"],
            f"{open_set['count']} != {EXPECTED['open_set']['count']}",
        ),
        check(
            "Open-set split counts",
            open_set["splits"] == {
                "dev": EXPECTED["open_set"]["dev"],
                "test": EXPECTED["open_set"]["test"],
            },
            str(open_set["splits"]),
        ),
        check(
            "Open-set domain counts",
            open_set["domains"] == {
                "control": EXPECTED["open_set"]["control"],
                "dysarthria": EXPECTED["open_set"]["dysarthria"],
            },
            str(open_set["domains"]),
        ),
        check(
            "Open-set not SCAF class",
            open_set.get("use_as_scaf_training_class") is False,
            str(open_set.get("use_as_scaf_training_class")),
        ),
        check(
            "No semantic merging",
            data["policy"].get("semantic_merging") is False,
            str(data["policy"].get("semantic_merging")),
        ),
        check(
            "Raw manifest unchanged",
            data["policy"].get("raw_manifest_rewrite") is False,
            str(data["policy"].get("raw_manifest_rewrite")),
        ),
    ]

    print("=" * 92)
    print("PAPR-SSL P2 FINAL CLOSURE VALIDATION")
    print("=" * 92)

    failed = 0
    for name, ok, details in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{name:42s} {status:4s}  {details}")
        failed += int(not ok)

    print("-" * 92)
    if failed:
        print(f"P2 FINAL STATUS: FAIL ({failed} failed checks)")
        return 1

    print("P2 FINAL STATUS: PASS / CLOSED")
    print("P3 may start only after the full unit-test suite also passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
