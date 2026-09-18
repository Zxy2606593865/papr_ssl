#!/usr/bin/env python
"""Audit Demo-04A outputs without changing thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime/result.json"),
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime/predictions.jsonl"),
    )
    args = p.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    errors = []

    if result.get("generic_test") != "sealed_not_accessed":
        errors.append("generic_test seal not intact")
    if result.get("training_or_threshold_fitting_performed") is not False:
        errors.append("training/threshold fitting unexpectedly performed")
    if result.get("shot") != 2:
        errors.append("expected 2-shot")
    if result.get("registered_intents") != 5:
        errors.append("expected 5 registered intents")

    known = [r for r in rows if r["kind"] == "known"]
    unknown = [r for r in rows if r["kind"] == "unknown"]

    if len(known) != result["known_queries"]:
        errors.append("known query count mismatch")
    if len(unknown) != result["unknown_queries"]:
        errors.append("unknown query count mismatch")

    print("=" * 112)
    print("DEMO-04A WANGHAO RUNTIME AUDIT")
    print("=" * 112)
    print(f"registered intents:       {result['registered_intents']}")
    print(f"shot:                     {result['shot']}")
    print(f"known queries:            {result['known_queries']}")
    print(
        f"source-disjoint known:    "
        f"{result['known_source_disjoint_queries']}/"
        f"{result['known_queries']}"
    )
    print(f"unknown queries:          {result['unknown_queries']}")
    print(
        "KNOWN   | "
        f"CA={result['known_summary']['correct_accept']:.4f} "
        f"WA={result['known_summary']['wrong_accept']:.4f} "
        f"CONFIRM={result['known_summary']['confirm']:.4f} "
        f"REJECT={result['known_summary']['reject']:.4f}"
    )
    print(
        "UNKNOWN | "
        f"ACCEPT={result['unknown_summary']['accept']:.4f} "
        f"CONFIRM={result['unknown_summary']['confirm']:.4f} "
        f"REJECT={result['unknown_summary']['reject']:.4f}"
    )
    print("-" * 112)
    print("PER INTENT")
    for intent_id, r in result["per_intent"].items():
        print(
            f"{intent_id:<30} "
            f"N={r['count']:>2d} "
            f"CA={r['correct_accept']:.3f} "
            f"WA={r['wrong_accept']:.3f} "
            f"C={r['confirm']:.3f} "
            f"R={r['reject']:.3f}"
        )

    if errors:
        print("-" * 112)
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(f"AUDIT STATUS: FAIL ({len(errors)} errors)")

    print("-" * 112)
    print("AUDIT STATUS:             PASS")
    print(
        "NOTE: PASS means the evaluation protocol/output is structurally valid; "
        "it does NOT mean Wanghao recognition performance met a predefined gate."
    )


if __name__ == "__main__":
    main()
