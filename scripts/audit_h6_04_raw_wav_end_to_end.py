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
        default=Path("artifacts/h6_04_raw_wav_end_to_end/summary.json"),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    assert d["schema"] == "papr_ssl.h6_04_raw_wav_end_to_end.v1"
    assert d["raw_wav_enrollment"] is True
    assert d["raw_wav_query"] is True
    assert d["registered_intents"] == 20
    assert d["shot"] == 2
    assert d["generic_test"] == "sealed_not_accessed"

    print("=" * 108)
    print("H6-04 RAW-WAV END-TO-END AUDIT")
    print("=" * 108)
    print(f"speaker:                  {d['speaker_id']}")
    print(f"registered intents:       {d['registered_intents']}")
    print(f"shot:                     {d['shot']}")
    print(f"queries:                  {d['query_count']}")
    print(f"status match rate:        {d['status_match_rate']:.6f}")
    print(f"intent match rate:        {d['intent_match_rate']:.6f}")
    print("-" * 108)

    k = d["known_summary"]
    u = d["unknown_summary"]

    print(
        "KNOWN   | "
        f"CA={k['correct_accept']:.6f} "
        f"WA={k['wrong_accept']:.6f} "
        f"CONFIRM={k['confirm']:.6f} "
        f"REJECT={k['reject']:.6f}"
    )
    print(
        "UNKNOWN | "
        f"ACCEPT={u['accept']:.6f} "
        f"CONFIRM={u['confirm']:.6f} "
        f"REJECT={u['reject']:.6f}"
    )
    print("-" * 108)
    print("generic_test accessed:     NO")

    if not d["all_predictions_match_cached_runtime"]:
        raise SystemExit(
            "AUDIT STATUS: FAIL — raw-WAV runtime does not reproduce cached runtime."
        )

    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
