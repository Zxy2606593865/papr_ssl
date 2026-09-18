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
        default=Path("artifacts/h6_02_cached_demo/result.json"),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    assert d["schema"] == "papr_ssl.h6_02_cached_demo.v1"
    assert d["feature_source"] == "frozen_cached_dev_features"
    assert d["raw_wav_used"] is False
    assert d["shot"] == 2
    assert d["registered_intents"] == 20
    assert d["unregistered_intents"] == 10
    assert d["scores_are_calibrated_probabilities"] is False
    assert d["speaker_authentication_performed"] is False
    assert d["generic_test"] == "sealed_not_accessed"

    k = d["known_summary"]
    u = d["unknown_summary"]

    assert abs(
        k["correct_accept"]
        + k["wrong_accept"]
        + k["confirm"]
        + k["reject"]
        - 1.0
    ) < 1e-9
    assert abs(
        u["accept"] + u["confirm"] + u["reject"] - 1.0
    ) < 1e-9

    valid = {"ACCEPT", "CONFIRM", "REJECT"}
    for ex in d["examples"].values():
        p = ex["prediction"]
        assert p["status"] in valid
        assert set(p["decision_scores"]) == {"C", "W", "U"}
        assert p["scores_are_calibrated_probabilities"] is False
        if p["status"] == "REJECT":
            assert p["intent_id"] is None
            assert p["canonical_text"] is None

    print("=" * 108)
    print("H6-02 PERSONALIZED RUNTIME AUDIT")
    print("=" * 108)
    print(f"user_id:               {d['user_id']}")
    print(f"registered intents:    {d['registered_intents']}")
    print(f"shot:                  {d['shot']}")
    print(
        "KNOWN  | "
        f"CA={k['correct_accept']:.6f} "
        f"WA={k['wrong_accept']:.6f} "
        f"CONFIRM={k['confirm']:.6f} "
        f"REJECT={k['reject']:.6f}"
    )
    print(
        "UNKNOWN| "
        f"ACCEPT={u['accept']:.6f} "
        f"CONFIRM={u['confirm']:.6f} "
        f"REJECT={u['reject']:.6f}"
    )
    print("-" * 108)
    print("Prediction schema:      PASS")
    print("Memory persistence:     PASS (run script save/reload)")
    print("Raw WAV adapter:        NOT YET")
    print("generic_test accessed:  NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
