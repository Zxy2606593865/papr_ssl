#!/usr/bin/env python
"""Audit the P4-05 statistical aggregate without ranking/selecting winners."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_COUNTS = {
    "wav2vec2_base": 13,
    "wavlm_large": 25,
    "w2v_bert2": 25,
}
EXPECTED_SEEDS = [17, 29, 43]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--statistics-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_05_statistics/"
            "p4_05_multiseed_statistics.json"
        ),
    )
    args = p.parse_args()

    payload = json.loads(
        args.statistics_json.read_text(encoding="utf-8")
    )

    if payload["schema"] != "papr_ssl.p4_05_multiseed_statistics.v1":
        raise RuntimeError("P4-05 schema mismatch")
    if payload["phase"] != "P4-05":
        raise RuntimeError("P4-05 phase mismatch")
    if payload["seeds"] != EXPECTED_SEEDS:
        raise RuntimeError("P4-05 seed contract changed")
    if payload["selection_metric"] != "generic_dev_score":
        raise RuntimeError("P4-05 metric changed")
    if payload["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if payload["ranking_performed"] is not False:
        raise RuntimeError("P4-05 must not rank configurations")
    if payload["best_layer_selected"] is not False:
        raise RuntimeError("P4-05 must not select a best layer")
    if payload["best_backbone_selected"] is not False:
        raise RuntimeError("P4-05 must not select a best backbone")

    rows = payload["rows"]
    if len(rows) != 63:
        raise RuntimeError(f"expected 63 rows, got {len(rows)}")
    if payload["raw_layer_seed_result_count"] != 189:
        raise RuntimeError("expected 189 raw layer-seed results")

    counts = {name: 0 for name in EXPECTED_COUNTS}
    keys = set()
    for row in rows:
        backbone = row["backbone"]
        layer = int(row["hidden_state_index"])
        key = (backbone, layer)
        if key in keys:
            raise RuntimeError(f"duplicate aggregate row: {key}")
        keys.add(key)
        counts[backbone] += 1

        for stat_name in (
            "trained_generic_dev_score",
            "untrained_projection_generic_dev_score",
            "absolute_delta",
        ):
            stat = row[stat_name]
            if stat["n"] != 3:
                raise RuntimeError(f"{key}: n != 3")
            if stat["std_ddof"] != 1:
                raise RuntimeError(f"{key}: ddof != 1")
            for field in (
                "mean",
                "sample_std",
                "standard_error",
                "ci95_lower",
                "ci95_upper",
            ):
                if not math.isfinite(float(stat[field])):
                    raise RuntimeError(
                        f"{key}: non-finite {stat_name}.{field}"
                    )
            if stat["ci95_lower"] > stat["mean"]:
                raise RuntimeError(f"{key}: CI lower > mean")
            if stat["ci95_upper"] < stat["mean"]:
                raise RuntimeError(f"{key}: CI upper < mean")

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Backbone layer counts mismatch: {counts}"
        )

    print("=" * 108)
    print("P4-05 STATISTICAL AGGREGATE AUDIT")
    print("=" * 108)
    print(f"Wav2Vec2 configurations:      {counts['wav2vec2_base']}")
    print(f"WavLM configurations:         {counts['wavlm_large']}")
    print(f"W2v-BERT 2.0 configurations:  {counts['w2v_bert2']}")
    print(f"Total configurations:         {len(rows)}")
    print("Seeds/config:                 3")
    print("Sample std ddof:              1")
    print("95% CI method:                Student-t, df=2")
    print("ranking performed:            NO")
    print("best layer selected:          NO")
    print("best backbone selected:       NO")
    print("generic_test accessed:        NO")
    print("-" * 108)
    print("P4-05 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
