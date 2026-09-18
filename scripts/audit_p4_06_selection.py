#!/usr/bin/env python
"""Audit P4-06 winner selection against the frozen mean-score rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKBONES = ("wav2vec2_base", "wavlm_large", "w2v_bert2")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selection-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_06_selection/"
            "p4_06_selection.json"
        ),
    )
    args = p.parse_args()

    payload = json.loads(
        args.selection_json.read_text(encoding="utf-8")
    )

    if payload["schema"] != "papr_ssl.p4_06_backbone_layer_selection.v1":
        raise RuntimeError("schema mismatch")
    if payload["phase"] != "P4-06":
        raise RuntimeError("phase mismatch")
    if payload["selection_metric"] != "generic_dev_score":
        raise RuntimeError("metric changed")
    if payload["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if payload["resource_cost_used_for_selection"] is not False:
        raise RuntimeError("P4-06 must not use resource cost")

    winners = payload["per_backbone_winners"]
    if len(winners) != 3:
        raise RuntimeError("expected one winner per Backbone")
    if {x["backbone"] for x in winners} != set(BACKBONES):
        raise RuntimeError("Backbone winner coverage mismatch")

    ranked = payload["backbone_winner_ranking"]
    means = [x["generic_dev_score_mean"] for x in ranked]
    if means != sorted(means, reverse=True):
        raise RuntimeError("Backbone ranking is not descending by mean")

    selected = payload["selected_mainline"]
    if selected["rank"] != 1:
        raise RuntimeError("selected mainline is not rank 1")

    print("=" * 108)
    print("P4-06 SELECTION AUDIT")
    print("=" * 108)
    for row in ranked:
        print(
            f"rank {row['rank']}: "
            f"{row['backbone']:18s} "
            f"hidden_states[{row['hidden_state_index']:2d}] "
            f"mean={row['generic_dev_score_mean']:.6f}"
        )
    print("-" * 108)
    print(
        "selected: "
        f"{selected['backbone']} "
        f"hidden_states[{selected['hidden_state_index']}]"
    )
    print("generic_test accessed: NO")
    print("resource cost used:    NO")
    print("-" * 108)
    print("P4-06 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
