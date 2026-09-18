#!/usr/bin/env python
"""P4-06: select best layer per Backbone, then best Backbone configuration.

Selection rule is frozen from P4:
1) primary metric = 3-seed mean generic_dev_score
   (= prototype_macro_f1)
2) within a Backbone, exact mean tie -> lower hidden_state_index
3) compare the three per-Backbone winners by the same 3-seed mean
4) generic_test remains sealed

P4-06 does not use resource cost to override embedding quality.
Resource cost belongs to P4-07.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BACKBONES = ("wav2vec2_base", "wavlm_large", "w2v_bert2")


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def choose_best_layer(rows: list[dict], backbone: str) -> dict:
    candidates = [
        r for r in rows if r["backbone"] == backbone
    ]
    if not candidates:
        raise RuntimeError(f"no rows for {backbone}")

    # Exact mean tie -> lower hidden_state_index.
    candidates = sorted(
        candidates,
        key=lambda r: (
            -float(r["trained_generic_dev_score"]["mean"]),
            int(r["hidden_state_index"]),
        ),
    )
    return candidates[0]


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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p4_runs/p4_06_selection"),
    )
    args = p.parse_args()

    payload = load_json(args.statistics_json)

    if payload.get("phase") != "P4-05":
        raise RuntimeError("input is not a P4-05 aggregate")
    if payload.get("selection_metric") != "generic_dev_score":
        raise RuntimeError("selection metric changed")
    if payload.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if payload.get("configuration_count") != 63:
        raise RuntimeError("expected 63 Backbone/Layer configurations")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 63:
        raise RuntimeError("P4-05 aggregate row count mismatch")

    per_backbone = []
    for backbone in BACKBONES:
        best = choose_best_layer(rows, backbone)
        stat = best["trained_generic_dev_score"]
        per_backbone.append(
            {
                "backbone": backbone,
                "hidden_state_index": int(best["hidden_state_index"]),
                "seeds": best["seeds"],
                "selected_epoch_by_seed": best[
                    "selected_epoch_by_seed"
                ],
                "generic_dev_score_mean": float(stat["mean"]),
                "generic_dev_score_sample_std": float(
                    stat["sample_std"]
                ),
                "generic_dev_score_ci95_lower": float(
                    stat["ci95_lower"]
                ),
                "generic_dev_score_ci95_upper": float(
                    stat["ci95_upper"]
                ),
                "untrained_projection_mean": float(
                    best[
                        "untrained_projection_generic_dev_score"
                    ]["mean"]
                ),
                "absolute_delta_mean": float(
                    best["absolute_delta"]["mean"]
                ),
            }
        )

    ranked = sorted(
        per_backbone,
        key=lambda x: -x["generic_dev_score_mean"],
    )

    if len(ranked) >= 2:
        top = ranked[0]["generic_dev_score_mean"]
        second = ranked[1]["generic_dev_score_mean"]
        if top == second:
            raise RuntimeError(
                "exact cross-Backbone mean tie; P4-06 has no "
                "resource-based tie-break. Resolve in P4-07."
            )

    winner = dict(ranked[0])
    winner["rank"] = 1

    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    output = {
        "schema": "papr_ssl.p4_06_backbone_layer_selection.v1",
        "phase": "P4-06",
        "selection_metric": "generic_dev_score",
        "selection_metric_definition": "prototype_macro_f1",
        "aggregation": "3-seed mean over seeds 17/29/43",
        "within_backbone_exact_tie_break": "lower_hidden_state_index",
        "resource_cost_used_for_selection": False,
        "per_backbone_winners": per_backbone,
        "backbone_winner_ranking": ranked,
        "selected_mainline": winner,
        "generic_test": "sealed_not_accessed",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "p4_06_selection.json"
    csv_path = args.output_dir / "p4_06_winners.csv"

    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "P4-06 output already exists; refusing silent overwrite"
        )

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "rank",
            "backbone",
            "hidden_state_index",
            "generic_dev_score_mean",
            "generic_dev_score_sample_std",
            "generic_dev_score_ci95_lower",
            "generic_dev_score_ci95_upper",
            "untrained_projection_mean",
            "absolute_delta_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in ranked:
            w.writerow({k: row[k] for k in fields})

    print("=" * 108)
    print("PAPR-SSL P4-06 BACKBONE / HIDDEN-STATE SELECTION")
    print("=" * 108)
    for row in ranked:
        print(
            f"rank {row['rank']}: "
            f"{row['backbone']:18s} "
            f"hidden_states[{row['hidden_state_index']:2d}]  "
            f"mean={row['generic_dev_score_mean']:.6f}  "
            f"std={row['generic_dev_score_sample_std']:.6f}  "
            f"95%CI=[{row['generic_dev_score_ci95_lower']:.6f}, "
            f"{row['generic_dev_score_ci95_upper']:.6f}]"
        )
    print("-" * 108)
    print(
        "selected mainline: "
        f"{winner['backbone']} hidden_states["
        f"{winner['hidden_state_index']}]"
    )
    print(
        "selected mean generic_dev_score: "
        f"{winner['generic_dev_score_mean']:.6f}"
    )
    print("resource cost used for selection: NO")
    print("generic_test accessed:            NO")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("-" * 108)
    print("P4-06 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
