#!/usr/bin/env python
"""P4-05: aggregate all Backbone/hidden-state 3-seed statistics.

This phase performs statistics only.
It does NOT rank layers and does NOT select a best Backbone/Layer.
generic_test remains sealed.

For each Backbone/Layer configuration:
- n = 3 seeds: 17, 29, 43
- mean
- sample standard deviation (ddof=1)
- standard error
- two-sided 95% Student-t confidence interval with df=2

The same statistics are reported for:
- selected trained generic_dev_score
- untrained projection generic_dev_score
- absolute delta = trained - untrained
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


SEEDS = (17, 29, 43)
T_CRIT_975_DF2 = 4.302652729911275

SOURCES = (
    {
        "backbone": "wav2vec2_base",
        "path": Path(
            "artifacts/p4_runs/wav2vec2_base/p4_02_raw_results.json"
        ),
        "expected_layers": tuple(range(13)),
        "expected_rows": 39,
    },
    {
        "backbone": "wavlm_large",
        "path": Path(
            "artifacts/p4_runs/wavlm_large/p4_03_raw_results.json"
        ),
        "expected_layers": tuple(range(25)),
        "expected_rows": 75,
    },
    {
        "backbone": "w2v_bert2",
        "path": Path(
            "artifacts/p4_runs/w2v_bert2/p4_04_raw_results.json"
        ),
        "expected_layers": tuple(range(25)),
        "expected_rows": 75,
    },
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def stats3(values: list[float]) -> dict:
    if len(values) != 3:
        raise ValueError(f"expected exactly 3 values, got {len(values)}")
    m = float(mean(values))
    s = float(stdev(values))
    se = s / math.sqrt(3.0)
    half = T_CRIT_975_DF2 * se
    return {
        "values": [float(x) for x in values],
        "n": 3,
        "mean": m,
        "sample_std": s,
        "std_ddof": 1,
        "standard_error": se,
        "ci95_method": "two-sided Student-t",
        "ci95_df": 2,
        "ci95_t_critical": T_CRIT_975_DF2,
        "ci95_lower": m - half,
        "ci95_upper": m + half,
    }


def validate_source(
    *,
    payload: dict,
    backbone: str,
    expected_layers: tuple[int, ...],
    expected_rows: int,
    path: Path,
) -> list[dict]:
    if payload.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError(f"{path}: generic_test is not sealed")
    if payload.get("selection_metric") != "generic_dev_score":
        raise RuntimeError(f"{path}: selection metric changed")
    if payload.get("ranking_performed") is not False:
        raise RuntimeError(f"{path}: ranking was already performed")
    if payload.get("best_layer_selected") is not False:
        raise RuntimeError(f"{path}: best layer was already selected")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing rows list")
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{path}: expected {expected_rows} rows, got {len(rows)}"
        )

    by_key: dict[tuple[int, int], dict] = {}
    for row in rows:
        layer = int(row["layer"])
        seed = int(row["seed"])
        key = (layer, seed)
        if key in by_key:
            raise RuntimeError(f"{path}: duplicate row {key}")
        by_key[key] = row

    expected_keys = {
        (layer, seed)
        for layer in expected_layers
        for seed in SEEDS
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise RuntimeError(
            f"{path}: layer/seed coverage mismatch; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    return rows


def aggregate_backbone(
    *,
    backbone: str,
    rows: list[dict],
    expected_layers: tuple[int, ...],
) -> list[dict]:
    out = []
    for layer in expected_layers:
        layer_rows = sorted(
            (row for row in rows if int(row["layer"]) == layer),
            key=lambda x: int(x["seed"]),
        )
        seeds = tuple(int(row["seed"]) for row in layer_rows)
        if seeds != SEEDS:
            raise RuntimeError(
                f"{backbone} layer {layer}: seed order/coverage mismatch: {seeds}"
            )

        trained = [
            float(row["selected_generic_dev_score"])
            for row in layer_rows
        ]
        baseline = [
            float(row["baseline_generic_dev_score"])
            for row in layer_rows
        ]
        delta = [
            t - b
            for t, b in zip(trained, baseline)
        ]
        epochs = [
            int(row["selected_epoch"])
            for row in layer_rows
        ]

        out.append(
            {
                "backbone": backbone,
                "hidden_state_index": layer,
                "seeds": list(SEEDS),
                "selected_epoch_by_seed": epochs,
                "trained_generic_dev_score": stats3(trained),
                "untrained_projection_generic_dev_score": stats3(
                    baseline
                ),
                "absolute_delta": stats3(delta),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "backbone",
        "hidden_state_index",
        "seed17",
        "seed29",
        "seed43",
        "mean",
        "sample_std",
        "ci95_lower",
        "ci95_upper",
        "baseline_mean",
        "delta_mean",
        "selected_epoch_seed17",
        "selected_epoch_seed29",
        "selected_epoch_seed43",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            trained = row["trained_generic_dev_score"]
            baseline = row["untrained_projection_generic_dev_score"]
            delta = row["absolute_delta"]
            epochs = row["selected_epoch_by_seed"]
            writer.writerow(
                {
                    "backbone": row["backbone"],
                    "hidden_state_index": row["hidden_state_index"],
                    "seed17": trained["values"][0],
                    "seed29": trained["values"][1],
                    "seed43": trained["values"][2],
                    "mean": trained["mean"],
                    "sample_std": trained["sample_std"],
                    "ci95_lower": trained["ci95_lower"],
                    "ci95_upper": trained["ci95_upper"],
                    "baseline_mean": baseline["mean"],
                    "delta_mean": delta["mean"],
                    "selected_epoch_seed17": epochs[0],
                    "selected_epoch_seed29": epochs[1],
                    "selected_epoch_seed43": epochs[2],
                }
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p4_runs/p4_05_statistics"),
    )
    args = p.parse_args()

    all_rows: list[dict] = []
    source_records = []

    for spec in SOURCES:
        payload = read_json(spec["path"])
        rows = validate_source(
            payload=payload,
            backbone=spec["backbone"],
            expected_layers=spec["expected_layers"],
            expected_rows=spec["expected_rows"],
            path=spec["path"],
        )
        aggregated = aggregate_backbone(
            backbone=spec["backbone"],
            rows=rows,
            expected_layers=spec["expected_layers"],
        )
        all_rows.extend(aggregated)
        source_records.append(
            {
                "backbone": spec["backbone"],
                "source_path": spec["path"].as_posix(),
                "layer_count": len(spec["expected_layers"]),
                "raw_row_count": len(rows),
            }
        )

    if len(all_rows) != 63:
        raise RuntimeError(
            f"expected 63 Backbone/Layer aggregate rows, got {len(all_rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "schema": "papr_ssl.p4_05_multiseed_statistics.v1",
        "phase": "P4-05",
        "purpose": "statistics_only_no_ranking_no_selection",
        "selection_metric": "generic_dev_score",
        "selection_metric_definition": "prototype_macro_f1",
        "seeds": list(SEEDS),
        "statistics": {
            "mean": True,
            "sample_std": True,
            "std_ddof": 1,
            "confidence_interval": "two-sided 95% Student-t",
            "confidence_interval_df": 2,
            "confidence_interval_t_critical": T_CRIT_975_DF2,
        },
        "sources": source_records,
        "configuration_count": 63,
        "raw_layer_seed_result_count": 189,
        "rows": all_rows,
        "ranking_performed": False,
        "best_layer_selected": False,
        "best_backbone_selected": False,
        "generic_test": "sealed_not_accessed",
    }

    json_path = args.output_dir / "p4_05_multiseed_statistics.json"
    csv_path = args.output_dir / "p4_05_multiseed_statistics.csv"

    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "P4-05 output already exists; refusing silent overwrite. "
            f"Remove/archive the old output explicitly: {args.output_dir}"
        )

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, all_rows)

    print("=" * 108)
    print("PAPR-SSL P4-05 THREE-SEED STATISTICAL AGGREGATION")
    print("=" * 108)
    for record in source_records:
        print(
            f"{record['backbone']:18s} "
            f"layers={record['layer_count']:2d}  "
            f"raw_results={record['raw_row_count']:3d}"
        )
    print("-" * 108)
    print(f"Backbone/Layer configs:       {len(all_rows)}")
    print("Seeds/config:                 3  (17, 29, 43)")
    print("Raw layer-seed results:       189")
    print("Std definition:               sample std, ddof=1")
    print("95% CI:                       Student-t, df=2")
    print("ranking performed:            NO")
    print("best layer selected:          NO")
    print("best backbone selected:       NO")
    print("generic_test accessed:        NO")
    print(f"JSON:                          {json_path}")
    print(f"CSV:                           {csv_path}")
    print("-" * 108)
    print("P4-05 STATISTICS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
