#!/usr/bin/env python
"""Aggregate P5 Mean-vs-Attention results and select the winning DR method."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


METHODS = ("mean", "attention")
SEEDS = (17, 29, 43)
TCRIT = 4.302652729911275
P4_MEAN_REFERENCE = 0.8864723619749779


def stats(values: list[float]) -> dict:
    m = mean(values)
    s = stdev(values)
    half = TCRIT * s / math.sqrt(3)
    return {
        "values": values,
        "mean": m,
        "sample_std": s,
        "ci95_lower": m - half,
        "ci95_upper": m + half,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/p5_runs/mean_vs_attention"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p5_runs/p5_summary"),
    )
    args = p.parse_args()

    summaries = []
    for method in METHODS:
        vals = []
        epochs = []
        for seed in SEEDS:
            path = (
                args.run_root
                / method
                / f"seed_{seed:04d}"
                / "p5_result.json"
            )
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("generic_test_accessed") is not False:
                raise RuntimeError("generic_test access violation")
            vals.append(float(obj["selected_generic_dev_score"]))
            epochs.append(int(obj["selected_epoch"]))
        s = stats(vals)
        s.update(
            {
                "method": method,
                "seeds": list(SEEDS),
                "selected_epochs": epochs,
            }
        )
        summaries.append(s)

    mean_summary = next(x for x in summaries if x["method"] == "mean")
    attention_summary = next(
        x for x in summaries if x["method"] == "attention"
    )

    # Reproduction guard: Mean branch should remain close to the P4 winner.
    mean_abs_diff_vs_p4 = abs(
        mean_summary["mean"] - P4_MEAN_REFERENCE
    )
    mean_reproduction_pass = mean_abs_diff_vs_p4 <= 0.02

    winner = max(summaries, key=lambda x: x["mean"])
    delta_attention_minus_mean = (
        attention_summary["mean"] - mean_summary["mean"]
    )

    output = {
        "schema": "papr_ssl.p5_mean_vs_attention_summary.v1",
        "phase": "P5",
        "backbone": "wavlm_large",
        "hidden_state_index": 15,
        "selection_metric": "generic_dev_score",
        "selection_metric_definition": "prototype_macro_f1",
        "methods": summaries,
        "p4_mean_reference": P4_MEAN_REFERENCE,
        "mean_reproduction_abs_diff": mean_abs_diff_vs_p4,
        "mean_reproduction_tolerance": 0.02,
        "mean_reproduction_pass": mean_reproduction_pass,
        "attention_minus_mean": delta_attention_minus_mean,
        "selected_dr_method": winner["method"],
        "selected_dr_mean": winner["mean"],
        "generic_test": "sealed_not_accessed",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "p5_mean_vs_attention_summary.json"
    out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P5 MEAN DR vs ATTENTION DR SUMMARY")
    print("=" * 108)
    for x in summaries:
        print(
            f"{x['method']:10s} "
            f"mean={x['mean']:.6f} "
            f"std={x['sample_std']:.6f} "
            f"95%CI=[{x['ci95_lower']:.6f}, {x['ci95_upper']:.6f}]"
        )
    print("-" * 108)
    print(f"P4 Mean reference:           {P4_MEAN_REFERENCE:.6f}")
    print(
        "P5 Mean reproduction diff:  "
        f"{mean_abs_diff_vs_p4:.6f}"
    )
    print(
        "P5 Mean reproduction gate:  "
        f"{'PASS' if mean_reproduction_pass else 'FAIL'}"
    )
    print(
        "Attention - Mean:            "
        f"{delta_attention_minus_mean:+.6f}"
    )
    print(f"selected DR method:           {winner['method']}")
    print("generic_test accessed:        NO")
    print("-" * 108)
    if not mean_reproduction_pass:
        print("P5 STATUS: FAIL (MEAN REPRODUCTION)")
        return 2
    print("P5 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
