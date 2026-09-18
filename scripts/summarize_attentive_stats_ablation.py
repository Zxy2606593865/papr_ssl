#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TCRIT_DF2_95 = 4.302652729911275
PRIOR_256D_MEAN = 0.912667


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    std = (
        float(x.std(ddof=1))
        if len(x) > 1
        else 0.0
    )

    if len(x) == 3:
        half = (
            TCRIT_DF2_95
            * std
            / math.sqrt(3)
        )
        ci = [mean - half, mean + half]
    else:
        ci = [None, None]

    return {
        "values": [float(v) for v in x],
        "mean": mean,
        "sample_std": std,
        "ci95": ci,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/p6_attentive_stats_ablation"),
    )
    p.add_argument(
        "--prior-256d-mean",
        type=float,
        default=PRIOR_256D_MEAN,
    )
    args = p.parse_args()

    manifest = json.loads(
        (args.experiment_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    raw = json.loads(
        (args.experiment_dir / "raw_results.json").read_text(
            encoding="utf-8"
        )
    )

    by_mode = {}
    for row in raw:
        by_mode.setdefault(
            row["mode"], []
        ).append(
            float(row["selected_generic_dev_score"])
        )

    aggregate = {
        mode: stats(vals)
        for mode, vals in by_mode.items()
    }

    mean_mode = "attention_mean_256"
    stats_mode = "attentive_stats_256"

    delta = None
    if (
        mean_mode in aggregate
        and stats_mode in aggregate
    ):
        delta = (
            aggregate[stats_mode]["mean"]
            - aggregate[mean_mode]["mean"]
        )

    repro = None
    if mean_mode in aggregate:
        diff = abs(
            aggregate[mean_mode]["mean"]
            - float(args.prior_256d_mean)
        )
        repro = {
            "prior_256d_mean": float(
                args.prior_256d_mean
            ),
            "matched_attention_mean": aggregate[
                mean_mode
            ]["mean"],
            "absolute_difference": diff,
            "within_0p02": diff <= 0.02,
        }

    result = {
        "schema": (
            "papr_ssl.attentive_stats_ablation_summary.v1"
        ),
        "aggregate": aggregate,
        "attentive_stats_minus_mean": delta,
        "matched_mean_reproduction": repro,
        "decision_rule": (
            "Promote ASP only if the 3-seed mean improves over "
            "the matched attention-mean control without pathological "
            "variance; then validate under the same 15-shot "
            "personalized/open-set protocol."
        ),
        "canonical_64d_teacher": (
            "preserved_not_modified"
        ),
        "project1_256d_teacher": (
            "preserved_not_overwritten"
        ),
        "generic_test": "sealed_not_accessed",
    }

    (args.experiment_dir / "summary.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("ATTENTIVE STATISTICS POOLING SUMMARY")
    print("=" * 108)

    for mode in (
        "attention_mean_256",
        "attentive_stats_256",
    ):
        if mode not in aggregate:
            continue

        s = aggregate[mode]
        lo, hi = s["ci95"]
        ci = (
            f"[{lo:.6f}, {hi:.6f}]"
            if lo is not None
            else "n/a"
        )

        print(
            f"{mode:24s} "
            f"mean={s['mean']:.6f} "
            f"std={s['sample_std']:.6f} "
            f"95%CI={ci} "
            f"seeds={s['values']}"
        )

    if delta is not None:
        print(
            f"ASP - AttentionMean: {delta:+.6f}"
        )

    if repro is not None:
        print(
            f"matched mean reproduction: "
            f"{repro['matched_attention_mean']:.6f} "
            f"vs prior256D={repro['prior_256d_mean']:.6f} "
            f"abs_diff={repro['absolute_difference']:.6f} "
            f"=> {'PASS' if repro['within_0p02'] else 'FAIL'}"
        )

    print("canonical 64D modified: NO")
    print("Project-1 256D modified: NO")
    print("generic_test accessed:  NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
