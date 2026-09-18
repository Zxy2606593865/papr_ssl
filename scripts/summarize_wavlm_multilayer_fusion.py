#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TCRIT_DF2_95 = 4.302652729911275
PRIOR_256D_SINGLE15_MEAN = 0.912667


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if len(x) == 3:
        half = TCRIT_DF2_95 * std / math.sqrt(3)
        ci = [mean-half, mean+half]
    else:
        ci = [None, None]
    return {
        "values": [float(v) for v in x],
        "mean": mean,
        "sample_std": std,
        "ci95": ci,
    }


def load_selected_weights(exp_dir: Path, seeds):
    rows = []
    for seed in seeds:
        ckpt = torch_load(exp_dir / "weighted13_17" / f"seed_{seed}" / "best.pt")
        rows.append(ckpt["selected_fusion_weights_13_17"])
    return np.asarray(rows, dtype=np.float64)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/p6_wavlm_multilayer_fusion"),
    )
    p.add_argument(
        "--prior-single15-256d-mean",
        type=float,
        default=PRIOR_256D_SINGLE15_MEAN,
    )
    args = p.parse_args()

    manifest = json.loads(
        (args.experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    raw = json.loads(
        (args.experiment_dir / "raw_results.json").read_text(encoding="utf-8")
    )
    by_mode = {}
    seeds_by_mode = {}
    for r in raw:
        by_mode.setdefault(r["mode"], []).append(float(r["selected_generic_dev_score"]))
        seeds_by_mode.setdefault(r["mode"], []).append(int(r["seed"]))

    aggregate = {mode: stats(vals) for mode, vals in by_mode.items()}

    delta = None
    if "single15" in aggregate and "weighted13_17" in aggregate:
        delta = aggregate["weighted13_17"]["mean"] - aggregate["single15"]["mean"]

    repro = None
    if "single15" in aggregate:
        repro = {
            "prior_256d_single15_mean": float(args.prior_single15_256d_mean),
            "same_precision_single15_mean": aggregate["single15"]["mean"],
            "absolute_difference": abs(
                aggregate["single15"]["mean"] - args.prior_single15_256d_mean
            ),
            "within_0p02": abs(
                aggregate["single15"]["mean"] - args.prior_single15_256d_mean
            ) <= 0.02,
        }

    fusion_weight_summary = None
    if "weighted13_17" in aggregate:
        seeds = sorted(seeds_by_mode["weighted13_17"])
        w = load_selected_weights(args.experiment_dir, seeds)
        fusion_weight_summary = {
            "layers": [13, 14, 15, 16, 17],
            "per_seed": {str(seed): [float(v) for v in row] for seed, row in zip(seeds, w)},
            "mean": [float(v) for v in w.mean(axis=0)],
            "sample_std": [float(v) for v in w.std(axis=0, ddof=1)] if len(w) > 1 else [0.0]*5,
        }

    result = {
        "schema": "papr_ssl.wavlm_multilayer_fusion_summary.v1",
        "aggregate": aggregate,
        "weighted_minus_single15_mean": delta,
        "single15_reproduction": repro,
        "fusion_weight_summary": fusion_weight_summary,
        "decision_rule": (
            "Do not promote fusion from closed-set mean alone. If weighted13_17 improves the 3-seed mean "
            "without pathological instability, evaluate it next under the same 15-shot personalized/open-set protocol."
        ),
        "canonical_64d_teacher": "preserved_not_modified",
        "generic_test": "sealed_not_accessed",
    }
    (args.experiment_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("WAVLM MULTI-LAYER FUSION SUMMARY")
    print("=" * 108)
    for mode in ("single15", "weighted13_17"):
        if mode not in aggregate:
            continue
        s = aggregate[mode]
        lo, hi = s["ci95"]
        ci = f"[{lo:.6f}, {hi:.6f}]" if lo is not None else "n/a"
        print(
            f"{mode:18s} mean={s['mean']:.6f} std={s['sample_std']:.6f} "
            f"95%CI={ci} seeds={s['values']}"
        )

    if delta is not None:
        print(f"weighted - single15: {delta:+.6f}")

    if repro is not None:
        print(
            f"single15 reproduction: {repro['same_precision_single15_mean']:.6f} "
            f"vs prior256D={repro['prior_256d_single15_mean']:.6f} "
            f"abs_diff={repro['absolute_difference']:.6f} "
            f"=> {'PASS' if repro['within_0p02'] else 'FAIL'}"
        )

    if fusion_weight_summary is not None:
        print("selected fusion weights mean:")
        for layer, weight, sd in zip(
            fusion_weight_summary["layers"],
            fusion_weight_summary["mean"],
            fusion_weight_summary["sample_std"],
        ):
            print(f"  layer {layer:2d}: mean={weight:.4f} std={sd:.4f}")

    print("canonical 64D modified: NO")
    print("generic_test accessed:  NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
