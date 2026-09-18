#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

TCRIT_DF2_95 = 4.302652729911275
CANONICAL_P5_ATTENTION_MEAN = 0.893376


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if len(x) == 3:
        half = TCRIT_DF2_95 * std / math.sqrt(3)
        ci = [mean - half, mean + half]
    else:
        ci = [None, None]
    return {"mean": mean, "sample_std": std, "ci95": ci, "values": [float(v) for v in x]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, default=Path("artifacts/p6_teacher_dim_sweep"))
    p.add_argument("--canonical-p5-mean", type=float, default=CANONICAL_P5_ATTENTION_MEAN)
    args = p.parse_args()

    manifest = json.loads((args.sweep_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    raw = json.loads((args.sweep_dir / "raw_results.json").read_text(encoding="utf-8"))
    by_dim = {}
    for r in raw:
        by_dim.setdefault(int(r["embedding_dim"]), []).append(float(r["selected_generic_dev_score"]))

    agg = {str(dim): summarize(vals) for dim, vals in sorted(by_dim.items())}
    dims = sorted(by_dim)
    ranked = sorted(dims, key=lambda d: (-agg[str(d)]["mean"], d))
    best_dim = ranked[0]

    repro64 = None
    if 64 in by_dim:
        repro64 = {
            "canonical_p5_attention_mean": float(args.canonical_p5_mean),
            "sweep_64d_mean": agg["64"]["mean"],
            "absolute_difference": abs(agg["64"]["mean"] - args.canonical_p5_mean),
            "reproduction_within_0p02": abs(agg["64"]["mean"] - args.canonical_p5_mean) <= 0.02,
        }

    result = {
        "schema": "papr_ssl.teacher_dim_sweep_summary.v1",
        "aggregate": agg,
        "ranked_dims": ranked,
        "highest_mean_dim": best_dim,
        "reproduction_64d": repro64,
        "project_2_64d_policy": "KEEP_CANONICAL_64D",
        "next_step": "Compare canonical 64D vs best high-dimensional candidate under same 15-shot personalized/open-set protocol.",
        "generic_test": "sealed_not_accessed",
    }
    (args.sweep_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("PAPR-SSL TEACHER EMBEDDING DIMENSION SWEEP SUMMARY")
    for d in ranked:
        s = agg[str(d)]
        lo, hi = s["ci95"]
        ci_text = f"[{lo:.6f}, {hi:.6f}]" if lo is not None else "n/a"
        print(f"{d:4d}D  mean={s['mean']:.6f}  std={s['sample_std']:.6f}  95%CI={ci_text}  seeds={s['values']}")
    print("-" * 108)
    if repro64:
        print(
            f"64D reproduction: {repro64['sweep_64d_mean']:.6f} vs canonical "
            f"{repro64['canonical_p5_attention_mean']:.6f} "
            f"(abs diff={repro64['absolute_difference']:.6f})"
        )
        print("64D reproduction within 0.02:",
              "PASS" if repro64["reproduction_within_0p02"] else "FAIL")
    print(f"highest 3-seed mean dimension: {best_dim}D")
    print("Project-2 canonical 64D Teacher: KEEP")
    print("generic_test accessed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
