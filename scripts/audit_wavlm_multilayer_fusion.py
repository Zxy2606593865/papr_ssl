#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/p6_wavlm_multilayer_fusion/summary.json"),
    )
    args = p.parse_args()

    d = json.loads(args.summary.read_text(encoding="utf-8"))
    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if d["canonical_64d_teacher"] != "preserved_not_modified":
        raise RuntimeError("canonical 64D preservation contract violated")

    print("=" * 108)
    print("WAVLM MULTI-LAYER FUSION AUDIT")
    print("=" * 108)
    for mode, s in d["aggregate"].items():
        print(
            f"{mode:18s} mean={s['mean']:.6f} std={s['sample_std']:.6f} "
            f"seeds={s['values']}"
        )
    print(f"weighted - single15: {d['weighted_minus_single15_mean']:+.6f}")

    repro = d.get("single15_reproduction")
    if repro:
        print(
            f"single15 reproduction: "
            f"{'PASS' if repro['within_0p02'] else 'FAIL'} "
            f"(abs diff={repro['absolute_difference']:.6f})"
        )

    fw = d.get("fusion_weight_summary")
    if fw:
        print("mean fusion weights:")
        for layer, weight in zip(fw["layers"], fw["mean"]):
            print(f"  layer {layer}: {weight:.4f}")

    print("canonical 64D modified: NO")
    print("generic_test accessed:  NO")
    print("AUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
