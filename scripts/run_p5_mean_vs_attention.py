#!/usr/bin/env python
"""Run the frozen P5 Mean DR vs Attention DR 3-seed comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


METHODS = ("mean", "attention")
SEEDS = (17, 29, 43)


def complete(root: Path, method: str, seed: int) -> bool:
    return (
        root / method / f"seed_{seed:04d}" / "p5_result.json"
    ).is_file()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/p5_runs/mean_vs_attention"),
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print("=" * 108)
    print("P5 MEAN DR vs ATTENTION DR EXECUTION PLAN")
    print("=" * 108)
    print(f"methods: {METHODS}")
    print(f"seeds:   {SEEDS}")
    print("total:   2 x 3 = 6 runs")
    print("generic_test: SEALED")
    print("-" * 108)

    for method in METHODS:
        for seed in SEEDS:
            if complete(args.run_root, method, seed):
                print(f"{method} seed={seed}: already_complete")
                continue
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_p5_dr.py",
                    "--method", method,
                    "--seed", str(seed),
                    "--device", args.device,
                    "--run-root", str(args.run_root),
                ],
                check=True,
            )

    print("-" * 108)
    print("P5 6-RUN EXECUTION: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
