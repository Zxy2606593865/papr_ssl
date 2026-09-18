#!/usr/bin/env python
"""P4-02 sequential Wav2Vec2 layer sweep orchestrator.

No ranking is performed here. P4-05/P4-06 own aggregation and selection.
Layer 12 may reuse the already completed P3 scientific runs because its data,
head, SCAF, optimizer, budget, sampler and seeds are identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


LAYERS = tuple(range(13))
SEEDS = (17, 29, 43)


def p4_run_complete(root: Path, layer: int, seed: int) -> bool:
    run = root / f"layer_{layer:02d}" / f"seed_{seed:04d}"
    return all(
        (run / name).is_file()
        for name in (
            "p4_result.json",
            "selected_checkpoint.json",
            "run_manifest.json",
        )
    )


def p3_layer12_complete(root: Path, seed: int) -> bool:
    run = root / f"seed_{seed:04d}"
    return all(
        (run / name).is_file()
        for name in (
            "selected_checkpoint.json",
            "run_manifest.json",
            "baseline_vs_trained.json",
        )
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("artifacts/p4_runs/wav2vec2_base"),
    )
    p.add_argument(
        "--p3-layer12-root",
        type=Path,
        default=Path(
            "artifacts/p3_runs/"
            "mdsc_core30__wav2vec2_base__layer12"
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--include-layer12-rerun",
        action="store_true",
        help=(
            "Normally P4 reuses the already valid P3 layer12 three-seed "
            "condition. Set this only if a separate duplicate P4 run is desired."
        ),
    )
    args = p.parse_args()

    plan = []
    for layer in LAYERS:
        for seed in SEEDS:
            if layer == 12 and not args.include_layer12_rerun:
                if not p3_layer12_complete(args.p3_layer12_root, seed):
                    raise FileNotFoundError(
                        f"P3 layer12 seed {seed} is not complete: "
                        f"{args.p3_layer12_root}"
                    )
                plan.append(
                    {
                        "layer": layer,
                        "seed": seed,
                        "action": "reuse_p3",
                    }
                )
                continue

            if p4_run_complete(args.sweep_root, layer, seed):
                plan.append(
                    {
                        "layer": layer,
                        "seed": seed,
                        "action": "already_complete",
                    }
                )
                continue

            plan.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "action": "run",
                }
            )

    print("=" * 104)
    print("P4-02 WAV2VEC2 LAYER SWEEP EXECUTION PLAN")
    print("=" * 104)
    print(f"layers:       {LAYERS[0]}..{LAYERS[-1]}")
    print(f"seeds:        {SEEDS}")
    print(
        "P3 layer12:   "
        + (
            "rerun requested"
            if args.include_layer12_rerun
            else "reuse existing three-seed evidence"
        )
    )
    print(
        "new runs:     "
        f"{sum(x['action'] == 'run' for x in plan)}"
    )
    print(
        "reuse P3:     "
        f"{sum(x['action'] == 'reuse_p3' for x in plan)}"
    )
    print(
        "already done: "
        f"{sum(x['action'] == 'already_complete' for x in plan)}"
    )
    print("-" * 104)

    for item in plan:
        if item["action"] != "run":
            print(
                f"layer={item['layer']:02d} seed={item['seed']:02d} "
                f"{item['action']}"
            )
            continue

        cmd = [
            sys.executable,
            "scripts/run_p4_02_wav2vec2_layer_seed.py",
            "--layer",
            str(item["layer"]),
            "--seed",
            str(item["seed"]),
            "--device",
            args.device,
            "--sweep-root",
            str(args.sweep_root),
        ]
        print(
            f">>> layer={item['layer']:02d} seed={item['seed']:02d}"
        )
        subprocess.run(cmd, check=True)

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.sweep_root / "p4_02_execution_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "papr_ssl.p4_02_execution_plan.v1",
                "layers": list(LAYERS),
                "seeds": list(SEEDS),
                "p3_layer12_root": args.p3_layer12_root.as_posix(),
                "layer12_policy": (
                    "rerun"
                    if args.include_layer12_rerun
                    else "reuse_identical_p3_condition"
                ),
                "plan": plan,
                "generic_test": "sealed_not_accessed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("-" * 104)
    print("P4-02 WAV2VEC2 SWEEP EXECUTION: COMPLETE")
    print("No best layer selected here; selection is deferred to P4-05/P4-06.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
