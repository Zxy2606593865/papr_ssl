#!/usr/bin/env python
"""P4-03 sequential WavLM hidden-state sweep.

No layer ranking occurs here. P4-05/P4-06 own aggregation and selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


LAYERS = tuple(range(25))
SEEDS = (17, 29, 43)


def run_complete(root: Path, layer: int, seed: int) -> bool:
    run = root / f"layer_{layer:02d}" / f"seed_{seed:04d}"
    return all(
        (run / name).is_file()
        for name in (
            "p4_result.json",
            "selected_checkpoint.json",
            "run_manifest.json",
        )
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("artifacts/p4_runs/wavlm_large"),
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    plan = []
    for layer in LAYERS:
        for seed in SEEDS:
            action = (
                "already_complete"
                if run_complete(args.sweep_root, layer, seed)
                else "run"
            )
            plan.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "action": action,
                }
            )

    print("=" * 104)
    print("P4-03 WAVLM-LARGE LAYER SWEEP EXECUTION PLAN")
    print("=" * 104)
    print(f"layers:       {LAYERS[0]}..{LAYERS[-1]}")
    print(f"seeds:        {SEEDS}")
    print(
        "new runs:     "
        f"{sum(x['action'] == 'run' for x in plan)}"
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
                "already_complete"
            )
            continue

        cmd = [
            sys.executable,
            "scripts/run_p4_03_wavlm_layer_seed.py",
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
    (args.sweep_root / "p4_03_execution_plan.json").write_text(
        json.dumps(
            {
                "schema": "papr_ssl.p4_03_execution_plan.v1",
                "layers": list(LAYERS),
                "seeds": list(SEEDS),
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
    print("P4-03 WAVLM SWEEP EXECUTION: COMPLETE")
    print("No best layer selected here; selection is deferred to P4-05/P4-06.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
