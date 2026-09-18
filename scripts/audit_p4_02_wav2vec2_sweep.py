#!/usr/bin/env python
"""Audit P4-02 Wav2Vec2 sweep completeness without selecting a best layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from papr_ssl.training.teacher.run_manifest import verify_run_manifest


LAYERS = tuple(range(13))
SEEDS = (17, 29, 43)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


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
    args = p.parse_args()

    rows = []
    for layer in LAYERS:
        for seed in SEEDS:
            if layer == 12:
                run = args.p3_layer12_root / f"seed_{seed:04d}"
                verify_run_manifest(run / "run_manifest.json")
                sel = read_json(run / "selected_checkpoint.json")
                baseline = read_json(
                    run / "untrained_projection_baseline.json"
                )
                selected = sel["selected"]
                score = float(selected["generic_dev_score"])
                baseline_score = float(
                    baseline["dev"]["generic_dev_score"]
                )
                source = "reused_p3_identical_condition"
            else:
                run = (
                    args.sweep_root
                    / f"layer_{layer:02d}"
                    / f"seed_{seed:04d}"
                )
                verify_run_manifest(run / "run_manifest.json")
                result = read_json(run / "p4_result.json")
                if result.get("generic_test_accessed") is not False:
                    raise RuntimeError(
                        f"test access violation: layer={layer}, seed={seed}"
                    )
                score = float(result["selected_generic_dev_score"])
                baseline_score = float(
                    result["untrained_projection_generic_dev_score"]
                )
                selected = read_json(
                    run / "selected_checkpoint.json"
                )["selected"]
                source = "p4_02"

            rows.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "selected_generic_dev_score": score,
                    "baseline_generic_dev_score": baseline_score,
                    "selected_epoch": int(selected["epoch"]),
                    "source": source,
                }
            )

    if len(rows) != 39:
        raise RuntimeError(f"expected 39 layer-seed rows, got {len(rows)}")

    out = args.sweep_root / "p4_02_raw_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "papr_ssl.p4_02_wav2vec2_raw_results.v1",
                "phase": "P4-02",
                "backbone": "wav2vec2_base",
                "hidden_state_indices": list(LAYERS),
                "seeds": list(SEEDS),
                "selection_metric": "generic_dev_score",
                "rows": rows,
                "ranking_performed": False,
                "best_layer_selected": False,
                "generic_test": "sealed_not_accessed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print("P4-02 WAV2VEC2 SWEEP COMPLETENESS AUDIT")
    print("=" * 104)
    print(f"layer count:              {len(LAYERS)}")
    print(f"seed count/layer:         {len(SEEDS)}")
    print(f"total layer-seed results: {len(rows)}")
    print("layer12 policy:           reused identical P3 condition")
    print("ranking performed:        NO")
    print("best layer selected:      NO")
    print("generic_test accessed:    NO")
    print("-" * 104)
    print("P4-02 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
