#!/usr/bin/env python
"""Aggregate the completed 17/29/43 real Core30 Wav2Vec2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from papr_ssl.training.teacher.run_manifest import (
    verify_run_manifest,
    write_experiment_manifest,
)


SEEDS = (17, 29, 43)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stats(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": float(mean(values)),
        "sample_std": float(stdev(values)),
        "std_ddof": 1,
        "n": len(values),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(
            "artifacts/p3_runs/"
            "mdsc_core30__wav2vec2_base__layer12"
        ),
    )
    args = p.parse_args()

    rows = []
    fingerprints = set()

    for seed in SEEDS:
        run_dir = args.experiment_dir / f"seed_{seed:04d}"
        manifest = verify_run_manifest(run_dir / "run_manifest.json")
        fingerprints.add(manifest["experiment_fingerprint"])

        baseline = read_json(
            run_dir / "untrained_projection_baseline.json"
        )
        comparison = read_json(run_dir / "baseline_vs_trained.json")
        selection = read_json(run_dir / "selected_checkpoint.json")
        metrics = read_jsonl(run_dir / "metrics.jsonl")

        if len(metrics) != 20:
            raise RuntimeError(
                f"seed {seed}: expected 20 epochs, got {len(metrics)}"
            )
        if comparison.get("generic_test_accessed") is not False:
            raise RuntimeError(f"seed {seed}: generic test violation")
        if baseline.get("generic_test_accessed") is not False:
            raise RuntimeError(f"seed {seed}: baseline test violation")

        selected = selection["selected"]
        selected_epoch = int(selected["epoch"])
        selected_row = next(
            row for row in metrics if int(row["epoch"]) == selected_epoch
        )
        selected_dev = selected_row["dev"]

        rows.append(
            {
                "seed": seed,
                "baseline": float(
                    comparison["baseline_generic_dev_score"]
                ),
                "trained_selected": float(
                    comparison["trained_selected_generic_dev_score"]
                ),
                "delta": float(comparison["absolute_delta"]),
                "selected_epoch": selected_epoch,
                "final_epoch_dev": float(
                    metrics[-1]["dev"]["generic_dev_score"]
                ),
                "selected_collapsed": bool(
                    selected_dev["collapsed"]
                ),
                "selected_norm_max_abs_error": float(
                    selected_dev["embedding_norm_max_abs_error"]
                ),
                "checkpoint_sha256": selected["checkpoint_sha256"],
            }
        )

    if len(fingerprints) != 1:
        raise RuntimeError(
            "non-seed experimental condition changed across runs"
        )

    baselines = [row["baseline"] for row in rows]
    trained = [row["trained_selected"] for row in rows]
    deltas = [row["delta"] for row in rows]
    finals = [row["final_epoch_dev"] for row in rows]

    summary = {
        "schema": "papr_ssl.p3_real_multiseed_scientific_summary.v1",
        "task_view": "mdsc_core30_exact_phrase",
        "backbone": "facebook/wav2vec2-base",
        "layer": 12,
        "seeds": list(SEEDS),
        "runs": rows,
        "untrained_projection_generic_dev_score": stats(baselines),
        "selected_trained_generic_dev_score": stats(trained),
        "absolute_delta": stats(deltas),
        "final_epoch_generic_dev_score": stats(finals),
        "all_selected_noncollapsed": all(
            not row["selected_collapsed"] for row in rows
        ),
        "all_selected_unit_norm": all(
            row["selected_norm_max_abs_error"] <= 1e-5
            for row in rows
        ),
        "all_seed_deltas_positive": all(x > 0.0 for x in deltas),
        "generic_test": "sealed_not_accessed",
        "scientific_gate_decision": (
            "EVIDENCE_READY_FOR_REVIEW; "
            "do not infer a threshold for 'clearly beats baseline' automatically"
        ),
    }

    out = args.experiment_dir / "scientific_gate_summary.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # P3-08 experiment-level provenance; this also verifies the shared
    # experiment fingerprint across the three seeds.
    write_experiment_manifest(
        experiment_dir=args.experiment_dir,
        seeds=SEEDS,
    )

    print("=" * 104)
    print("PAPR-SSL P3 REAL MULTI-SEED SCIENTIFIC SUMMARY")
    print("=" * 104)
    for row in rows:
        print(
            f"seed {row['seed']:2d}: "
            f"baseline={row['baseline']:.6f}  "
            f"trained={row['trained_selected']:.6f}  "
            f"delta={row['delta']:+.6f}  "
            f"selected_epoch={row['selected_epoch']}"
        )
    print("-" * 104)
    print(
        "baseline mean ± sample std: "
        f"{summary['untrained_projection_generic_dev_score']['mean']:.6f} "
        f"± {summary['untrained_projection_generic_dev_score']['sample_std']:.6f}"
    )
    print(
        "trained mean ± sample std:  "
        f"{summary['selected_trained_generic_dev_score']['mean']:.6f} "
        f"± {summary['selected_trained_generic_dev_score']['sample_std']:.6f}"
    )
    print(
        "delta mean ± sample std:    "
        f"{summary['absolute_delta']['mean']:+.6f} "
        f"± {summary['absolute_delta']['sample_std']:.6f}"
    )
    print(
        "all selected noncollapsed:  "
        f"{summary['all_selected_noncollapsed']}"
    )
    print(
        "all selected unit norm:     "
        f"{summary['all_selected_unit_norm']}"
    )
    print("generic_test accessed:         NO")
    print("-" * 104)
    print("P3 MULTI-SEED EVIDENCE: READY FOR SCIENTIFIC REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
