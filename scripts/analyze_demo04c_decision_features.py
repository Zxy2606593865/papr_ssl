#!/usr/bin/env python
"""Demo-04C — full feature diagnostics for Wanghao Demo-04A.

Purpose:
- explain the two known REJECTs and one unknown false ACCEPT
- inspect all frozen H5/H6 decision features, including duration_log_ratio
- decompose C/W/U logits into per-feature contributions
- compare failures with successful known / rejected unknown samples

This is diagnostic only:
- no training
- no threshold fitting
- no representation change
- generic_test remains sealed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from papr_ssl.inference.h6_personalized_runtime import PersonalizedRuntime, UserMemory
from papr_ssl.inference.h6_raw_wav_adapter import RawWavFeatureAdapter, resolve_selected_checkpoint


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p25": pct(values, 25),
        "p75": pct(values, 75),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def classify_group(row: dict) -> str:
    if row["kind"] == "known":
        if row.get("correct_accept", False):
            return "KNOWN_CORRECT_ACCEPT"
        if row["runtime_status"] == "REJECT":
            return "KNOWN_REJECT"
        if row["runtime_status"] == "CONFIRM":
            return "KNOWN_CONFIRM"
        return "KNOWN_OTHER"
    else:
        if row["runtime_status"] == "REJECT":
            return "UNKNOWN_REJECT"
        if row["runtime_status"] == "ACCEPT":
            return "UNKNOWN_FALSE_ACCEPT"
        return "UNKNOWN_CONFIRM"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime/predictions.jsonl"),
    )
    p.add_argument(
        "--memory",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime/wanghao_user_memory.pt"),
    )
    p.add_argument(
        "--head-json",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head/cwu_head.json"),
    )
    p.add_argument(
        "--policy-json",
        type=Path,
        default=Path("artifacts/h6_01_three_state_policy/h6_01_eval.json"),
    )
    p.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path("artifacts/p6_teacher_256_15shot/embeddings/manifest.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/demo_04c_wanghao_feature_diagnostics"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(args.predictions)
    memory = UserMemory.load(args.memory)
    runtime = PersonalizedRuntime(
        head_json=args.head_json,
        policy_json=args.policy_json,
    )

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(
        checkpoint=checkpoint,
        device=args.device,
    )

    feature_names = list(runtime.head.feature_names)
    if len(feature_names) != len(runtime.head.mean):
        raise RuntimeError("feature_names / standardization vector mismatch")

    out_rows = []
    contribution_rows = []

    print("=" * 118)
    print("DEMO-04C WANGHAO DECISION-FEATURE DIAGNOSTICS")
    print("=" * 118)
    print(f"queries:                  {len(predictions)}")
    print(f"features:                 {len(feature_names)}")
    print(f"device:                   {args.device}")
    print(f"feature names:            {feature_names}")
    print("-" * 118)

    for idx, pred in enumerate(predictions, 1):
        wav = args.dataset_root / pred["audio_path"]
        feat_out = adapter.extract_wav(wav)

        feature, candidates = runtime._candidate_evidence(
            memory=memory,
            query_global=feat_out["global_embedding"],
            query_temporal=feat_out["temporal_sequence"],
        )

        z = (feature - runtime.head.mean) / runtime.head.std
        logits = runtime.head.weight @ z + runtime.head.bias

        group = classify_group(pred)
        true_intent = (
            pred.get("true_intent_id")
            if pred["kind"] == "known"
            else pred.get("ground_truth_unregistered_intent_id")
        )
        top1 = candidates[0]
        true_rank = ""
        if pred["kind"] == "known":
            for c in candidates:
                if c["intent_id"] == true_intent:
                    true_rank = c["rank"]
                    break

        row = {
            "query_index": idx,
            "group": group,
            "kind": pred["kind"],
            "runtime_status": pred["runtime_status"],
            "true_or_unknown_intent": true_intent,
            "predicted_intent_id": pred.get("predicted_intent_id"),
            "top1_intent": top1["intent_id"],
            "true_intent_rank_in_top3": true_rank,
            "audio_path": pred["audio_path"],
            "source_file": pred["source_file"],
            "temporal_frames": int(feat_out["temporal_frames"]),
            "hidden_frames": int(feat_out["hidden_frames"]),
            "decision_C": float(pred["decision_scores"]["C"]),
            "decision_W": float(pred["decision_scores"]["W"]),
            "decision_U": float(pred["decision_scores"]["U"]),
        }
        for name, value, zv in zip(feature_names, feature, z):
            row[name] = float(value)
            row[f"z__{name}"] = float(zv)
        out_rows.append(row)

        for class_idx, cls in enumerate(("C", "W", "U")):
            for j, name in enumerate(feature_names):
                contribution_rows.append({
                    "query_index": idx,
                    "group": group,
                    "true_or_unknown_intent": true_intent,
                    "runtime_status": pred["runtime_status"],
                    "class": cls,
                    "feature": name,
                    "raw_value": float(feature[j]),
                    "z_value": float(z[j]),
                    "weight": float(runtime.head.weight[class_idx, j]),
                    "logit_contribution": float(runtime.head.weight[class_idx, j] * z[j]),
                })
            contribution_rows.append({
                "query_index": idx,
                "group": group,
                "true_or_unknown_intent": true_intent,
                "runtime_status": pred["runtime_status"],
                "class": cls,
                "feature": "__BIAS__",
                "raw_value": "",
                "z_value": "",
                "weight": "",
                "logit_contribution": float(runtime.head.bias[class_idx]),
            })

        if group in {"KNOWN_REJECT", "UNKNOWN_FALSE_ACCEPT"}:
            print(
                f"[{idx:02d}] {group:<22} "
                f"true={str(true_intent):<28} "
                f"top1={top1['intent_id']:<28} "
                f"C/W/U={pred['decision_scores']['C']:.3f}/"
                f"{pred['decision_scores']['W']:.3f}/"
                f"{pred['decision_scores']['U']:.3f}"
            )
            for name, value, zv in zip(feature_names, feature, z):
                print(f"     {name:<30} raw={value: .6f}  z={zv: .3f}")

    # Save query feature table.
    feature_csv = args.output_dir / "query_feature_diagnostics.csv"
    with feature_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    contrib_csv = args.output_dir / "failure_logit_contributions.csv"
    failure_indices = {
        r["query_index"]
        for r in out_rows
        if r["group"] in {"KNOWN_REJECT", "UNKNOWN_FALSE_ACCEPT"}
    }
    failure_contrib = [r for r in contribution_rows if r["query_index"] in failure_indices]
    with contrib_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(failure_contrib[0].keys()))
        writer.writeheader()
        writer.writerows(failure_contrib)

    # Group distributions of each feature.
    grouped = defaultdict(list)
    for r in out_rows:
        grouped[r["group"]].append(r)

    group_summary = {}
    for group, rr in grouped.items():
        group_summary[group] = {
            "count": len(rr),
            "decision_C": stats([float(r["decision_C"]) for r in rr]),
            "decision_U": stats([float(r["decision_U"]) for r in rr]),
            "features": {
                name: stats([float(r[name]) for r in rr])
                for name in feature_names
            },
        }

    # Feature-wise top standardized deviations for each failure.
    failure_details = []
    for r in out_rows:
        if r["group"] not in {"KNOWN_REJECT", "UNKNOWN_FALSE_ACCEPT"}:
            continue

        z_pairs = sorted(
            (
                (name, float(r[f"z__{name}"]), float(r[name]))
                for name in feature_names
            ),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        per_class_top_contrib = {}
        for cls in ("C", "W", "U"):
            rr = [
                x for x in failure_contrib
                if x["query_index"] == r["query_index"]
                and x["class"] == cls
                and x["feature"] != "__BIAS__"
            ]
            rr.sort(key=lambda x: abs(float(x["logit_contribution"])), reverse=True)
            per_class_top_contrib[cls] = rr[:5]

        failure_details.append({
            "query_index": r["query_index"],
            "group": r["group"],
            "true_or_unknown_intent": r["true_or_unknown_intent"],
            "top1_intent": r["top1_intent"],
            "runtime_status": r["runtime_status"],
            "most_out_of_distribution_features": [
                {
                    "feature": n,
                    "z": z,
                    "raw": raw,
                }
                for n, z, raw in z_pairs[:5]
            ],
            "top_logit_contributions": per_class_top_contrib,
        })

    summary = {
        "schema": "papr_ssl.demo_04c_feature_diagnostics.v1",
        "query_count": len(out_rows),
        "feature_names": feature_names,
        "group_summary": group_summary,
        "failure_details": failure_details,
        "training_performed": False,
        "threshold_fitting_performed": False,
        "representation_changed": False,
        "generic_test": "sealed_not_accessed",
        "status": "PASS",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 118)
    print("GROUP COUNTS")
    for group in sorted(grouped):
        print(f"{group:<28} {len(grouped[group])}")
    print("-" * 118)

    # If duration feature exists, print its distribution explicitly because
    # the prefix hard-negative hypothesis depends on it.
    duration_name = None
    for name in feature_names:
        if "duration" in name.lower() or "length" in name.lower():
            duration_name = name
            break

    if duration_name:
        print(f"DURATION/LENGTH FEATURE: {duration_name}")
        for group in sorted(grouped):
            s = group_summary[group]["features"][duration_name]
            print(
                f"{group:<28} "
                f"median={s['median']:.4f} "
                f"P25={s['p25']:.4f} "
                f"P75={s['p75']:.4f}"
            )
        print("-" * 118)

    print(f"feature table:            {feature_csv}")
    print(f"failure contributions:    {contrib_csv}")
    print(f"summary:                  {args.output_dir/'summary.json'}")
    print("threshold fitting:        NO")
    print("representation changed:   NO")
    print("generic_test accessed:    NO")
    print("DEMO-04C STATUS:          PASS")


if __name__ == "__main__":
    main()
