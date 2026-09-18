#!/usr/bin/env python
"""Demo-04D — signed duration-direction audit.

The frozen H6 feature uses:
    abs(log(query_temporal_frames / median_support_temporal_frames))

That discards whether the query is shorter or longer than the enrolled phrase.

This audit adds NO runtime rule and performs NO fitting. It only measures:
    signed_log_ratio = log(T_query / median(T_support))
    query_support_ratio = T_query / median(T_support)

on the corrected Wanghao v2 dataset.

No training, no threshold fitting, no generic_test access.
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

from papr_ssl.inference.h6_personalized_runtime import UserMemory
from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_of(row: dict) -> str:
    if row["kind"] == "known":
        if row.get("correct_accept", False):
            return "KNOWN_CORRECT_ACCEPT"
        if row["runtime_status"] == "REJECT":
            return "KNOWN_REJECT"
        if row["runtime_status"] == "CONFIRM":
            return "KNOWN_CONFIRM"
        if row.get("wrong_accept", False):
            return "KNOWN_WRONG_ACCEPT"
        return "KNOWN_OTHER"

    if row["runtime_status"] == "REJECT":
        return "UNKNOWN_REJECT"
    if row["runtime_status"] == "ACCEPT":
        return "UNKNOWN_FALSE_ACCEPT"
    return "UNKNOWN_CONFIRM"


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Corrected wanghao_demo_v2 export root",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime_v2/predictions.jsonl"),
    )
    p.add_argument(
        "--memory",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime_v2/wanghao_user_memory.pt"),
    )
    p.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path("artifacts/p6_teacher_256_15shot/embeddings/manifest.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/demo_04d_signed_duration_audit"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preds = read_jsonl(args.predictions)
    memory = UserMemory.load(args.memory)

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(
        checkpoint=checkpoint,
        device=args.device,
    )

    rows = []

    print("=" * 116)
    print("DEMO-04D SIGNED DURATION-DIRECTION AUDIT")
    print("=" * 116)
    print(f"dataset:                  {args.dataset_root}")
    print(f"queries:                  {len(preds)}")
    print(f"device:                   {args.device}")
    print("-" * 116)

    for idx, pred in enumerate(preds, 1):
        candidates = pred.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidates in prediction row {idx}")

        top1_intent = candidates[0]["intent_id"]
        if top1_intent not in memory.intents:
            raise KeyError(f"Top1 intent missing from memory: {top1_intent}")

        support_frames = [
            len(ex.temporal_sequence)
            for ex in memory.intents[top1_intent].examples
        ]
        support_median = float(np.median(support_frames))

        wav = args.dataset_root / pred["audio_path"]
        feat = adapter.extract_wav(wav)
        query_frames = int(feat["temporal_frames"])

        signed = math.log(max(query_frames, 1) / max(support_median, 1.0))
        abs_ratio = abs(signed)
        linear_ratio = query_frames / support_median

        row = {
            "query_index": idx,
            "group": group_of(pred),
            "kind": pred["kind"],
            "runtime_status": pred["runtime_status"],
            "true_or_unknown_intent": (
                pred.get("true_intent_id")
                if pred["kind"] == "known"
                else pred.get("ground_truth_unregistered_intent_id")
            ),
            "top1_intent": top1_intent,
            "audio_path": pred["audio_path"],
            "query_temporal_frames": query_frames,
            "support_frame_1": support_frames[0],
            "support_frame_2": support_frames[1] if len(support_frames) > 1 else "",
            "support_median_frames": support_median,
            "query_support_ratio": linear_ratio,
            "signed_duration_log_ratio": signed,
            "abs_duration_log_ratio": abs_ratio,
            "query_shorter_than_support": bool(signed < 0),
        }
        rows.append(row)

    csv_path = args.output_dir / "signed_duration_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["group"]].append(r)

    group_summary = {}
    for group, rr in grouped.items():
        group_summary[group] = {
            "count": len(rr),
            "signed_duration_log_ratio": summary(
                [float(x["signed_duration_log_ratio"]) for x in rr]
            ),
            "query_support_ratio": summary(
                [float(x["query_support_ratio"]) for x in rr]
            ),
            "shorter_count": sum(bool(x["query_shorter_than_support"]) for x in rr),
        }

    false_accepts = grouped.get("UNKNOWN_FALSE_ACCEPT", [])
    known_correct = grouped.get("KNOWN_CORRECT_ACCEPT", [])

    prefix_analysis = []
    for fa in false_accepts:
        signed = float(fa["signed_duration_log_ratio"])
        abs_v = float(fa["abs_duration_log_ratio"])

        if signed < 0:
            same_or_more_short_known = [
                x for x in known_correct
                if float(x["signed_duration_log_ratio"]) <= signed
            ]
        else:
            same_or_more_short_known = []

        abs_as_or_more_extreme_known = [
            x for x in known_correct
            if float(x["abs_duration_log_ratio"]) >= abs_v
        ]

        prefix_analysis.append({
            "query_index": fa["query_index"],
            "true_unknown_intent": fa["true_or_unknown_intent"],
            "top1_intent": fa["top1_intent"],
            "query_support_ratio": float(fa["query_support_ratio"]),
            "signed_duration_log_ratio": signed,
            "abs_duration_log_ratio": abs_v,
            "known_correct_count": len(known_correct),
            "known_correct_same_or_more_short_count": len(same_or_more_short_known),
            "known_correct_abs_as_or_more_extreme_count": len(abs_as_or_more_extreme_known),
        })

    output = {
        "schema": "papr_ssl.demo_04d_signed_duration_audit.v1",
        "query_count": len(rows),
        "group_summary": group_summary,
        "unknown_false_accept_direction_analysis": prefix_analysis,
        "interpretation_rule": (
            "This is descriptive only. No duration threshold or production guard "
            "is selected from this Wanghao sample."
        ),
        "training_performed": False,
        "threshold_fitting_performed": False,
        "runtime_policy_changed": False,
        "generic_test": "sealed_not_accessed",
        "status": "PASS",
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("GROUP DISTRIBUTIONS")
    for group in sorted(grouped):
        s = group_summary[group]
        ss = s["signed_duration_log_ratio"]
        rr = s["query_support_ratio"]
        print("-" * 116)
        print(f"{group} | N={s['count']} | shorter={s['shorter_count']}/{s['count']}")
        print(
            f"  signed log ratio: min={ss['min']:.4f} "
            f"P25={ss['p25']:.4f} median={ss['median']:.4f} "
            f"P75={ss['p75']:.4f} max={ss['max']:.4f}"
        )
        print(
            f"  linear Q/S ratio: min={rr['min']:.3f} "
            f"P25={rr['p25']:.3f} median={rr['median']:.3f} "
            f"P75={rr['p75']:.3f} max={rr['max']:.3f}"
        )

    if prefix_analysis:
        print("-" * 116)
        print("UNKNOWN FALSE ACCEPT DIRECTION CHECK")
        for x in prefix_analysis:
            print(
                f"unknown={x['true_unknown_intent']} -> top1={x['top1_intent']} | "
                f"Q/S={x['query_support_ratio']:.3f} | "
                f"signed_log={x['signed_duration_log_ratio']:.4f} | "
                f"abs_log={x['abs_duration_log_ratio']:.4f}"
            )
            print(
                f"  known correct equally/more short: "
                f"{x['known_correct_same_or_more_short_count']}/"
                f"{x['known_correct_count']}"
            )
            print(
                f"  known correct abs equally/more extreme: "
                f"{x['known_correct_abs_as_or_more_extreme_count']}/"
                f"{x['known_correct_count']}"
            )

    print("-" * 116)
    print(f"csv:                      {csv_path}")
    print(f"summary:                  {args.output_dir/'summary.json'}")
    print("threshold fitting:        NO")
    print("runtime policy changed:   NO")
    print("generic_test accessed:    NO")
    print("DEMO-04D STATUS:          PASS")


if __name__ == "__main__":
    main()
