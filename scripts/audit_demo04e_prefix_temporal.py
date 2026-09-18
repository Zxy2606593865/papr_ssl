#!/usr/bin/env python
"""Demo-04E — prefix/endpoint temporal audit for Wanghao v2.

Question:
Can the hard negative "大家好" -> registered "大家好 我叫王灏"
be distinguished from legitimately short known queries by temporal shape,
not just duration?

Diagnostic only:
- no training
- no threshold fitting
- no runtime rule
- no representation change
- generic_test remains sealed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from papr_ssl.inference.h6_personalized_runtime import (
    UserMemory,
    _dtw_distance,
    _l2,
)
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


def cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).mean(axis=0)
    bb = np.asarray(b, dtype=np.float64).mean(axis=0)
    aa = aa / max(np.linalg.norm(aa), 1e-12)
    bb = bb / max(np.linalg.norm(bb), 1e-12)
    return float(np.clip(aa @ bb, -1.0, 1.0))


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": len(vals),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
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
        default=Path("artifacts/demo_04e_prefix_temporal_audit"),
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

    print("=" * 118)
    print("DEMO-04E PREFIX / ENDPOINT TEMPORAL AUDIT")
    print("=" * 118)
    print(f"queries:                  {len(preds)}")
    print(f"device:                   {args.device}")
    print("-" * 118)

    for idx, pred in enumerate(preds, start=1):
        candidates = pred.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidate evidence for query {idx}")

        top1_intent = candidates[0]["intent_id"]
        intent_mem = memory.intents[top1_intent]

        wav = args.dataset_root / pred["audio_path"]
        feat = adapter.extract_wav(wav)
        q = _l2(np.asarray(feat["temporal_sequence"], dtype=np.float64))

        # Runtime DTW uses the best of the two enrollment templates.
        template_infos = []
        for template_idx, ex in enumerate(intent_mem.examples, start=1):
            s = _l2(np.asarray(ex.temporal_sequence, dtype=np.float64))
            full_sim = 1.0 - 0.5 * _dtw_distance(q, s)

            qn, sn = len(q), len(s)
            win = max(1, min(qn, sn))
            prefix = s[:win]
            suffix = s[-win:]

            prefix_sim = 1.0 - 0.5 * _dtw_distance(q, prefix)
            suffix_sim = 1.0 - 0.5 * _dtw_distance(q, suffix)

            k = max(1, min(5, int(round(0.20 * min(qn, sn)))))
            start_cos = cosine_mean(q[:k], s[:k])
            end_cos = cosine_mean(q[-k:], s[-k:])

            template_infos.append({
                "template_idx": template_idx,
                "support_frames": sn,
                "full_dtw_sim": full_sim,
                "prefix_dtw_sim": prefix_sim,
                "suffix_dtw_sim": suffix_sim,
                "prefix_minus_suffix": prefix_sim - suffix_sim,
                "prefix_minus_full": prefix_sim - full_sim,
                "start_cosine": start_cos,
                "end_cosine": end_cos,
                "start_minus_end": start_cos - end_cos,
                "endpoint_window_frames": k,
            })

        # Match the frozen runtime: choose the support template with max full DTW sim.
        best = max(template_infos, key=lambda x: x["full_dtw_sim"])
        support_frames = best["support_frames"]
        q_support_ratio = len(q) / support_frames
        signed_log_ratio = math.log(max(len(q), 1) / max(support_frames, 1))

        true_intent = (
            pred.get("true_intent_id")
            if pred["kind"] == "known"
            else pred.get("ground_truth_unregistered_intent_id")
        )

        row = {
            "query_index": idx,
            "group": group_of(pred),
            "kind": pred["kind"],
            "runtime_status": pred["runtime_status"],
            "true_or_unknown_intent": true_intent,
            "top1_intent": top1_intent,
            "audio_path": pred["audio_path"],
            "query_frames": len(q),
            "best_template_idx": best["template_idx"],
            "support_frames": support_frames,
            "query_support_ratio": q_support_ratio,
            "signed_duration_log_ratio": signed_log_ratio,
            "full_dtw_sim": best["full_dtw_sim"],
            "prefix_dtw_sim": best["prefix_dtw_sim"],
            "suffix_dtw_sim": best["suffix_dtw_sim"],
            "prefix_minus_suffix": best["prefix_minus_suffix"],
            "prefix_minus_full": best["prefix_minus_full"],
            "start_cosine": best["start_cosine"],
            "end_cosine": best["end_cosine"],
            "start_minus_end": best["start_minus_end"],
            "endpoint_window_frames": best["endpoint_window_frames"],
        }
        rows.append(row)

    csv_path = args.output_dir / "prefix_temporal_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["group"]].append(r)

    diagnostic_names = [
        "query_support_ratio",
        "full_dtw_sim",
        "prefix_dtw_sim",
        "suffix_dtw_sim",
        "prefix_minus_suffix",
        "start_cosine",
        "end_cosine",
        "start_minus_end",
    ]

    group_summary = {}
    for group, rr in grouped.items():
        group_summary[group] = {
            "count": len(rr),
            "metrics": {
                name: summarize([float(x[name]) for x in rr])
                for name in diagnostic_names
            },
        }

    false_accepts = grouped.get("UNKNOWN_FALSE_ACCEPT", [])
    known_correct = grouped.get("KNOWN_CORRECT_ACCEPT", [])
    hard_negative_analysis = []

    for fa in false_accepts:
        ratio = float(fa["query_support_ratio"])
        short_controls = [
            x for x in known_correct
            if float(x["query_support_ratio"]) <= ratio
        ]

        # Compare prefix signatures descriptively against equally/more-short
        # correct-known controls.
        hard_negative_analysis.append({
            "query_index": fa["query_index"],
            "true_unknown_intent": fa["true_or_unknown_intent"],
            "top1_intent": fa["top1_intent"],
            "query_support_ratio": ratio,
            "full_dtw_sim": float(fa["full_dtw_sim"]),
            "prefix_dtw_sim": float(fa["prefix_dtw_sim"]),
            "suffix_dtw_sim": float(fa["suffix_dtw_sim"]),
            "prefix_minus_suffix": float(fa["prefix_minus_suffix"]),
            "start_cosine": float(fa["start_cosine"]),
            "end_cosine": float(fa["end_cosine"]),
            "start_minus_end": float(fa["start_minus_end"]),
            "short_known_control_count": len(short_controls),
            "short_known_controls": [
                {
                    "true_intent": x["true_or_unknown_intent"],
                    "query_support_ratio": float(x["query_support_ratio"]),
                    "prefix_minus_suffix": float(x["prefix_minus_suffix"]),
                    "start_minus_end": float(x["start_minus_end"]),
                    "full_dtw_sim": float(x["full_dtw_sim"]),
                    "audio_path": x["audio_path"],
                }
                for x in sorted(
                    short_controls,
                    key=lambda z: float(z["query_support_ratio"])
                )
            ],
        })

    output = {
        "schema": "papr_ssl.demo_04e_prefix_temporal_audit.v1",
        "query_count": len(rows),
        "group_summary": group_summary,
        "hard_negative_analysis": hard_negative_analysis,
        "notes": [
            "prefix/suffix DTW is diagnostic only; it is not a production feature yet",
            "the selected support template is the same template that maximizes frozen full DTW similarity",
            "no threshold is selected from this single-user sample",
        ],
        "training_performed": False,
        "threshold_fitting_performed": False,
        "runtime_policy_changed": False,
        "representation_changed": False,
        "generic_test": "sealed_not_accessed",
        "status": "PASS",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("GROUP SUMMARY")
    for group in sorted(grouped):
        s = group_summary[group]["metrics"]
        print("-" * 118)
        print(f"{group} | N={group_summary[group]['count']}")
        print(
            "  Q/S ratio:           "
            f"median={s['query_support_ratio']['median']:.3f} "
            f"min={s['query_support_ratio']['min']:.3f} "
            f"max={s['query_support_ratio']['max']:.3f}"
        )
        print(
            "  prefix-suffix DTW:   "
            f"median={s['prefix_minus_suffix']['median']:.4f} "
            f"min={s['prefix_minus_suffix']['min']:.4f} "
            f"max={s['prefix_minus_suffix']['max']:.4f}"
        )
        print(
            "  start-end cosine:    "
            f"median={s['start_minus_end']['median']:.4f} "
            f"min={s['start_minus_end']['min']:.4f} "
            f"max={s['start_minus_end']['max']:.4f}"
        )

    if hard_negative_analysis:
        print("-" * 118)
        print("PREFIX HARD-NEGATIVE CHECK")
        for x in hard_negative_analysis:
            print(
                f"unknown={x['true_unknown_intent']} -> top1={x['top1_intent']}"
            )
            print(
                f"  Q/S={x['query_support_ratio']:.3f} "
                f"full={x['full_dtw_sim']:.4f} "
                f"prefix={x['prefix_dtw_sim']:.4f} "
                f"suffix={x['suffix_dtw_sim']:.4f}"
            )
            print(
                f"  prefix-suffix={x['prefix_minus_suffix']:.4f} "
                f"start-end={x['start_minus_end']:.4f}"
            )
            print(
                f"  equally/more-short known controls: "
                f"{x['short_known_control_count']}"
            )
            for c in x["short_known_controls"]:
                print(
                    f"    known={c['true_intent']:<28} "
                    f"Q/S={c['query_support_ratio']:.3f} "
                    f"prefix-suffix={c['prefix_minus_suffix']:.4f} "
                    f"start-end={c['start_minus_end']:.4f} "
                    f"full={c['full_dtw_sim']:.4f}"
                )

    print("-" * 118)
    print(f"csv:                      {csv_path}")
    print(f"summary:                  {args.output_dir/'summary.json'}")
    print("threshold fitting:        NO")
    print("runtime policy changed:   NO")
    print("representation changed:   NO")
    print("generic_test accessed:    NO")
    print("DEMO-04E STATUS:          PASS")


if __name__ == "__main__":
    main()
