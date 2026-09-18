#!/usr/bin/env python
"""Demo-04B — error analysis for Wanghao Demo-04A.

Read-only analysis:
- no training
- no threshold fitting
- no model rerun
- no generic_test access

It extracts:
1. known queries that were not correct ACCEPT
2. unknown queries that were ACCEPTed
3. candidate/ranking evidence already saved by Demo-04A
4. copies only the failing WAVs into a tiny listening folder
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_scores(scores) -> str:
    if scores is None:
        return ""
    if isinstance(scores, dict):
        return " | ".join(f"{k}={float(v):.6f}" for k, v in scores.items())
    return json.dumps(scores, ensure_ascii=False)


def candidate_brief(candidates) -> str:
    if not candidates:
        return ""
    out = []
    for i, c in enumerate(candidates[:5], start=1):
        if isinstance(c, dict):
            intent = (
                c.get("intent_id")
                or c.get("candidate_intent_id")
                or c.get("label")
                or c.get("class_id")
                or "?"
            )
            numeric = []
            for k, v in c.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric.append(f"{k}={float(v):.5f}")
            out.append(f"#{i}:{intent}" + (f"({','.join(numeric)})" if numeric else ""))
        else:
            out.append(f"#{i}:{c}")
    return " || ".join(out)


def top_candidate_intent(candidates):
    if not candidates:
        return None
    c = candidates[0]
    if not isinstance(c, dict):
        return str(c)
    return (
        c.get("intent_id")
        or c.get("candidate_intent_id")
        or c.get("label")
        or c.get("class_id")
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime/predictions.jsonl"),
    )
    p.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime/result.json"),
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="papr_audio_toolkit/data/exports/wanghao_demo",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/demo_04b_wanghao_failure_analysis"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.predictions)
    result = json.loads(args.result.read_text(encoding="utf-8"))

    failures = []

    for row in rows:
        kind = row["kind"]
        is_failure = False
        failure_type = ""

        if kind == "known":
            if not bool(row.get("correct_accept", False)):
                is_failure = True
                if row["runtime_status"] == "REJECT":
                    failure_type = "KNOWN_REJECT"
                elif row["runtime_status"] == "CONFIRM":
                    failure_type = "KNOWN_CONFIRM"
                elif bool(row.get("wrong_accept", False)):
                    failure_type = "KNOWN_WRONG_ACCEPT"
                else:
                    failure_type = "KNOWN_OTHER"
        elif kind == "unknown":
            if row["runtime_status"] == "ACCEPT":
                is_failure = True
                failure_type = "UNKNOWN_FALSE_ACCEPT"

        if not is_failure:
            continue

        candidates = row.get("candidates", [])
        top_intent = top_candidate_intent(candidates)

        true_intent = (
            row.get("true_intent_id")
            if kind == "known"
            else row.get("ground_truth_unregistered_intent_id")
        )

        ranking_correct = (
            kind == "known"
            and top_intent is not None
            and top_intent == true_intent
        )

        rec = {
            "failure_type": failure_type,
            "kind": kind,
            "audio_path": row["audio_path"],
            "segment_id": row["segment_id"],
            "source_file": row["source_file"],
            "true_or_unknown_intent": true_intent,
            "runtime_status": row["runtime_status"],
            "predicted_intent_id": row.get("predicted_intent_id"),
            "top_candidate_intent": top_intent,
            "top1_ranking_correct_for_known": ranking_correct if kind == "known" else "",
            "decision_scores": fmt_scores(row.get("decision_scores")),
            "candidates": candidate_brief(candidates),
            "latency_ms": row.get("latency_ms"),
            "listening_note": "",
        }
        failures.append(rec)

    review_dir = args.output_dir / "failure_wavs"
    review_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(failures, start=1):
        src = args.dataset_root / rec["audio_path"]
        if not src.exists():
            raise FileNotFoundError(src)
        dst = review_dir / f"{i:02d}_{rec['failure_type']}_{src.name}"
        shutil.copy2(src, dst)
        rec["copied_wav"] = str(dst.relative_to(args.output_dir)).replace("\\", "/")

    fields = [
        "failure_type",
        "kind",
        "audio_path",
        "copied_wav",
        "segment_id",
        "source_file",
        "true_or_unknown_intent",
        "runtime_status",
        "predicted_intent_id",
        "top_candidate_intent",
        "top1_ranking_correct_for_known",
        "decision_scores",
        "candidates",
        "latency_ms",
        "listening_note",
    ]

    with (args.output_dir / "failure_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)

    known_fail = [r for r in failures if r["kind"] == "known"]
    unknown_fa = [r for r in failures if r["failure_type"] == "UNKNOWN_FALSE_ACCEPT"]

    known_rank_correct = sum(
        bool(r["top1_ranking_correct_for_known"])
        for r in known_fail
    )

    summary = {
        "schema": "papr_ssl.demo_04b_failure_analysis.v1",
        "demo04a_known_ca": result["known_summary"]["correct_accept"],
        "demo04a_known_wa": result["known_summary"]["wrong_accept"],
        "demo04a_unknown_accept": result["unknown_summary"]["accept"],
        "known_failure_count": len(known_fail),
        "known_failure_top1_ranking_correct_count": known_rank_correct,
        "unknown_false_accept_count": len(unknown_fa),
        "training_performed": False,
        "threshold_fitting_performed": False,
        "generic_test": "sealed_not_accessed",
        "status": "PASS",
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 118)
    print("DEMO-04B WANGHAO FAILURE ANALYSIS")
    print("=" * 118)
    print(f"known failures:           {len(known_fail)}")
    print(f"  top1 still correct:     {known_rank_correct}/{len(known_fail)}")
    print(f"unknown false accepts:    {len(unknown_fa)}")
    print("-" * 118)

    for i, r in enumerate(failures, start=1):
        print(
            f"[{i:02d}] {r['failure_type']:<22} "
            f"true={str(r['true_or_unknown_intent']):<28} "
            f"status={r['runtime_status']:<7} "
            f"pred={str(r['predicted_intent_id']):<28} "
            f"top1={str(r['top_candidate_intent'])}"
        )
        print(f"     scores: {r['decision_scores']}")
        print(f"     cand:   {r['candidates']}")

    print("-" * 118)
    print(f"review CSV:               {args.output_dir/'failure_review.csv'}")
    print(f"failure WAVs:             {review_dir}")
    print("training performed:       NO")
    print("threshold fitting:        NO")
    print("generic_test accessed:    NO")
    print("DEMO-04B STATUS:          PASS")


if __name__ == "__main__":
    main()
