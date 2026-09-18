#!/usr/bin/env python
"""Demo-04A — Evaluate frozen H6 runtime on Wanghao real WAV export.

Input dataset is produced by papr_audio_toolkit Demo-03B.

This script:
- enrolls the 5 registered intents from real WAVs
- runs all registered query WAVs
- runs all unknown query WAVs without enrolling them
- reports ACCEPT / CONFIRM / REJECT behavior
- saves a real UserMemory for later demo/API use

No training or threshold fitting is performed here.
generic_test is not accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from papr_ssl.inference.h6_personalized_runtime import (
    PersonalizedRuntime,
    UserMemory,
)
from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to papr_audio_toolkit/data/exports/wanghao_demo",
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
        "--parity-summary",
        type=Path,
        default=Path("artifacts/h6_03_raw_wav_parity/summary.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifest.jsonl"
    summary_path = dataset_root / "summary.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    # Keep the H6 raw-WAV parity gate intact.
    parity = json.loads(args.parity_summary.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS":
        raise RuntimeError(
            f"H6-03 raw-WAV parity is not PASS: {args.parity_summary}"
        )

    rows = read_jsonl(manifest_path)
    ds_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    enrollment_rows = [r for r in rows if r["role"] == "enrollment"]
    known_query_rows = [r for r in rows if r["role"] == "query"]
    unknown_query_rows = [r for r in rows if r["role"] == "unknown_query"]

    enrolled_by_intent = defaultdict(list)
    for r in enrollment_rows:
        enrolled_by_intent[r["intent_id"]].append(r)

    if not enrolled_by_intent:
        raise RuntimeError("No enrollment rows")

    shot_counts = {len(v) for v in enrolled_by_intent.values()}
    if shot_counts != {2}:
        raise RuntimeError(
            f"Frozen H6 Demo-04A expects exactly 2-shot per intent; got {sorted(shot_counts)}"
        )

    # Every query path must resolve inside the exported dataset.
    for row in rows:
        wav = dataset_root / row["audio_path"]
        if not wav.exists():
            raise FileNotFoundError(wav)

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(
        checkpoint=checkpoint,
        device=args.device,
    )
    runtime = PersonalizedRuntime(
        head_json=args.head_json,
        policy_json=args.policy_json,
    )

    memory = UserMemory(user_id="wanghao")

    print("=" * 112)
    print("DEMO-04A WANGHAO RAW-WAV PERSONALIZED RUNTIME")
    print("=" * 112)
    print(f"dataset:                  {dataset_root}")
    print(f"registered intents:       {len(enrolled_by_intent)}")
    print(f"shot:                     2")
    print(f"known queries:            {len(known_query_rows)}")
    print(f"unknown queries:          {len(unknown_query_rows)}")
    print(f"device:                   {args.device}")
    print("-" * 112)
    print("ENROLLMENT")

    # Enroll from raw WAV.
    for intent_id in sorted(enrolled_by_intent):
        items = enrolled_by_intent[intent_id]
        canonical_values = {r["canonical_text"] for r in items}
        if len(canonical_values) != 1:
            raise RuntimeError(
                f"canonical_text mismatch inside intent {intent_id}: {canonical_values}"
            )
        canonical_text = next(iter(canonical_values))

        for row in items:
            wav = dataset_root / row["audio_path"]
            feat = adapter.extract_wav(wav)
            memory.enroll(
                intent_id=intent_id,
                canonical_text=canonical_text,
                global_embedding=feat["global_embedding"],
                temporal_sequence=feat["temporal_sequence"],
            )

        print(
            f"  + {intent_id:<30} "
            f"2-shot | {canonical_text}"
        )

    # Verify frozen runtime shot contract before prediction.
    if memory.shot_count() != 2:
        raise RuntimeError("Unexpected memory shot count")

    memory_path = args.output_dir / "wanghao_user_memory.pt"
    memory.save(memory_path)
    memory = UserMemory.load(memory_path)

    prediction_rows = []
    t0 = time.perf_counter()

    def predict_one(row: dict, kind: str) -> dict:
        wav = dataset_root / row["audio_path"]
        t_start = time.perf_counter()
        feat = adapter.extract_wav(wav)
        pred = runtime.predict_feature(
            memory=memory,
            query_global=feat["global_embedding"],
            query_temporal=feat["temporal_sequence"],
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        rec = {
            "kind": kind,
            "audio_path": row["audio_path"],
            "segment_id": row["segment_id"],
            "source_file": row["source_file"],
            "runtime_status": pred["status"],
            "predicted_intent_id": pred["intent_id"],
            "predicted_canonical_text": pred["canonical_text"],
            "decision_scores": pred["decision_scores"],
            "scores_are_calibrated_probabilities": False,
            "latency_ms": latency_ms,
            "candidates": pred.get("candidates", []),
        }

        if kind == "known":
            rec.update(
                {
                    "true_intent_id": row["intent_id"],
                    "true_canonical_text": row["canonical_text"],
                    "query_source_disjoint_from_enrollment": bool(
                        row.get("query_source_disjoint_from_enrollment", False)
                    ),
                    "correct_accept": (
                        pred["status"] == "ACCEPT"
                        and pred["intent_id"] == row["intent_id"]
                    ),
                    "wrong_accept": (
                        pred["status"] == "ACCEPT"
                        and pred["intent_id"] != row["intent_id"]
                    ),
                }
            )
        else:
            rec.update(
                {
                    "ground_truth_unregistered_intent_id": row[
                        "ground_truth_intent_id"
                    ],
                    "ground_truth_canonical_text": row[
                        "ground_truth_canonical_text"
                    ],
                }
            )

        return rec

    print("-" * 112)
    print("QUERY")

    for i, row in enumerate(known_query_rows, 1):
        rec = predict_one(row, "known")
        prediction_rows.append(rec)
        print(
            f"[K {i:02d}/{len(known_query_rows):02d}] "
            f"{row['intent_id']:<30} -> "
            f"{rec['runtime_status']:<7} "
            f"{str(rec['predicted_intent_id']):<30}"
        )

    for i, row in enumerate(unknown_query_rows, 1):
        rec = predict_one(row, "unknown")
        prediction_rows.append(rec)
        print(
            f"[U {i:02d}/{len(unknown_query_rows):02d}] "
            f"{row['ground_truth_intent_id']:<30} -> "
            f"{rec['runtime_status']:<7} "
            f"{str(rec['predicted_intent_id']):<30}"
        )

    known = [r for r in prediction_rows if r["kind"] == "known"]
    unknown = [r for r in prediction_rows if r["kind"] == "unknown"]

    known_ca = sum(r["correct_accept"] for r in known)
    known_wa = sum(r["wrong_accept"] for r in known)
    known_confirm = sum(r["runtime_status"] == "CONFIRM" for r in known)
    known_reject = sum(r["runtime_status"] == "REJECT" for r in known)

    unk_accept = sum(r["runtime_status"] == "ACCEPT" for r in unknown)
    unk_confirm = sum(r["runtime_status"] == "CONFIRM" for r in unknown)
    unk_reject = sum(r["runtime_status"] == "REJECT" for r in unknown)

    disjoint_known = [
        r for r in known
        if r["query_source_disjoint_from_enrollment"]
    ]

    # Per-intent breakdown.
    per_intent = {}
    for intent_id in sorted(enrolled_by_intent):
        rr = [r for r in known if r["true_intent_id"] == intent_id]
        per_intent[intent_id] = {
            "count": len(rr),
            "correct_accept": safe_rate(
                sum(r["correct_accept"] for r in rr), len(rr)
            ),
            "wrong_accept": safe_rate(
                sum(r["wrong_accept"] for r in rr), len(rr)
            ),
            "confirm": safe_rate(
                sum(r["runtime_status"] == "CONFIRM" for r in rr), len(rr)
            ),
            "reject": safe_rate(
                sum(r["runtime_status"] == "REJECT" for r in rr), len(rr)
            ),
        }

    result = {
        "schema": "papr_ssl.demo_04a_wanghao_runtime.v1",
        "evaluation_scope": (
            "real_wanghao_external_domain_engineering_demo; "
            "source-disjoint where exported; not a pristine research test"
        ),
        "dataset_summary": ds_summary,
        "user_id": memory.user_id,
        "registered_intents": len(memory.intents),
        "shot": memory.shot_count(),
        "known_queries": len(known),
        "known_source_disjoint_queries": len(disjoint_known),
        "unknown_queries": len(unknown),
        "known_summary": {
            "correct_accept": safe_rate(known_ca, len(known)),
            "wrong_accept": safe_rate(known_wa, len(known)),
            "confirm": safe_rate(known_confirm, len(known)),
            "reject": safe_rate(known_reject, len(known)),
        },
        "unknown_summary": {
            "accept": safe_rate(unk_accept, len(unknown)),
            "confirm": safe_rate(unk_confirm, len(unknown)),
            "reject": safe_rate(unk_reject, len(unknown)),
        },
        "per_intent": per_intent,
        "scores_are_calibrated_probabilities": False,
        "speaker_authentication_performed": False,
        "generic_test": "sealed_not_accessed",
        "training_or_threshold_fitting_performed": False,
        "h6_policy_note": (
            "H6 decision head/policy was developed on a 20-registered-intent "
            "MDSC protocol. This 5-intent Wanghao run is an external-domain "
            "engineering validation; poor rejection/calibration may indicate "
            "policy distribution shift rather than representation failure."
        ),
        "elapsed_sec": time.perf_counter() - t0,
        "status": "COMPLETE",
    }

    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 112)
    print(
        "KNOWN   | "
        f"CA={result['known_summary']['correct_accept']:.4f} "
        f"WA={result['known_summary']['wrong_accept']:.4f} "
        f"CONFIRM={result['known_summary']['confirm']:.4f} "
        f"REJECT={result['known_summary']['reject']:.4f}"
    )
    print(
        "UNKNOWN | "
        f"ACCEPT={result['unknown_summary']['accept']:.4f} "
        f"CONFIRM={result['unknown_summary']['confirm']:.4f} "
        f"REJECT={result['unknown_summary']['reject']:.4f}"
    )
    print(
        f"known source-disjoint:    "
        f"{len(disjoint_known)}/{len(known)}"
    )
    print(f"memory:                   {memory_path}")
    print(f"predictions:              {args.output_dir/'predictions.jsonl'}")
    print(f"result:                   {args.output_dir/'result.json'}")
    print("scores calibrated:        NO")
    print("speaker authentication:   NO")
    print("generic_test accessed:    NO")
    print("training performed:       NO")
    print("DEMO-04A STATUS:          COMPLETE")


if __name__ == "__main__":
    main()
