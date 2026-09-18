#!/usr/bin/env python
"""Demo-04F — expand real-user unknown-set evaluation using all human-confirmed
unregistered Wanghao intents.

This is an engineering stress-test only:
- frozen Wanghao v2 UserMemory
- frozen H6 runtime / thresholds
- no enrollment changes
- no training / fitting
- no generic_test access

It uses all CONFIRMED toolkit intents that are NOT among the 5 registered intents.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from papr_ssl.inference.h6_personalized_runtime import PersonalizedRuntime, UserMemory
from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(n: int, d: int) -> float:
    return float(n / d) if d else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--toolkit-root",
        type=Path,
        required=True,
        help="Path to papr_audio_toolkit",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to corrected wanghao_demo_v2",
    )
    p.add_argument(
        "--memory",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime_v2/wanghao_user_memory.pt"),
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
        default=Path("artifacts/demo_04f_unknown_expansion"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    toolkit = args.toolkit_root.resolve()
    dataset_root = args.dataset_root.resolve()

    labels_path = toolkit / "artifacts/demo_03a_labeling/cluster_label_queue.csv"
    membership_path = toolkit / "artifacts/demo_02d_final_clusters/cluster_membership.csv"
    segments_manifest_path = toolkit / "data/manifests/wanghao_segments.jsonl"
    segments_dir = toolkit / "data/segments"
    demo_manifest_path = dataset_root / "manifest.jsonl"

    labels = read_csv(labels_path)
    membership = read_csv(membership_path)
    segments = read_jsonl(segments_manifest_path)
    demo_manifest = read_jsonl(demo_manifest_path)

    registered_intents = sorted({
        r["intent_id"]
        for r in demo_manifest
        if r.get("registered") is True and r.get("role") == "enrollment"
    })
    registered_set = set(registered_intents)

    seg_meta = {r["segment_id"]: r for r in segments}
    members_by_cluster = defaultdict(list)
    for r in membership:
        members_by_cluster[r["cluster_id"]].append(r)

    # Build all human-confirmed unregistered intent samples.
    unknown_samples = []
    seen_segment_ids = set()
    intent_canonical = {}

    for lab in labels:
        if lab.get("label_status") != "CONFIRMED":
            continue
        intent = lab.get("intent_id", "").strip()
        canonical = lab.get("canonical_text", "").strip()
        cluster_id = lab.get("cluster_id", "").strip()

        if not intent or not canonical or not cluster_id:
            continue
        if intent in registered_set:
            continue

        if intent in intent_canonical and intent_canonical[intent] != canonical:
            raise RuntimeError(
                f"Canonical mismatch for intent {intent}: "
                f"{intent_canonical[intent]!r} vs {canonical!r}"
            )
        intent_canonical[intent] = canonical

        for m in members_by_cluster.get(cluster_id, []):
            sid = m["segment_id"]
            if sid in seen_segment_ids:
                continue
            seen_segment_ids.add(sid)

            if sid not in seg_meta:
                raise KeyError(f"Missing segment metadata: {sid}")
            meta = seg_meta[sid]

            unknown_samples.append({
                "intent_id": intent,
                "canonical_text": canonical,
                "cluster_id": cluster_id,
                "segment_id": sid,
                "source_file": meta["source_file"],
                "segment_relpath": meta["segment_relpath"],
            })

    if not unknown_samples:
        raise RuntimeError("No confirmed unregistered samples found")

    memory = UserMemory.load(args.memory)
    if sorted(memory.intents) != registered_intents:
        raise RuntimeError(
            "Memory registered intents do not match corrected demo manifest:\n"
            f"memory={sorted(memory.intents)}\n"
            f"manifest={registered_intents}"
        )

    runtime = PersonalizedRuntime(
        head_json=args.head_json,
        policy_json=args.policy_json,
    )
    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(
        checkpoint=checkpoint,
        device=args.device,
    )

    print("=" * 116)
    print("DEMO-04F WANGHAO EXPANDED UNKNOWN STRESS TEST")
    print("=" * 116)
    print(f"registered intents:       {len(registered_intents)}")
    print(f"confirmed unknown intents:{len(intent_canonical)}")
    print(f"unknown samples:          {len(unknown_samples)}")
    print(f"device:                   {args.device}")
    print("-" * 116)

    rows = []

    for i, sample in enumerate(
        sorted(
            unknown_samples,
            key=lambda r: (r["intent_id"], r["source_file"], r["segment_id"])
        ),
        start=1,
    ):
        wav = segments_dir / sample["segment_relpath"]
        if not wav.exists():
            raise FileNotFoundError(wav)

        feat = adapter.extract_wav(wav)
        pred = runtime.predict_feature(
            memory=memory,
            query_global=feat["global_embedding"],
            query_temporal=feat["temporal_sequence"],
        )

        row = {
            **sample,
            "runtime_status": pred["status"],
            "predicted_intent_id": pred["intent_id"],
            "predicted_canonical_text": pred["canonical_text"],
            "decision_C": float(pred["decision_scores"]["C"]),
            "decision_W": float(pred["decision_scores"]["W"]),
            "decision_U": float(pred["decision_scores"]["U"]),
            "scores_are_calibrated_probabilities": False,
            "candidates": pred.get("candidates", []),
        }
        rows.append(row)

        print(
            f"[{i:02d}/{len(unknown_samples):02d}] "
            f"{sample['intent_id']:<28} "
            f"{sample['segment_id']:<32} -> "
            f"{pred['status']:<7} "
            f"{str(pred['intent_id'])}"
        )

    out_jsonl = args.output_dir / "predictions.jsonl"
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-intent summary.
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["intent_id"]].append(r)

    per_intent = {}
    summary_rows = []

    for intent in sorted(grouped):
        rr = grouped[intent]
        sources = sorted({r["source_file"] for r in rr})
        n_accept = sum(r["runtime_status"] == "ACCEPT" for r in rr)
        n_confirm = sum(r["runtime_status"] == "CONFIRM" for r in rr)
        n_reject = sum(r["runtime_status"] == "REJECT" for r in rr)

        accepted_as = Counter(
            r["predicted_intent_id"]
            for r in rr
            if r["runtime_status"] == "ACCEPT"
        )

        item = {
            "intent_id": intent,
            "canonical_text": intent_canonical[intent],
            "count": len(rr),
            "source_count": len(sources),
            "sources": sources,
            "accept": rate(n_accept, len(rr)),
            "confirm": rate(n_confirm, len(rr)),
            "reject": rate(n_reject, len(rr)),
            "accepted_as": dict(accepted_as),
        }
        per_intent[intent] = item
        summary_rows.append({
            "intent_id": intent,
            "canonical_text": intent_canonical[intent],
            "count": len(rr),
            "source_count": len(sources),
            "accept_count": n_accept,
            "confirm_count": n_confirm,
            "reject_count": n_reject,
            "accept_rate": item["accept"],
            "confirm_rate": item["confirm"],
            "reject_rate": item["reject"],
            "accepted_as": " | ".join(
                f"{k}:{v}" for k, v in sorted(accepted_as.items())
            ),
        })

    n = len(rows)
    n_accept = sum(r["runtime_status"] == "ACCEPT" for r in rows)
    n_confirm = sum(r["runtime_status"] == "CONFIRM" for r in rows)
    n_reject = sum(r["runtime_status"] == "REJECT" for r in rows)

    result = {
        "schema": "papr_ssl.demo_04f_unknown_expansion.v1",
        "scope": (
            "single-user engineering stress-test using all human-confirmed "
            "unregistered Wanghao intents; not an independent research benchmark"
        ),
        "registered_intents": registered_intents,
        "unknown_intent_count": len(per_intent),
        "unknown_sample_count": n,
        "overall": {
            "accept": rate(n_accept, n),
            "confirm": rate(n_confirm, n),
            "reject": rate(n_reject, n),
        },
        "per_intent": per_intent,
        "greet_focus": per_intent.get("greet"),
        "training_performed": False,
        "threshold_fitting_performed": False,
        "runtime_policy_changed": False,
        "generic_test": "sealed_not_accessed",
        "status": "PASS",
    }

    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = args.output_dir / "per_intent_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 116)
    print(
        "OVERALL UNKNOWN | "
        f"ACCEPT={result['overall']['accept']:.4f} "
        f"CONFIRM={result['overall']['confirm']:.4f} "
        f"REJECT={result['overall']['reject']:.4f}"
    )
    print("-" * 116)
    print("PER UNKNOWN INTENT")
    for r in summary_rows:
        print(
            f"{r['intent_id']:<28} "
            f"N={r['count']:>2d} "
            f"sources={r['source_count']:>2d} "
            f"A={r['accept_rate']:.3f} "
            f"C={r['confirm_rate']:.3f} "
            f"R={r['reject_rate']:.3f} "
            f"accepted_as={r['accepted_as']}"
        )

    if "greet" in per_intent:
        g = per_intent["greet"]
        print("-" * 116)
        print(
            f"GREET FOCUS | N={g['count']} "
            f"ACCEPT={g['accept']:.3f} "
            f"REJECT={g['reject']:.3f} "
            f"accepted_as={g['accepted_as']}"
        )

    print("-" * 116)
    print(f"predictions:              {out_jsonl}")
    print(f"per-intent summary:       {csv_path}")
    print(f"result:                   {args.output_dir/'result.json'}")
    print("training performed:       NO")
    print("threshold fitting:        NO")
    print("runtime policy changed:   NO")
    print("generic_test accessed:    NO")
    print("DEMO-04F STATUS:          PASS")


if __name__ == "__main__":
    main()
