#!/usr/bin/env python
"""H6-02 cached-feature runtime demo.

This does NOT use raw WAV yet.
It verifies the FINAL enrollment/prediction semantics on frozen DEV features:

    enroll(user_id, intent, feature)
    predict_feature(user_id, query_feature)
        -> ACCEPT / CONFIRM / REJECT
        -> intent_id / canonical_text / candidate list

The selected DEV speaker remains evaluation-only; nothing is trained here.
generic_test is not accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# `python scripts/xxx.py` puts scripts/ at sys.path[0].
# Force the repository root to the front so newly added local subpackages
# (papr_ssl/inference) are visible even if an older editable/install copy exists.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from papr_ssl.inference.h6_personalized_runtime import (
    PersonalizedRuntime,
    UserMemory,
)


REGISTERED_CLASSES = 20
SHOT = 2
REPEAT = 0


def stable_hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def choose_registered(labels, repeat, speaker):
    ranked = sorted(
        labels,
        key=lambda ph: stable_hash(
            "registered", repeat, speaker, ph
        ),
    )
    return set(ranked[:REGISTERED_CLASSES])


def stable_support(indices, shot, repeat, speaker, label, utt_ids):
    return sorted(
        indices,
        key=lambda i: stable_hash(
            "support", repeat, speaker, label, utt_ids[i]
        ),
    )[:shot]


def load_temporal_index(feature_dir):
    out = {}
    p = feature_dir / "temporal" / "index.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["utt_id"]] = feature_dir / "temporal" / row["cache_relpath"]
    return out


def load_seq(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["features"].to(torch.float32).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    ap.add_argument(
        "--head-json",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head/cwu_head.json"),
    )
    ap.add_argument(
        "--policy-json",
        type=Path,
        default=Path("artifacts/h6_01_three_state_policy/h6_01_eval.json"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h6_02_cached_demo"),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir.exists():
        if not args.overwrite and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing overwrite: {args.output_dir}; use --overwrite"
            )
        if args.overwrite:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )
    utt_id = d["dev_utt_id"].astype(str)
    speaker_id = d["dev_speaker_id"].astype(str)
    label = d["dev_label"].astype(str)
    embedding = d["dev_embedding"].astype(np.float32)

    temporal = load_temporal_index(args.feature_dir)

    speakers = sorted(set(speaker_id))
    speaker = speakers[0]
    idx_sp = np.where(speaker_id == speaker)[0]
    labels = sorted(set(label[idx_sp]))
    registered = choose_registered(labels, REPEAT, speaker)

    groups = defaultdict(list)
    for i in idx_sp:
        groups[label[i]].append(int(i))

    memory = UserMemory(user_id=f"demo_{speaker}")

    support_ids = set()
    for phrase in sorted(registered):
        idx = groups[phrase]
        if len(idx) < SHOT + 1:
            raise RuntimeError(
                f"Selected registered phrase lacks {SHOT}-shot+query: {phrase}"
            )
        sup = stable_support(
            idx, SHOT, REPEAT, speaker, phrase, utt_id
        )
        support_ids.update(sup)

        for sidx in sup:
            memory.enroll(
                intent_id=phrase,
                canonical_text=phrase,
                global_embedding=embedding[sidx],
                temporal_sequence=load_seq(temporal[utt_id[sidx]]),
            )

    memory_path = args.output_dir / "demo_user_memory.pt"
    memory.save(memory_path)

    # Reload once to verify persistence.
    memory = UserMemory.load(memory_path)

    runtime = PersonalizedRuntime(
        head_json=args.head_json,
        policy_json=args.policy_json,
    )

    known_results = []
    unknown_results = []

    for i in idx_sp:
        if i in support_ids:
            continue

        is_known = label[i] in registered
        result = runtime.predict_feature(
            memory=memory,
            query_global=embedding[i],
            query_temporal=load_seq(temporal[utt_id[i]]),
        )
        rec = {
            "utt_id": utt_id[i],
            "true_label": label[i],
            "is_registered": is_known,
            "prediction": result,
        }
        if is_known:
            known_results.append(rec)
        else:
            unknown_results.append(rec)

    def summarize_known(rows):
        n = len(rows)
        ca = sum(
            r["prediction"]["status"] == "ACCEPT"
            and r["prediction"]["intent_id"] == r["true_label"]
            for r in rows
        )
        wa = sum(
            r["prediction"]["status"] == "ACCEPT"
            and r["prediction"]["intent_id"] != r["true_label"]
            for r in rows
        )
        cf = sum(r["prediction"]["status"] == "CONFIRM" for r in rows)
        rj = sum(r["prediction"]["status"] == "REJECT" for r in rows)
        return {
            "count": n,
            "correct_accept": ca / n,
            "wrong_accept": wa / n,
            "confirm": cf / n,
            "reject": rj / n,
        }

    def summarize_unknown(rows):
        n = len(rows)
        ac = sum(r["prediction"]["status"] == "ACCEPT" for r in rows)
        cf = sum(r["prediction"]["status"] == "CONFIRM" for r in rows)
        rj = sum(r["prediction"]["status"] == "REJECT" for r in rows)
        return {
            "count": n,
            "accept": ac / n,
            "confirm": cf / n,
            "reject": rj / n,
        }

    known_summary = summarize_known(known_results)
    unknown_summary = summarize_unknown(unknown_results)

    # Prefer showing one ACCEPT/CONFIRM/REJECT example if available.
    all_rows = known_results + unknown_results
    examples = {}
    for desired in ("ACCEPT", "CONFIRM", "REJECT"):
        hit = next(
            (r for r in all_rows if r["prediction"]["status"] == desired),
            None,
        )
        if hit is not None:
            examples[desired] = hit

    out = {
        "schema": "papr_ssl.h6_02_cached_demo.v1",
        "feature_source": "frozen_cached_dev_features",
        "raw_wav_used": False,
        "speaker_id": speaker,
        "user_id": memory.user_id,
        "shot": SHOT,
        "registered_intents": len(memory.intents),
        "unregistered_intents": len(labels) - len(memory.intents),
        "known_summary": known_summary,
        "unknown_summary": unknown_summary,
        "examples": examples,
        "scores_are_calibrated_probabilities": False,
        "speaker_authentication_performed": False,
        "generic_test": "sealed_not_accessed",
    }

    (args.output_dir / "result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H6-02 PERSONALIZED RUNTIME DEMO — CACHED FEATURES")
    print("=" * 108)
    print(f"user_id:               {memory.user_id}")
    print(f"registered intents:    {len(memory.intents)}")
    print(f"shot:                  {memory.shot_count()}")
    print("-" * 108)
    print(
        "KNOWN  | "
        f"CA={known_summary['correct_accept']:.6f} "
        f"WA={known_summary['wrong_accept']:.6f} "
        f"CONFIRM={known_summary['confirm']:.6f} "
        f"REJECT={known_summary['reject']:.6f}"
    )
    print(
        "UNKNOWN| "
        f"ACCEPT={unknown_summary['accept']:.6f} "
        f"CONFIRM={unknown_summary['confirm']:.6f} "
        f"REJECT={unknown_summary['reject']:.6f}"
    )
    print("-" * 108)
    for status in ("ACCEPT", "CONFIRM", "REJECT"):
        if status not in examples:
            continue
        r = examples[status]
        p = r["prediction"]
        compact = {
            "status": p["status"],
            "intent_id": p["intent_id"],
            "canonical_text": p["canonical_text"],
            "decision_scores": p["decision_scores"],
            "true_label": r["true_label"],
            "is_registered": r["is_registered"],
        }
        print(f"{status} EXAMPLE")
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    print("-" * 108)
    print("raw WAV used:           NO")
    print("speaker authentication: NO")
    print("generic_test accessed:  NO")
    print("H6-02 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
