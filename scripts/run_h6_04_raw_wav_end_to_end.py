#!/usr/bin/env python
"""H6-04 — Full raw-WAV enrollment + query end-to-end smoke test.

This is NOT a new benchmark.
It verifies that the promoted RawWavFeatureAdapter can replace cached features
for BOTH enrollment and query while preserving H6-02 runtime decisions.

Protocol
--------
- same frozen DEV speaker episode style as H6-02
- 20 registered intents
- 2-shot enrollment from raw WAV
- remaining known + 10 unregistered phrases queried from raw WAV
- compare every raw-WAV runtime decision with cached-feature runtime decision

PASS condition
--------------
- status match for every query
- intent_id match for every query
- no generic_test access
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from papr_ssl.inference.h6_personalized_runtime import (
    PersonalizedRuntime,
    UserMemory,
)
from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
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
        key=lambda ph: stable_hash("registered", repeat, speaker, ph),
    )
    return set(ranked[:REGISTERED_CLASSES])


def stable_support(indices, shot, repeat, speaker, label, utt_ids):
    return sorted(
        indices,
        key=lambda i: stable_hash("support", repeat, speaker, label, utt_ids[i]),
    )[:shot]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get(row, keys, default=""):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return default


def build_audio_map(manifest: Path, audio_root: Path):
    out = {}
    for row in read_jsonl(manifest):
        if str(get(row, ("split",))).lower() == "test":
            continue
        utt = str(get(row, ("utt_id", "id", "audio_id")))
        rel = get(
            row,
            (
                "audio_relpath",
                "audio_path",
                "wav_path",
                "source_path",
                "filepath",
                "file_path",
                "path",
                "file",
            ),
        )
        if not utt or not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = audio_root / p
        if p.exists():
            out[utt] = p.resolve()
    return out


def load_temporal_index(feature_dir: Path):
    out = {}
    p = feature_dir / "temporal" / "index.jsonl"
    for row in read_jsonl(p):
        out[row["utt_id"]] = feature_dir / "temporal" / row["cache_relpath"]
    return out


def load_seq(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["features"].to(torch.float32).numpy()


def summarize_known(rows):
    n = len(rows)
    if n == 0:
        return {}
    ca = sum(
        r["raw"]["status"] == "ACCEPT"
        and r["raw"]["intent_id"] == r["true_label"]
        for r in rows
    )
    wa = sum(
        r["raw"]["status"] == "ACCEPT"
        and r["raw"]["intent_id"] != r["true_label"]
        for r in rows
    )
    cf = sum(r["raw"]["status"] == "CONFIRM" for r in rows)
    rj = sum(r["raw"]["status"] == "REJECT" for r in rows)
    return {
        "count": n,
        "correct_accept": ca / n,
        "wrong_accept": wa / n,
        "confirm": cf / n,
        "reject": rj / n,
    }


def summarize_unknown(rows):
    n = len(rows)
    if n == 0:
        return {}
    ac = sum(r["raw"]["status"] == "ACCEPT" for r in rows)
    cf = sum(r["raw"]["status"] == "CONFIRM" for r in rows)
    rj = sum(r["raw"]["status"] == "REJECT" for r in rows)
    return {
        "count": n,
        "accept": ac / n,
        "confirm": cf / n,
        "reject": rj / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--audio-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
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
        default=Path("artifacts/h6_04_raw_wav_end_to_end"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )
    utt_id = d["dev_utt_id"].astype(str)
    speaker_id = d["dev_speaker_id"].astype(str)
    label = d["dev_label"].astype(str)
    embedding = d["dev_embedding"].astype(np.float32)

    temporal_ref = load_temporal_index(args.feature_dir)
    audio_map = build_audio_map(args.manifest, args.audio_root)

    speakers = sorted(set(speaker_id))
    speaker = speakers[0]
    idx_sp = np.where(speaker_id == speaker)[0]
    labels = sorted(set(label[idx_sp]))
    registered = choose_registered(labels, REPEAT, speaker)

    groups = defaultdict(list)
    for i in idx_sp:
        groups[label[i]].append(int(i))

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(
        checkpoint=checkpoint,
        device=args.device,
    )
    runtime = PersonalizedRuntime(
        head_json=args.head_json,
        policy_json=args.policy_json,
    )

    raw_memory = UserMemory(user_id=f"raw_demo_{speaker}")
    cached_memory = UserMemory(user_id=f"cached_demo_{speaker}")

    support_ids = set()

    print("=" * 108)
    print("H6-04 RAW-WAV ENROLLMENT")
    print("=" * 108)

    for phrase in sorted(registered):
        idx = groups[phrase]
        if len(idx) < SHOT + 1:
            raise RuntimeError(
                f"Registered phrase lacks {SHOT}-shot+query: {phrase}"
            )
        sup = stable_support(
            idx,
            SHOT,
            REPEAT,
            speaker,
            phrase,
            utt_id,
        )
        support_ids.update(sup)

        for sidx in sup:
            uid = utt_id[sidx]
            if uid not in audio_map:
                raise FileNotFoundError(f"Missing WAV for support: {uid}")

            raw_feat = adapter.extract_wav(audio_map[uid])

            raw_memory.enroll(
                intent_id=phrase,
                canonical_text=phrase,
                global_embedding=raw_feat["global_embedding"],
                temporal_sequence=raw_feat["temporal_sequence"],
            )
            cached_memory.enroll(
                intent_id=phrase,
                canonical_text=phrase,
                global_embedding=embedding[sidx],
                temporal_sequence=load_seq(temporal_ref[uid]),
            )

    raw_memory_path = args.output_dir / "raw_user_memory.pt"
    raw_memory.save(raw_memory_path)
    raw_memory = UserMemory.load(raw_memory_path)

    print(f"user_id:            {raw_memory.user_id}")
    print(f"registered intents: {len(raw_memory.intents)}")
    print(f"shot:               {raw_memory.shot_count()}")

    rows = []
    status_match = 0
    intent_match = 0

    print("=" * 108)
    print("H6-04 RAW-WAV QUERY + CACHED-RUNTIME PARITY")
    print("=" * 108)

    for i in idx_sp:
        if i in support_ids:
            continue

        uid = utt_id[i]
        if uid not in audio_map:
            raise FileNotFoundError(f"Missing WAV for query: {uid}")

        raw_feat = adapter.extract_wav(audio_map[uid])
        raw_result = runtime.predict_feature(
            memory=raw_memory,
            query_global=raw_feat["global_embedding"],
            query_temporal=raw_feat["temporal_sequence"],
        )
        cached_result = runtime.predict_feature(
            memory=cached_memory,
            query_global=embedding[i],
            query_temporal=load_seq(temporal_ref[uid]),
        )

        sm = raw_result["status"] == cached_result["status"]
        im = raw_result["intent_id"] == cached_result["intent_id"]
        status_match += int(sm)
        intent_match += int(im)

        rows.append(
            {
                "utt_id": uid,
                "true_label": label[i],
                "is_registered": bool(label[i] in registered),
                "raw": raw_result,
                "cached": cached_result,
                "status_match": sm,
                "intent_match": im,
            }
        )

    known_rows = [r for r in rows if r["is_registered"]]
    unknown_rows = [r for r in rows if not r["is_registered"]]

    known_summary = summarize_known(known_rows)
    unknown_summary = summarize_unknown(unknown_rows)

    result = {
        "schema": "papr_ssl.h6_04_raw_wav_end_to_end.v1",
        "speaker_id": speaker,
        "raw_wav_enrollment": True,
        "raw_wav_query": True,
        "registered_intents": len(raw_memory.intents),
        "shot": raw_memory.shot_count(),
        "query_count": len(rows),
        "status_match_count": status_match,
        "intent_match_count": intent_match,
        "status_match_rate": status_match / len(rows),
        "intent_match_rate": intent_match / len(rows),
        "known_summary": known_summary,
        "unknown_summary": unknown_summary,
        "all_predictions_match_cached_runtime": (
            status_match == len(rows) and intent_match == len(rows)
        ),
        "generic_test": "sealed_not_accessed",
        "rows": rows,
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"prediction parity | status={status_match}/{len(rows)} "
        f"intent={intent_match}/{len(rows)}"
    )
    print(
        "KNOWN   | "
        f"CA={known_summary['correct_accept']:.6f} "
        f"WA={known_summary['wrong_accept']:.6f} "
        f"CONFIRM={known_summary['confirm']:.6f} "
        f"REJECT={known_summary['reject']:.6f}"
    )
    print(
        "UNKNOWN | "
        f"ACCEPT={unknown_summary['accept']:.6f} "
        f"CONFIRM={unknown_summary['confirm']:.6f} "
        f"REJECT={unknown_summary['reject']:.6f}"
    )
    print("-" * 108)
    print(
        "raw-vs-cached runtime: "
        + (
            "PASS"
            if result["all_predictions_match_cached_runtime"]
            else "FAIL"
        )
    )
    print("generic_test accessed: NO")
    print(
        "H6-04 STATUS: "
        + (
            "PASS"
            if result["all_predictions_match_cached_runtime"]
            else "FAIL"
        )
    )


if __name__ == "__main__":
    main()
