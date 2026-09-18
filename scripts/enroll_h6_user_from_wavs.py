#!/usr/bin/env python
"""Create a real user enrollment memory from raw WAV files.

Config format:
{
  "user_id": "student_001",
  "intents": [
    {
      "intent_id": "drink_water",
      "canonical_text": "我要喝水",
      "wavs": ["data/a.wav", "data/b.wav"]
    }
  ]
}

Current frozen H6 policy supports equal 1-shot or 2-shot enrollment per intent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from papr_ssl.inference.h6_personalized_runtime import UserMemory
from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path("artifacts/p6_teacher_256_15shot/embeddings/manifest.json"),
    )
    args = p.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    user_id = str(cfg["user_id"])
    intents = cfg["intents"]

    if not intents:
        raise ValueError("No intents in enrollment config")

    shot_counts = {len(item["wavs"]) for item in intents}
    if len(shot_counts) != 1:
        raise ValueError(
            f"All intents must currently use equal shot count; got {sorted(shot_counts)}"
        )
    shot = next(iter(shot_counts))
    if shot not in (1, 2):
        raise ValueError(
            f"Frozen H6 policy currently supports 1-shot or 2-shot; got {shot}"
        )

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(checkpoint=checkpoint)
    memory = UserMemory(user_id=user_id)

    for item in intents:
        intent_id = str(item["intent_id"])
        canonical_text = str(item["canonical_text"])
        for wav in item["wavs"]:
            feat = adapter.extract_wav(Path(wav))
            memory.enroll(
                intent_id=intent_id,
                canonical_text=canonical_text,
                global_embedding=feat["global_embedding"],
                temporal_sequence=feat["temporal_sequence"],
            )

    memory.save(args.output)

    print("=" * 100)
    print("H6 RAW-WAV USER ENROLLMENT")
    print("=" * 100)
    print("user_id:", memory.user_id)
    print("intents:", len(memory.intents))
    print("shot:", memory.shot_count())
    print("memory:", args.output)
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
