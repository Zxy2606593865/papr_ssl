#!/usr/bin/env python
"""Minimal H6 raw-WAV inference helper.

Run this only AFTER H6-03 parity is PASS.

It loads:
- existing H6-02 user memory
- frozen RawWavFeatureAdapter
- existing H6-02 PersonalizedRuntime

and predicts one WAV.

This script does NOT enroll new WAVs yet; it is a query-side smoke test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from papr_ssl.inference.h6_personalized_runtime import (
    PersonalizedRuntime,
    UserMemory,
)
from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=Path)
    p.add_argument(
        "--memory",
        type=Path,
        default=Path("artifacts/h6_02_cached_demo/demo_user_memory.pt"),
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
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/manifest.json"
        ),
    )
    args = p.parse_args()

    parity = json.loads(
        Path("artifacts/h6_03_raw_wav_parity/summary.json").read_text(
            encoding="utf-8"
        )
    )
    if parity.get("status") != "PASS":
        raise RuntimeError("H6-03 parity is not PASS. Raw WAV runtime is blocked.")

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    adapter = RawWavFeatureAdapter(checkpoint=checkpoint)
    memory = UserMemory.load(args.memory)
    runtime = PersonalizedRuntime(
        head_json=args.head_json,
        policy_json=args.policy_json,
    )

    feat = adapter.extract_wav(args.wav)
    result = runtime.predict_feature(
        memory=memory,
        query_global=feat["global_embedding"],
        query_temporal=feat["temporal_sequence"],
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
