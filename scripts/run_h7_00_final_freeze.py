#!/usr/bin/env python
"""H7-00 — Freeze the final research/runtime configuration before opening generic_test.

This script does NOT access generic_test audio/features/manifests.
It only snapshots the already-selected model/runtime artifacts and hashes them.

Purpose:
1. make the final test configuration immutable and auditable;
2. prevent accidental tuning after test exposure;
3. record exactly what will be evaluated.

No training. No threshold search. No DEV re-selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


REQUIRED_FILES = {
    "teacher_checkpoint": Path(
        "artifacts/p6_teacher_dim_sweep/dim_256/seed_43/best.pt"
    ),
    "cwu_head": Path(
        "artifacts/h5_01_cwu_decision_head/cwu_head.json"
    ),
    "three_state_policy": Path(
        "artifacts/h6_01_three_state_policy/h6_01_eval.json"
    ),
    "personalized_runtime": Path(
        "src/papr_ssl/inference/h6_personalized_runtime.py"
    ),
    "raw_wav_adapter": Path(
        "src/papr_ssl/inference/h6_raw_wav_adapter.py"
    ),
    "raw_wav_parity_summary": Path(
        "artifacts/h6_03_raw_wav_parity/summary.json"
    ),
    "raw_wav_e2e_summary": Path(
        "artifacts/h6_04_raw_wav_end_to_end/summary.json"
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prerequisites():
    missing = [str(p) for p in REQUIRED_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required final-freeze artifacts:\n  - "
            + "\n  - ".join(missing)
        )

    h603 = load_json(REQUIRED_FILES["raw_wav_parity_summary"])
    if h603.get("status") != "PASS":
        raise RuntimeError("H6-03 parity is not PASS; final freeze blocked.")
    if h603.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("H6-03 generic_test seal is not intact.")

    h604 = load_json(REQUIRED_FILES["raw_wav_e2e_summary"])
    if not h604.get("all_predictions_match_cached_runtime", False):
        raise RuntimeError("H6-04 raw-vs-cached runtime parity is not PASS.")
    if h604.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("H6-04 generic_test seal is not intact.")

    return h603, h604


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h7_00_final_freeze"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h603, h604 = validate_prerequisites()

    hashes = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in REQUIRED_FILES.items()
    }

    frozen_config = {
        "schema": "papr_ssl.h7_00_final_freeze.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_state": "FROZEN_BEFORE_GENERIC_TEST",
        "generic_test": "SEALED_NOT_ACCESSED",
        "backbone": {
            "model_id": "microsoft/wavlm-large",
            "revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
            "hidden_state_index": 15,
            "input_mode": "raw_float32_no_hf_normalization",
            "trainable": False,
        },
        "representation": {
            "global_embedding_dim": 256,
            "teacher_checkpoint": str(REQUIRED_FILES["teacher_checkpoint"]),
            "attention_dr": True,
            "l2_normalization": True,
        },
        "temporal": {
            "projection": "same frozen 1024->256 projection as teacher DR",
            "downsample_factor": 3,
            "dtw_local_cost": "1-cosine",
            "dtw_path_normalization": "path_length",
            "dtw_band_ratio": 0.25,
            "global_top_k_for_dtw": 3,
            "templates_per_class": 2,
            "fixed_fusion_lambda": 0.50,
        },
        "ranking": {
            "mainline": "fixed_global_plus_dtw",
            "h4_learned_shared_ranker_promoted": False,
        },
        "decision": {
            "head": "shared_linear_3class_C_W_U",
            "head_artifact": str(REQUIRED_FILES["cwu_head"]),
            "policy_artifact": str(REQUIRED_FILES["three_state_policy"]),
            "states": ["ACCEPT", "CONFIRM", "REJECT"],
            "scores_are_calibrated_probabilities": False,
        },
        "enrollment": {
            "supported_shots_for_frozen_policy": [1, 2],
            "loo_shrinkage_mainline": False,
            "high_shot_loo_status": "deferred_until_real_child_or_high_shot_data",
            "temporal_medoid_mainline": False,
        },
        "runtime": {
            "raw_wav_adapter_promoted": True,
            "raw_wav_parity_global_min_cosine": h603["global"]["min_cosine"],
            "raw_wav_parity_temporal_min_frame_cosine":
                h603["temporal"]["min_frame_cosine"],
            "raw_vs_cached_status_match_rate": h604["status_match_rate"],
            "raw_vs_cached_intent_match_rate": h604["intent_match_rate"],
        },
        "development_evidence": {
            "h6_01_note": (
                "Use H6-01 multi-speaker DEV results as development operating-point "
                "evidence; do not treat H6-04 single-speaker 100% as benchmark."
            ),
            "dev_is_not_pristine_final_validation": True,
        },
        "final_test_policy": {
            "no_parameter_update_after_open": True,
            "no_threshold_search_after_open": True,
            "no_architecture_change_after_open": True,
            "no_backbone_or_embedding_reselection_after_open": True,
            "report_failures_as_results_not_as_tuning_targets": True,
        },
        "artifact_hashes": hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    config_path = args.output_dir / "frozen_config.json"
    config_path.write_text(
        json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Independent seal digest over the exact serialized frozen config.
    seal_sha256 = sha256_file(config_path)
    seal = {
        "schema": "papr_ssl.h7_00_freeze_seal.v1",
        "frozen_config": str(config_path),
        "frozen_config_sha256": seal_sha256,
        "generic_test": "SEALED_NOT_ACCESSED",
    }
    (args.output_dir / "freeze_seal.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H7-00 FINAL CONFIGURATION FREEZE")
    print("=" * 108)
    print("research state:            FROZEN_BEFORE_GENERIC_TEST")
    print("generic_test accessed:     NO")
    print("backbone:                  microsoft/wavlm-large")
    print("revision:                  c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c")
    print("hidden_states index:       15")
    print("embedding:                 Attention DR / 256D")
    print("ranking:                   Global + fixed DTW lambda=0.50")
    print("decision:                  shared C/W/U -> ACCEPT/CONFIRM/REJECT")
    print("supported enrollment:      1-shot / 2-shot")
    print("-" * 108)
    print(f"frozen config:             {config_path}")
    print(f"freeze seal SHA256:        {seal_sha256}")
    print("H7-00 STATUS:              PASS")


if __name__ == "__main__":
    main()
