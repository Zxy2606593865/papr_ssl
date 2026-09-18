#!/usr/bin/env python
"""P6-06: freeze Teacher, threshold policy and code state after dev Gate PASS.

This script refuses to freeze unless:
- P6-04 absolute Gate passes on generic_dev_score
- P6-05 E0b drop Gate passes
- generic_test remains sealed
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


CODE_PATHS = (
    Path("src/papr_ssl/training/teacher/p5_dr.py"),
    Path("scripts/prepare_p6_protocol.py"),
    Path("scripts/run_p6_dev_qualification.py"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol-json",
        type=Path,
        default=Path("artifacts/p6_dev/protocol/p6_protocol.json"),
    )
    p.add_argument(
        "--gate-json",
        type=Path,
        default=Path(
            "artifacts/p6_dev/qualification/p6_dev_gate.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_teacher"),
    )
    args = p.parse_args()

    protocol = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_json.read_text(encoding="utf-8"))

    if protocol["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("protocol test seal violated")
    if gate["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("gate test seal violated")
    if gate["p6_04_mainline_absolute_gate_pass"] is not True:
        raise RuntimeError("cannot freeze: P6-04 absolute Gate failed")
    if gate["p6_05_e0b"]["pass"] is not True:
        raise RuntimeError("cannot freeze: P6-05 E0b Gate failed")
    if gate["eligible_for_p6_06_freeze"] is not True:
        raise RuntimeError("cannot freeze: not eligible")

    mainline = gate["results"]["attention"]
    checkpoint = Path(mainline["checkpoint"]["checkpoint"])
    if sha256(checkpoint) != mainline["checkpoint"]["checkpoint_sha256"]:
        raise RuntimeError("Teacher checkpoint hash mismatch")

    code_hashes = {}
    for path in CODE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(
                f"expected frozen code path not found: {path}"
            )
        code_hashes[path.as_posix()] = sha256(path)

    lock = {
        "schema": "papr_ssl.p6_teacher_lock.v1",
        "phase": "P6-06",
        "status": "FROZEN_BEFORE_GENERIC_TEST",
        "teacher": {
            "model_id": "microsoft/wavlm-large",
            "model_revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
            "hidden_state_index": 15,
            "head": "AttentionDR",
            "embedding_dim": 64,
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": mainline["checkpoint"][
                "checkpoint_sha256"
            ],
            "representative_seed": mainline["checkpoint"]["seed"],
            "selected_epoch": mainline["checkpoint"]["selected_epoch"],
        },
        "prototype_policy": protocol["p6_01"],
        "threshold_policy": {
            "accept_rule": (
                "top1_cosine_score >= score_threshold AND "
                "top1_minus_top2_margin >= margin_threshold"
            ),
            "score_threshold": mainline["calibration"][
                "score_threshold"
            ],
            "margin_threshold": mainline["calibration"][
                "margin_threshold"
            ],
            "calibration_split": "generic_dev_cal",
            "calibration_feasible": mainline["calibration"][
                "calibration_feasible"
            ],
        },
        "dev_gate_metrics": mainline["generic_dev_score_metrics"],
        "absolute_gates": gate["absolute_gates"],
        "e0b": gate["p6_05_e0b"],
        "code_sha256": code_hashes,
        "git": git_state(),
        "generic_test": "sealed_not_accessed",
        "test_may_tune_model_or_thresholds": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "teacher_lock.json"
    rendered = json.dumps(lock, ensure_ascii=False, indent=2) + "\n"
    if out.exists():
        if out.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                "Teacher lock already exists with different content"
            )
    else:
        out.write_text(rendered, encoding="utf-8")

    print("=" * 108)
    print("P6-06 TEACHER FREEZE")
    print("=" * 108)
    print("Backbone:            microsoft/wavlm-large")
    print("hidden_state_index:  15")
    print("Head:                AttentionDR -> 64D")
    print(
        "checkpoint seed:     "
        f"{lock['teacher']['representative_seed']}"
    )
    print(
        "score threshold:     "
        f"{lock['threshold_policy']['score_threshold']:.6f}"
    )
    print(
        "margin threshold:    "
        f"{lock['threshold_policy']['margin_threshold']:.6f}"
    )
    print("generic_test accessed: NO")
    print("test may tune policy:  NO")
    print("-" * 108)
    print("P6-06 STATUS: FROZEN / READY FOR P6-07")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
