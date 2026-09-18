#!/usr/bin/env python
"""Audit the immutable P4-08 Teacher mainline lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MODEL_ID = "microsoft/wavlm-large"
EXPECTED_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
EXPECTED_LAYER = 15
EXPECTED_DIM = 1024


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lock-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_08_mainline/p4_08_mainline_lock.json"
        ),
    )
    p.add_argument(
        "--p5-contract-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_08_mainline/p5_input_contract.json"
        ),
    )
    args = p.parse_args()

    lock = json.loads(args.lock_json.read_text(encoding="utf-8"))
    p5 = json.loads(args.p5_contract_json.read_text(encoding="utf-8"))

    if lock["schema"] != "papr_ssl.p4_08_mainline_lock.v1":
        raise RuntimeError("P4-08 schema mismatch")
    if lock["phase"] != "P4-08" or lock["status"] != "FROZEN":
        raise RuntimeError("P4-08 not frozen")
    if lock["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if lock["resource_cost_may_change_mainline"] is not False:
        raise RuntimeError("resource cost must not alter mainline")

    mainline = lock["mainline"]
    if mainline["role"] != "sole_teacher_mainline_for_p5":
        raise RuntimeError("mainline role mismatch")
    if mainline["model_id"] != EXPECTED_MODEL_ID:
        raise RuntimeError("unexpected model_id")
    if mainline["model_revision"] != EXPECTED_REVISION:
        raise RuntimeError("unexpected model revision")
    if int(mainline["hidden_state_index"]) != EXPECTED_LAYER:
        raise RuntimeError("unexpected hidden_state_index")
    if int(mainline["hidden_dim"]) != EXPECTED_DIM:
        raise RuntimeError("unexpected hidden_dim")

    if len(lock["ablation_backbone_winners"]) != 2:
        raise RuntimeError("expected two alternative Backbone winners")
    if any(
        row["role"] != "ablation_baseline"
        for row in lock["ablation_backbone_winners"]
    ):
        raise RuntimeError("ablation role mismatch")

    if p5["schema"] != "papr_ssl.p5_input_contract.v1":
        raise RuntimeError("P5 contract schema mismatch")
    if p5["model_id"] != EXPECTED_MODEL_ID:
        raise RuntimeError("P5 model_id mismatch")
    if p5["model_revision"] != EXPECTED_REVISION:
        raise RuntimeError("P5 revision mismatch")
    if int(p5["hidden_state_index"]) != EXPECTED_LAYER:
        raise RuntimeError("P5 layer mismatch")
    if p5["teacher_backbone_frozen"] is not True:
        raise RuntimeError("P5 Backbone must remain frozen")
    if p5["dr_comparison"] != ["mean", "attention"]:
        raise RuntimeError("P5 comparison contract changed")
    if p5["seeds"] != [17, 29, 43]:
        raise RuntimeError("P5 seed contract changed")
    if p5["attention_requires_frame_level_features"] is not True:
        raise RuntimeError("P5 attention must require frame-level features")
    if p5["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("P5 generic_test seal violated")

    print("=" * 108)
    print("P4-08 MAINLINE FREEZE AUDIT")
    print("=" * 108)
    print(f"model_id:            {mainline['model_id']}")
    print(f"revision:            {mainline['model_revision']}")
    print(f"hidden_state_index:  {mainline['hidden_state_index']}")
    print(f"hidden_dim:          {mainline['hidden_dim']}")
    print(f"generic_dev mean:    {mainline['generic_dev_score_mean']:.6f}")
    print(f"generic_dev std:     {mainline['generic_dev_score_sample_std']:.6f}")
    print(
        "95% CI:              "
        f"[{mainline['generic_dev_score_ci95_lower']:.6f}, "
        f"{mainline['generic_dev_score_ci95_upper']:.6f}]"
    )
    print("-" * 108)
    print("sole P5 mainline:             YES")
    print("alternative Backbone winners: ABLATION ONLY")
    print("P5 Mean vs Attention:         FROZEN")
    print("P5 frame-level cache:         REQUIRED")
    print("generic_test accessed:        NO")
    print("-" * 108)
    print("P4-08 STATUS: PASS / P4 CLOSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
