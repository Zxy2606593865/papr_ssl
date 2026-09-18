#!/usr/bin/env python
"""Audit P4-07 resource-cost artifact.

The audit verifies that resource profiling did not change the P4-06 winner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_BACKBONES = {
    "wav2vec2_base",
    "wavlm_large",
    "w2v_bert2",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--resource-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_07_resource_cost/"
            "p4_07_resource_cost.json"
        ),
    )
    p.add_argument(
        "--selection-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_06_selection/"
            "p4_06_selection.json"
        ),
    )
    args = p.parse_args()

    resource = json.loads(
        args.resource_json.read_text(encoding="utf-8")
    )
    selection = json.loads(
        args.selection_json.read_text(encoding="utf-8")
    )

    if resource["schema"] != "papr_ssl.p4_07_resource_cost.v1":
        raise RuntimeError("P4-07 schema mismatch")
    if resource["phase"] != "P4-07":
        raise RuntimeError("P4-07 phase mismatch")
    if resource["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if resource["teacher_selection_changed_by_resource_cost"] is not False:
        raise RuntimeError("P4-07 illegally changed Teacher selection")
    if resource["ranking_by_resource_cost_performed"] is not False:
        raise RuntimeError("P4-07 illegally ranked by resource cost")

    before = resource["selected_mainline_before_resource_profiling"]
    expected = selection["selected_mainline"]
    if before["backbone"] != expected["backbone"]:
        raise RuntimeError("selected Backbone changed")
    if int(before["hidden_state_index"]) != int(
        expected["hidden_state_index"]
    ):
        raise RuntimeError("selected layer changed")

    records = resource["records"]
    if len(records) != 3:
        raise RuntimeError("expected exactly 3 Backbone resource records")
    if {r["backbone"] for r in records} != EXPECTED_BACKBONES:
        raise RuntimeError("resource Backbone coverage mismatch")

    for r in records:
        if int(r["parameter_count_total"]) <= 0:
            raise RuntimeError(f"{r['backbone']}: invalid parameter count")
        if int(r["parameter_storage_bytes_float32"]) <= 0:
            raise RuntimeError(f"{r['backbone']}: invalid parameter bytes")
        if r["standard_hf_full_forward"] is not True:
            raise RuntimeError(f"{r['backbone']}: not standard full forward")
        if r["early_exit_or_encoder_truncation_used"] is not False:
            raise RuntimeError(f"{r['backbone']}: early exit unexpectedly used")

        for section in (
            "model_forward_latency_ms",
            "waveform_to_hidden_latency_ms",
        ):
            stats = r["benchmark"][section]
            for key in ("median", "mean", "p95", "min", "max"):
                if not math.isfinite(float(stats[key])):
                    raise RuntimeError(
                        f"{r['backbone']}: nonfinite {section}.{key}"
                    )
                if float(stats[key]) <= 0:
                    raise RuntimeError(
                        f"{r['backbone']}: nonpositive {section}.{key}"
                    )

    print("=" * 108)
    print("P4-07 RESOURCE COST AUDIT")
    print("=" * 108)
    for r in records:
        print(
            f"{r['backbone']:18s} "
            f"layer={r['selected_hidden_state_index']:2d}  "
            f"params={r['parameter_count_total']:,}  "
            f"fwd_med={r['benchmark']['model_forward_latency_ms']['median']:.3f} ms  "
            f"e2e_med={r['benchmark']['waveform_to_hidden_latency_ms']['median']:.3f} ms"
        )
    print("-" * 108)
    print(
        "Teacher remains: "
        f"{expected['backbone']} "
        f"hidden_states[{expected['hidden_state_index']}]"
    )
    print("teacher selection changed:        NO")
    print("resource-cost ranking performed:  NO")
    print("generic_test accessed:            NO")
    print("-" * 108)
    print("P4-07 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
