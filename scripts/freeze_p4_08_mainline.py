#!/usr/bin/env python
"""P4-08: freeze the single Teacher mainline for P5.

This phase does NOT re-rank models.
It consumes the already-selected P4-06 winner and the P4-07 resource audit,
then writes one immutable mainline lock.

Frozen P4 definition:
- best Backbone + best exact hidden_state_index becomes sole mainline
- all other P4 configurations remain ablation baselines
- generic_test remains sealed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from papr_ssl.training.teacher.p4_candidates import get_p4_backbone


EXPECTED_MAINLINE_BACKBONE = "wavlm_large"
EXPECTED_MAINLINE_LAYER = 15


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def write_json_once(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != rendered:
            raise FileExistsError(
                f"immutable P4-08 lock already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selection-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_06_selection/p4_06_selection.json"
        ),
    )
    p.add_argument(
        "--resource-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_07_resource_cost/p4_07_resource_cost.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p4_runs/p4_08_mainline"),
    )
    args = p.parse_args()

    selection = load_json(args.selection_json)
    resource = load_json(args.resource_json)

    if selection.get("phase") != "P4-06":
        raise RuntimeError("P4-06 selection artifact required")
    if selection.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("P4-06 generic_test seal violated")
    if selection.get("resource_cost_used_for_selection") is not False:
        raise RuntimeError("P4-06 selection was unexpectedly resource-based")

    if resource.get("phase") != "P4-07":
        raise RuntimeError("P4-07 resource artifact required")
    if resource.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("P4-07 generic_test seal violated")
    if resource.get("teacher_selection_changed_by_resource_cost") is not False:
        raise RuntimeError("P4-07 illegally changed Teacher selection")

    selected = selection["selected_mainline"]
    backbone = str(selected["backbone"])
    layer = int(selected["hidden_state_index"])

    # This assertion makes the lock fail loudly if old/stale artifacts are mixed.
    if backbone != EXPECTED_MAINLINE_BACKBONE or layer != EXPECTED_MAINLINE_LAYER:
        raise RuntimeError(
            "P4-06 winner does not match the completed P4 study: "
            f"got {backbone} hidden_states[{layer}], expected "
            f"{EXPECTED_MAINLINE_BACKBONE} hidden_states[{EXPECTED_MAINLINE_LAYER}]"
        )

    before_resource = resource["selected_mainline_before_resource_profiling"]
    if before_resource["backbone"] != backbone:
        raise RuntimeError("P4-07 Backbone does not match P4-06")
    if int(before_resource["hidden_state_index"]) != layer:
        raise RuntimeError("P4-07 hidden-state index does not match P4-06")

    spec = get_p4_backbone(backbone)

    ablations = []
    for row in selection["backbone_winner_ranking"]:
        if (
            row["backbone"] == backbone
            and int(row["hidden_state_index"]) == layer
        ):
            continue
        ablations.append(
            {
                "backbone": row["backbone"],
                "hidden_state_index": int(row["hidden_state_index"]),
                "generic_dev_score_mean": float(
                    row["generic_dev_score_mean"]
                ),
                "role": "ablation_baseline",
            }
        )

    lock = {
        "schema": "papr_ssl.p4_08_mainline_lock.v1",
        "phase": "P4-08",
        "status": "FROZEN",
        "selection_source": args.selection_json.as_posix(),
        "resource_source": args.resource_json.as_posix(),
        "selection_metric": "generic_dev_score",
        "selection_metric_definition": "prototype_macro_f1",
        "selection_aggregation": "3-seed mean over seeds 17/29/43",
        "mainline": {
            "role": "sole_teacher_mainline_for_p5",
            "backbone_key": backbone,
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "hidden_state_index": layer,
            "hidden_dim": int(spec.embedding_dim),
            "p4_mean_dr_embedding_dim": 64,
            "p4_pooling": "masked_mean",
            "p4_scaf": {
                "k": 3,
                "margin_rad": 0.2,
                "scale": 30.0,
            },
            "generic_dev_score_mean": float(
                selected["generic_dev_score_mean"]
            ),
            "generic_dev_score_sample_std": float(
                selected["generic_dev_score_sample_std"]
            ),
            "generic_dev_score_ci95_lower": float(
                selected["generic_dev_score_ci95_lower"]
            ),
            "generic_dev_score_ci95_upper": float(
                selected["generic_dev_score_ci95_upper"]
            ),
        },
        "ablation_backbone_winners": ablations,
        "p5_contract": {
            "backbone_frozen": True,
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "hidden_state_index": layer,
            "hidden_dim": int(spec.embedding_dim),
            "comparison": [
                "Mean DR",
                "Attention DR",
            ],
            "seeds": [17, 29, 43],
            "frame_level_cache_required_for_attention": True,
            "required_frame_feature_shape": "[T,D]",
            "masked_mean_cache_alone_is_insufficient_for_attention": True,
            "generic_test": "sealed_not_accessed",
        },
        "resource_cost_may_change_mainline": False,
        "generic_test": "sealed_not_accessed",
    }

    p5_contract = {
        "schema": "papr_ssl.p5_input_contract.v1",
        "source_phase": "P4-08",
        "backbone_key": backbone,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "hidden_state_index": layer,
        "hidden_dim": int(spec.embedding_dim),
        "teacher_backbone_frozen": True,
        "dr_comparison": ["mean", "attention"],
        "seeds": [17, 29, 43],
        "attention_requires_frame_level_features": True,
        "frame_feature_shape": "[T,D]",
        "generic_test": "sealed_not_accessed",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "p4_08_mainline_lock.json"
    contract_path = args.output_dir / "p5_input_contract.json"

    write_json_once(lock_path, lock)
    write_json_once(contract_path, p5_contract)

    print("=" * 108)
    print("PAPR-SSL P4-08 MAINLINE FREEZE")
    print("=" * 108)
    print(f"Backbone:           {spec.model_id}")
    print(f"revision:           {spec.model_revision}")
    print(f"hidden_state_index: {layer}")
    print(f"hidden_dim:         {spec.embedding_dim}")
    print(
        "P4 mean Macro-F1: "
        f"{selected['generic_dev_score_mean']:.6f} "
        f"± {selected['generic_dev_score_sample_std']:.6f}"
    )
    print(
        "P4 95% CI:        "
        f"[{selected['generic_dev_score_ci95_lower']:.6f}, "
        f"{selected['generic_dev_score_ci95_upper']:.6f}]"
    )
    print("-" * 108)
    print("sole P5 mainline:              YES")
    print("other Backbones kept as:       ABLATION BASELINES")
    print("resource cost changes winner:  NO")
    print("generic_test accessed:         NO")
    print("P5 attention frame cache:      REQUIRED")
    print(f"lock:                          {lock_path}")
    print(f"P5 contract:                   {contract_path}")
    print("-" * 108)
    print("P4-08 STATUS: FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
