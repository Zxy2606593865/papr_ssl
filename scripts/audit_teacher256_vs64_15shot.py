#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--result",type=Path,
                    default=Path("artifacts/p6_teacher_256_15shot/repeated_splits/p6_teacher_256_vs_64_15shot.json"))
    args=ap.parse_args()
    d=json.loads(args.result.read_text(encoding="utf-8"))
    if d["generic_test"]!="sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if d["project_2_canonical_64d"]!="preserved_not_modified":
        raise RuntimeError("64D preservation contract violated")
    print("="*108)
    print("P6 256D vs 64D — 15-SHOT COMPARISON AUDIT")
    print(f"repeated splits:           {d['num_splits']}")
    print(f"calibration feasible rate: {d['calibration_feasible_rate']:.3f}")
    print(f"absolute Gate pass rate:   {d['absolute_gate_pass_rate']:.3f}")
    print("-"*108)
    for k in ("macro_f1","correct_accept","wrong_intent","known_reject","unknown_reject"):
        ref=d["reference_64d_summary"][k]["mean"]
        cur=d["score_metric_summary"][k]["mean"]
        delta=d["delta_vs_64d_consistency_summary"][k]["mean"]
        print(f"{k:20s} 64D={ref:.6f} 256D={cur:.6f} delta={delta:+.6f}")
    print("-"*108)
    print("canonical 64D modified: NO")
    print("generic_test accessed:  NO")
    print("P6 256D vs 64D AUDIT: PASS")

if __name__=="__main__":
    main()
