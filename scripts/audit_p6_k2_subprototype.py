#!/usr/bin/env python
import json
from pathlib import Path
p=Path("artifacts/p6_dev_k2/qualification/p6_k2_dev_gate.json")
d=json.loads(p.read_text(encoding="utf-8"))
m=d["results"]["attention"]["generic_dev_score_metrics"]
print("="*88)
print("P6 K2 SUBPROTOTYPE AUDIT")
print("="*88)
print(f"Macro F1:       {m['macro_f1']:.6f} >= 0.80")
print(f"Correct Accept: {m['correct_accept']:.6f} >= 0.80")
print(f"Wrong Intent:   {m['wrong_intent']:.6f} <= 0.10")
print(f"Known Reject:   {m['known_reject']:.6f} <= 0.20")
print(f"Unknown Reject: {m['unknown_reject']:.6f} >= 0.85")
print("eligible_for_P6_06:",d["eligible_for_p6_06_freeze"])
print("generic_test accessed: NO")
