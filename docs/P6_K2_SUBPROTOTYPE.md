# P6 K=2 Subprototype fallback

K=1 calibration was infeasible. This activates the already-defined P6-01 fallback.

Important distinction:

- `K=2 enrollment average` = two samples averaged into one prototype.
- `K=2 subprototype` = keep two independent prototypes per class.

This bundle implements the second definition:

```text
class_score(c) = max(cos(z,p_c1), cos(z,p_c2))
```

Top1/top2 margin is computed over the 30 class scores.

Run:

```powershell
python scripts/prepare_p6_k2_subprototype.py
python scripts/run_p6_k2_subprototype.py --device cuda
python scripts/audit_p6_k2_subprototype.py
```

K=1 artifacts are preserved under `artifacts/p6_dev`; K=2 writes to
`artifacts/p6_dev_k2`. `generic_test` remains sealed.
