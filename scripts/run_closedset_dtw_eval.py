#!/usr/bin/env python
"""Closed-set evaluation for current 256D Global vs Global+DTW system.

Protocol
--------
- Dataset: MDSC Core30 DEV only (442 known utterances)
- Enrollment: the same 15-shot enrollment used by the current 256D system
- No unknown data
- No rejector
- No threshold
- No ACCEPT/REJECT decision

20 repeated known-only splits:
- calibration half: choose lambda from {0.50, 0.75, 1.00}
- score half: report held-out closed-set metrics

This measures pure 30-way phrase recognition, not open-set performance.
generic_test remains sealed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

LAMBDA_GRID = (0.50, 0.75, 1.00)


def split_known(y, seed):
    rng = random.Random(seed)
    cal, score = [], []
    for c in range(30):
        idx = np.flatnonzero(y == c).tolist()
        rng.shuffle(idx)
        n = len(idx) // 2
        cal.extend(idx[:n])
        score.extend(idx[n:])
    return np.asarray(sorted(cal)), np.asarray(sorted(score))


def macro_f1(true, pred):
    vals = []
    for c in range(30):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        den = 2 * tp + fp + fn
        vals.append((2 * tp / den) if den else 0.0)
    return float(np.mean(vals))


def accuracy(true, pred):
    return float(np.mean(true == pred))


def global_pred(global_scores):
    return np.argmax(global_scores, axis=1).astype(np.int64)


def fused_pred(topk_class, topk_global, topk_dtw, lam):
    fused = lam * topk_global + (1.0 - lam) * topk_dtw
    best_local = np.argmax(fused, axis=1)
    row = np.arange(len(fused))
    return topk_class[row, best_local].astype(np.int64)


def choose_lambda(true_cal, topk_class, topk_global, topk_dtw):
    rows = []
    for lam in LAMBDA_GRID:
        pred = fused_pred(topk_class, topk_global, topk_dtw, lam)
        f1 = macro_f1(true_cal, pred)
        acc = accuracy(true_cal, pred)
        rows.append(
            {
                "lambda": float(lam),
                "macro_f1": f1,
                "accuracy": acc,
            }
        )

    # Primary: Macro-F1; secondary: accuracy; tie -> larger lambda
    # (prefer simpler / more global reliance).
    best = max(
        rows,
        key=lambda r: (r["macro_f1"], r["accuracy"], r["lambda"])
    )
    return best, rows


def summarize(vals):
    x = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "sample_std": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/"
            "teacher_256d_15shot_embeddings.npz"
        ),
    )
    p.add_argument(
        "--dtw-evidence",
        type=Path,
        default=Path(
            "artifacts/p6_temporal_dtw/precomputed/"
            "dtw_rerank_evidence.npz"
        ),
    )
    p.add_argument(
        "--split-policy",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_stability/repeated_splits/"
            "p6_15shot_stability.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_temporal_dtw/closedset_eval"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splitp = json.loads(args.split_policy.read_text(encoding="utf-8"))
    if splitp["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("split policy test seal violated")

    emb = np.load(args.embedding_npz, allow_pickle=False)
    true = emb["known_true"].astype(np.int64)
    if true.shape != (442,):
        raise RuntimeError(f"Expected 442 known DEV labels, got {true.shape}")

    e = np.load(args.dtw_evidence, allow_pickle=False)
    global_scores = e["known_global_scores"].astype(np.float64)
    topk_class = e["known_topk_class"].astype(np.int64)
    topk_global = e["known_topk_global"].astype(np.float64)
    topk_dtw = e["known_topk_dtw"].astype(np.float64)

    if global_scores.shape != (442, 30):
        raise RuntimeError(f"Unexpected global_scores shape {global_scores.shape}")
    if topk_class.shape != (442, 3):
        raise RuntimeError(f"Unexpected topk_class shape {topk_class.shape}")

    n = int(splitp["num_splits"])
    base = int(splitp["base_seed"])
    if n != 20:
        raise RuntimeError(f"Expected 20 splits, got {n}")

    runs = []

    for i in range(n):
        seed = base + i
        cal, score = split_known(true, seed)

        best, candidates = choose_lambda(
            true[cal],
            topk_class[cal],
            topk_global[cal],
            topk_dtw[cal],
        )
        lam = float(best["lambda"])

        pred_global = global_pred(global_scores[score])
        pred_dtw = fused_pred(
            topk_class[score],
            topk_global[score],
            topk_dtw[score],
            lam,
        )

        gm = {
            "macro_f1": macro_f1(true[score], pred_global),
            "accuracy": accuracy(true[score], pred_global),
        }
        dm = {
            "macro_f1": macro_f1(true[score], pred_dtw),
            "accuracy": accuracy(true[score], pred_dtw),
        }

        runs.append(
            {
                "split_index": i,
                "seed": seed,
                "selected_lambda": lam,
                "lambda_calibration_candidates": candidates,
                "global": gm,
                "dtw": dm,
                "delta": {
                    "macro_f1": dm["macro_f1"] - gm["macro_f1"],
                    "accuracy": dm["accuracy"] - gm["accuracy"],
                },
            }
        )

        print(
            f"[split {i+1:02d}/{n}] lambda={lam:.2f} | "
            f"GLOBAL F1={gm['macro_f1']:.4f} ACC={gm['accuracy']:.4f} | "
            f"DTW F1={dm['macro_f1']:.4f} ACC={dm['accuracy']:.4f} | "
            f"dF1={dm['macro_f1']-gm['macro_f1']:+.4f}"
        )

    global_summary = {
        "macro_f1": summarize([r["global"]["macro_f1"] for r in runs]),
        "accuracy": summarize([r["global"]["accuracy"] for r in runs]),
    }
    dtw_summary = {
        "macro_f1": summarize([r["dtw"]["macro_f1"] for r in runs]),
        "accuracy": summarize([r["dtw"]["accuracy"] for r in runs]),
    }
    delta_summary = {
        "macro_f1": summarize([r["delta"]["macro_f1"] for r in runs]),
        "accuracy": summarize([r["delta"]["accuracy"] for r in runs]),
    }

    lambda_hist = {
        str(lam): int(sum(r["selected_lambda"] == lam for r in runs))
        for lam in LAMBDA_GRID
    }

    # Descriptive full-DEV fixed-lambda results. These are NOT used for model
    # selection because they reuse the whole DEV set.
    full_dev = {
        "global": {
            "macro_f1": macro_f1(true, global_pred(global_scores)),
            "accuracy": accuracy(true, global_pred(global_scores)),
        }
    }
    for lam in LAMBDA_GRID:
        pdtw = fused_pred(topk_class, topk_global, topk_dtw, lam)
        full_dev[f"dtw_lambda_{lam:.2f}"] = {
            "macro_f1": macro_f1(true, pdtw),
            "accuracy": accuracy(true, pdtw),
        }

    out = {
        "schema": "papr_ssl.temporal_dtw_closedset_eval.v1",
        "task": "MDSC Core30 known-only closed-set DEV",
        "num_classes": 30,
        "known_dev_rows": 442,
        "shots_per_class": 15,
        "unknown_data_used": False,
        "rejector_used": False,
        "threshold_used": False,
        "num_splits": n,
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_selection": (
            "known calibration half only; primary Macro-F1, secondary accuracy, "
            "tie prefers larger lambda"
        ),
        "global_summary": global_summary,
        "dtw_summary": dtw_summary,
        "dtw_minus_global_summary": delta_summary,
        "lambda_histogram": lambda_hist,
        "full_dev_descriptive_only": full_dev,
        "canonical_64d_teacher": "preserved_not_modified",
        "project1_256d_teacher": "preserved_not_modified",
        "generic_test": "sealed_not_accessed",
        "runs": runs,
    }

    (args.output_dir / "closedset_dtw_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("CLOSED-SET GLOBAL vs DTW SUMMARY")
    print("=" * 108)
    for metric in ("macro_f1", "accuracy"):
        print(
            f"{metric:12s} "
            f"global={global_summary[metric]['mean']:.6f} "
            f"dtw={dtw_summary[metric]['mean']:.6f} "
            f"delta={delta_summary[metric]['mean']:+.6f}"
        )
    print(f"lambda histogram: {lambda_hist}")
    print("-" * 108)
    print("unknown data used:     NO")
    print("rejector used:         NO")
    print("threshold used:        NO")
    print("generic_test accessed: NO")
    print("CLOSED-SET EVAL STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
