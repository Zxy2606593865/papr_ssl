#!/usr/bin/env python
"""P6 enrollment-shot ablation: 10-shot vs 15-shot mean prototype.

Goal
----
Estimate whether simply increasing REAL enrollment count can rescue the current
frozen Teacher before changing its training objective.

Frozen:
- WavLM-large, outputs.hidden_states[15]
- AttentionDR 64D representative checkpoint from P6 protocol
- same generic_dev_cal / generic_dev_score partitions
- same score+margin accept rule
- same absolute Gates
- exact empirical threshold search on generic_dev_cal only
- generic_test remains sealed

Only changed factor:
- enrollment shots per intent: 10 vs 15

Prototype policy:
- all N enrollment embeddings are averaged
- resulting centroid is L2-normalized
- one prototype per intent

Important:
- 10-shot enrollment is a strict subset of the 15-shot enrollment because both
  use the same deterministic train-side ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WavLMModel

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset, frame_collate
from papr_ssl.training.teacher.real_experiment import (
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)

MODEL_ID = "microsoft/wavlm-large"
MODEL_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
HIDDEN_STATE_INDEX = 15
SAMPLE_RATE = 16000
SHOTS = (10, 15)
SALT = "papr_ssl_p6_protocol_v1"

GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}


def stable_key(*parts: str) -> str:
    h = hashlib.sha256()
    h.update(SALT.encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_attention_dr(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if obj.get("method") != "attention":
        raise RuntimeError(
            f"expected AttentionDR checkpoint, got {obj.get('method')}"
        )
    dr = build_dr("attention")
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)
    for p in dr.parameters():
        p.requires_grad_(False)
    return dr


@torch.inference_mode()
def embed_dataset(
    dataset: FrameCacheDataset,
    dr: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )
    out = {}
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        z = dr(x, mask).cpu()
        for utt_id, emb in zip(batch["utt_ids"], z):
            out[utt_id] = emb.contiguous()
    return out


def load_wave(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"expected 16 kHz: {path}")
    x = torch.from_numpy(data).to(torch.float32).mean(dim=1)
    if x.numel() <= 0 or not torch.isfinite(x).all():
        raise RuntimeError(f"invalid waveform: {path}")
    return x.contiguous()


@torch.inference_mode()
def embed_unknown_dev(
    *,
    open_index: Path,
    mdsc_root: Path,
    wanted_utt_ids: set[str],
    backbone: WavLMModel,
    dr: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = load_real_task_rows(open_index, splits=("dev",))
    rows = [r for r in rows if r.utt_id in wanted_utt_ids]
    if len(rows) != len(wanted_utt_ids):
        raise RuntimeError(
            f"wanted {len(wanted_utt_ids)} unknown rows, got {len(rows)}"
        )

    legacy = [
        r for r in rows
        if r.audio_relpath is None and qualified_utt_relpath(r) is None
    ]
    catalog = build_mdsc_audio_catalog(mdsc_root) if legacy else None

    out = {}
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=catalog,
        )
        wav = load_wave(path).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(
            wav,
            dtype=torch.long,
            device=device,
        )
        model_out = backbone(
            input_values=wav,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = model_out.hidden_states[HIDDEN_STATE_INDEX].to(torch.float32)
        frame_mask = backbone._get_feature_vector_attention_mask(
            hs.shape[1],
            attention_mask,
        ).to(torch.bool)
        out[row.utt_id] = dr(
            hs,
            frame_mask,
        ).squeeze(0).cpu().contiguous()

        if i % 100 == 0 or i == len(rows):
            print(f"[unknown dev] {i}/{len(rows)}")
    return out


def train_groups(dataset: FrameCacheDataset) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for row in dataset.rows:
        groups.setdefault(row["label_text"], []).append(row["utt_id"])
    return groups


def build_enrollment_sets(
    dataset: FrameCacheDataset,
) -> dict[int, list[dict]]:
    groups = train_groups(dataset)
    out = {shot: [] for shot in SHOTS}

    for label in sorted(groups):
        ordered = sorted(
            groups[label],
            key=lambda utt: stable_key("enrollment", label, utt),
        )
        if len(ordered) < max(SHOTS):
            raise RuntimeError(
                f"class {label!r} has only {len(ordered)} train samples; "
                f"need at least {max(SHOTS)}"
            )
        label_index = dataset.class_to_index[label]
        for shot in SHOTS:
            for utt_id in ordered[:shot]:
                out[shot].append(
                    {
                        "utt_id": utt_id,
                        "label_text": label,
                        "label_index": label_index,
                    }
                )

    # Strict nesting check.
    ids10 = {r["utt_id"] for r in out[10]}
    ids15 = {r["utt_id"] for r in out[15]}
    if not ids10.issubset(ids15):
        raise RuntimeError("10-shot enrollment must be subset of 15-shot")

    return out


def build_mean_prototypes(
    enrollment_rows: list[dict],
    embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    by_class: dict[int, list[torch.Tensor]] = {}
    for row in enrollment_rows:
        by_class.setdefault(int(row["label_index"]), []).append(
            embeddings[row["utt_id"]]
        )
    if sorted(by_class) != list(range(30)):
        raise RuntimeError("expected exactly 30 intent classes")

    protos = []
    for c in range(30):
        stacked = torch.stack(by_class[c], dim=0)
        p = stacked.mean(dim=0)
        protos.append(F.normalize(p, p=2, dim=0))
    return torch.stack(protos, dim=0)


def score_rows(
    rows: list[dict],
    embeddings: dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    *,
    known: bool,
) -> dict[str, np.ndarray]:
    z = torch.stack(
        [embeddings[row["utt_id"]] for row in rows],
        dim=0,
    )
    z = F.normalize(z, p=2, dim=-1)
    sim = z @ prototypes.T
    top2 = torch.topk(sim, k=2, dim=1)

    out = {
        "score": top2.values[:, 0].numpy().astype(np.float64),
        "margin": (
            top2.values[:, 0] - top2.values[:, 1]
        ).numpy().astype(np.float64),
        "pred": top2.indices[:, 0].numpy().astype(np.int64),
    }
    if known:
        out["true"] = np.asarray(
            [int(row["label_index"]) for row in rows],
            dtype=np.int64,
        )
    return out


def empirical_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(values, dtype=np.float64))
    return np.concatenate(
        [
            np.asarray([np.nextafter(unique[0], -np.inf)]),
            unique,
            np.asarray([np.nextafter(unique[-1], np.inf)]),
        ]
    )


def macro_f1_vectorized(
    accepted: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    total = np.zeros(accepted.shape[0], dtype=np.float64)
    for c in range(30):
        true_c = true == c
        pred_c = pred == c
        tp = np.sum(
            accepted & true_c[None, :] & pred_c[None, :],
            axis=1,
        )
        fp = np.sum(
            accepted & (~true_c)[None, :] & pred_c[None, :],
            axis=1,
        )
        fn = int(np.sum(true_c)) - tp
        denom = 2.0 * tp + fp + fn
        total += np.divide(
            2.0 * tp,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
    return total / 30.0


def metric_vectors(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
    score_threshold: float,
    margin_candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    k_score = known["score"] >= score_threshold
    u_score = unknown["score"] >= score_threshold
    ka = (
        known["margin"][None, :] >= margin_candidates[:, None]
    ) & k_score[None, :]
    ua = (
        unknown["margin"][None, :] >= margin_candidates[:, None]
    ) & u_score[None, :]

    correct = known["pred"] == known["true"]
    wrong = ~correct

    return {
        "macro_f1": macro_f1_vectorized(
            ka,
            known["true"],
            known["pred"],
        ),
        "correct_accept": np.mean(
            ka & correct[None, :],
            axis=1,
        ),
        "wrong_intent": np.mean(
            ka & wrong[None, :],
            axis=1,
        ),
        "known_reject": 1.0 - np.mean(ka, axis=1),
        "unknown_reject": 1.0 - np.mean(ua, axis=1),
        "unknown_accept_far": np.mean(ua, axis=1),
    }


def feasible_mask(v: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (v["macro_f1"] >= GATES["macro_f1_min"])
        & (v["correct_accept"] >= GATES["correct_accept_min"])
        & (v["wrong_intent"] <= GATES["wrong_intent_max"])
        & (v["known_reject"] <= GATES["known_reject_max"])
        & (v["unknown_reject"] >= GATES["unknown_reject_min"])
    )


def violation_vector(v: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.maximum(
            0.0,
            GATES["macro_f1_min"] - v["macro_f1"],
        ) / GATES["macro_f1_min"]
        + np.maximum(
            0.0,
            GATES["correct_accept_min"] - v["correct_accept"],
        ) / GATES["correct_accept_min"]
        + np.maximum(
            0.0,
            v["wrong_intent"] - GATES["wrong_intent_max"],
        ) / GATES["wrong_intent_max"]
        + np.maximum(
            0.0,
            v["known_reject"] - GATES["known_reject_max"],
        ) / GATES["known_reject_max"]
        + np.maximum(
            0.0,
            GATES["unknown_reject_min"] - v["unknown_reject"],
        ) / GATES["unknown_reject_min"]
    )


def scalar_metrics(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
    ts: float,
    tm: float,
) -> dict:
    ka = (known["score"] >= ts) & (known["margin"] >= tm)
    ua = (unknown["score"] >= ts) & (unknown["margin"] >= tm)
    correct = ka & (known["pred"] == known["true"])
    wrong = ka & (known["pred"] != known["true"])

    f1s = []
    for c in range(30):
        pred_c = ka & (known["pred"] == c)
        true_c = known["true"] == c
        tp = int(np.sum(pred_c & true_c))
        fp = int(np.sum(pred_c & ~true_c))
        fn = int(np.sum(~pred_c & true_c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)

    return {
        "macro_f1": float(np.mean(f1s)),
        "correct_accept": float(np.mean(correct)),
        "wrong_intent": float(np.mean(wrong)),
        "known_reject": float(np.mean(~ka)),
        "unknown_reject": float(np.mean(~ua)),
        "unknown_accept_far": float(np.mean(ua)),
        "known_count": int(len(ka)),
        "unknown_count": int(len(ua)),
    }


def gate_pass(m: dict) -> bool:
    return (
        m["macro_f1"] >= GATES["macro_f1_min"]
        and m["correct_accept"] >= GATES["correct_accept_min"]
        and m["wrong_intent"] <= GATES["wrong_intent_max"]
        and m["known_reject"] <= GATES["known_reject_max"]
        and m["unknown_reject"] >= GATES["unknown_reject_min"]
    )


def exact_calibrate(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
) -> dict:
    score_candidates = empirical_candidates(
        np.concatenate([known["score"], unknown["score"]])
    )
    margin_candidates = empirical_candidates(
        np.concatenate([known["margin"], unknown["margin"]])
    )

    best_feasible = None
    best_feasible_key = None
    best_diag = None
    best_diag_key = None
    feasible_count = 0

    for i, ts in enumerate(score_candidates, start=1):
        v = metric_vectors(
            known,
            unknown,
            float(ts),
            margin_candidates,
        )
        ok = feasible_mask(v)
        idxs = np.flatnonzero(ok)
        feasible_count += int(idxs.size)

        for j in idxs:
            key = (
                float(v["macro_f1"][j]),
                float(v["correct_accept"][j]),
                float(v["unknown_reject"][j]),
                -float(v["wrong_intent"][j]),
                -float(v["known_reject"][j]),
                float(ts),
                float(margin_candidates[j]),
            )
            if best_feasible_key is None or key > best_feasible_key:
                best_feasible_key = key
                best_feasible = {
                    "score_threshold": float(ts),
                    "margin_threshold": float(margin_candidates[j]),
                    "metrics": {
                        name: float(arr[j])
                        for name, arr in v.items()
                    },
                }

        if best_feasible is None:
            vio = violation_vector(v)
            j = int(np.argmin(vio))
            key = (
                -float(vio[j]),
                float(v["macro_f1"][j]),
                float(v["correct_accept"][j]),
                float(v["unknown_reject"][j]),
                -float(v["known_reject"][j]),
                -float(v["wrong_intent"][j]),
            )
            if best_diag_key is None or key > best_diag_key:
                best_diag_key = key
                best_diag = {
                    "score_threshold": float(ts),
                    "margin_threshold": float(margin_candidates[j]),
                    "metrics": {
                        name: float(arr[j])
                        for name, arr in v.items()
                    },
                    "normalized_gate_violation": float(vio[j]),
                }

        if i % 100 == 0 or i == len(score_candidates):
            print(
                f"[exact] {i}/{len(score_candidates)} score thresholds; "
                f"feasible_pairs={feasible_count}"
            )

    selected = best_feasible if best_feasible is not None else best_diag
    if selected is None:
        raise RuntimeError("no empirical threshold candidate")

    selected["calibration_feasible"] = best_feasible is not None
    selected["search"] = {
        "score_candidate_count": int(len(score_candidates)),
        "margin_candidate_count": int(len(margin_candidates)),
        "threshold_pair_count": int(
            len(score_candidates) * len(margin_candidates)
        ),
        "feasible_pair_count": int(feasible_count),
    }
    return selected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("artifacts/p6_dev_k2/protocol"),
    )
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--open-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/"
            "mdsc_core30_open_set_eval.index.jsonl"
        ),
    )
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_shot_ablation_10_15"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output exists; refusing overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(
        (args.protocol_dir / "p6_k2_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("P6 protocol test seal violated")

    # Reuse the exact same cal/score partitions as the K=2 protocol.
    known_cal = read_jsonl(
        args.protocol_dir / "generic_dev_cal_known.jsonl"
    )
    known_score = read_jsonl(
        args.protocol_dir / "generic_dev_score_known.jsonl"
    )
    unknown_cal = read_jsonl(
        args.protocol_dir / "generic_dev_cal_unknown.jsonl"
    )
    unknown_score = read_jsonl(
        args.protocol_dir / "generic_dev_score_unknown.jsonl"
    )

    attention_meta = protocol["representative_checkpoints"]["attention"]
    checkpoint = Path(attention_meta["checkpoint"])
    if sha256(checkpoint) != attention_meta["checkpoint_sha256"]:
        raise RuntimeError("Attention checkpoint hash mismatch")

    device = torch.device(args.device)
    dr = load_attention_dr(checkpoint, device)

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )

    print("[shot ablation] embedding known train/dev...")
    train_emb = embed_dataset(train_ds, dr, device)
    dev_emb = embed_dataset(dev_ds, dr, device)

    wanted_unknown = {
        row["utt_id"] for row in unknown_cal + unknown_score
    }
    print("[shot ablation] embedding open-set DEV once...")
    backbone = WavLMModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    backbone.eval().to(device)
    for p_ in backbone.parameters():
        p_.requires_grad_(False)

    unknown_emb = embed_unknown_dev(
        open_index=args.open_index,
        mdsc_root=args.mdsc_root,
        wanted_utt_ids=wanted_unknown,
        backbone=backbone,
        dr=dr,
        device=device,
    )

    enrollment_sets = build_enrollment_sets(train_ds)
    results = {}

    for shot in SHOTS:
        print("=" * 108)
        print(f"P6 {shot}-SHOT MEAN PROTOTYPE")
        print("=" * 108)

        enrollment = enrollment_sets[shot]
        prototype = build_mean_prototypes(
            enrollment,
            train_emb,
        )

        cal_known = score_rows(
            known_cal,
            dev_emb,
            prototype,
            known=True,
        )
        cal_unknown = score_rows(
            unknown_cal,
            unknown_emb,
            prototype,
            known=False,
        )
        score_known = score_rows(
            known_score,
            dev_emb,
            prototype,
            known=True,
        )
        score_unknown = score_rows(
            unknown_score,
            unknown_emb,
            prototype,
            known=False,
        )

        calibrated = exact_calibrate(
            cal_known,
            cal_unknown,
        )
        score_metrics = scalar_metrics(
            score_known,
            score_unknown,
            calibrated["score_threshold"],
            calibrated["margin_threshold"],
        )
        passed = bool(
            calibrated["calibration_feasible"]
            and gate_pass(score_metrics)
        )

        results[str(shot)] = {
            "shots_per_intent": shot,
            "prototype_policy": "mean_then_l2",
            "enrollment_count": len(enrollment),
            "calibration": calibrated,
            "generic_dev_score_metrics": score_metrics,
            "absolute_gate_pass": passed,
        }

        print(
            "calibration feasible: "
            f"{calibrated['calibration_feasible']}"
        )
        print(
            "feasible threshold pairs: "
            f"{calibrated['search']['feasible_pair_count']:,}"
        )
        print(
            "score/margin threshold: "
            f"{calibrated['score_threshold']:.12f} / "
            f"{calibrated['margin_threshold']:.12f}"
        )
        for name, value in score_metrics.items():
            if isinstance(value, float):
                print(f"{name:20s} {value:.6f}")
        print(f"absolute Gate:       {'PASS' if passed else 'FAIL'}")

    m10 = results["10"]["generic_dev_score_metrics"]
    m15 = results["15"]["generic_dev_score_metrics"]

    output = {
        "schema": "papr_ssl.p6_shot_ablation_10_15.v1",
        "phase": "P6",
        "purpose": (
            "development shot-scaling ablation before changing Teacher training"
        ),
        "teacher": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "hidden_state_index": HIDDEN_STATE_INDEX,
            "head": "AttentionDR",
            "embedding_dim": 64,
            "checkpoint": attention_meta,
        },
        "shots_evaluated": list(SHOTS),
        "prototype_policy": (
            "one mean prototype per intent using all real enrollment shots"
        ),
        "dev_partition_policy": (
            "identical generic_dev_cal/generic_dev_score partitions "
            "to previous P6 K=2 experiment"
        ),
        "absolute_gates": GATES,
        "results": results,
        "delta_15_minus_10": {
            key: float(m15[key] - m10[key])
            for key in (
                "macro_f1",
                "correct_accept",
                "wrong_intent",
                "known_reject",
                "unknown_reject",
                "unknown_accept_far",
            )
        },
        "generic_test": "sealed_not_accessed",
        "note": (
            "generic_dev_score has already been observed in prior P6 iterations; "
            "this experiment is development ablation, not an untouched final Gate"
        ),
    }

    out_path = args.output_dir / "p6_10_15_shot_results.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 10-SHOT vs 15-SHOT SUMMARY")
    print("=" * 108)
    for shot in SHOTS:
        r = results[str(shot)]
        m = r["generic_dev_score_metrics"]
        print(
            f"{shot:2d}-shot "
            f"Gate={'PASS' if r['absolute_gate_pass'] else 'FAIL'} "
            f"F1={m['macro_f1']:.6f} "
            f"CA={m['correct_accept']:.6f} "
            f"WI={m['wrong_intent']:.6f} "
            f"KR={m['known_reject']:.6f} "
            f"UR={m['unknown_reject']:.6f}"
        )

    print("-" * 108)
    d = output["delta_15_minus_10"]
    print("15-shot minus 10-shot:")
    print(
        f"MacroF1={d['macro_f1']:+.6f} "
        f"CA={d['correct_accept']:+.6f} "
        f"WI={d['wrong_intent']:+.6f} "
        f"KR={d['known_reject']:+.6f} "
        f"UR={d['unknown_reject']:+.6f}"
    )
    print("generic_test accessed: NO")
    print("P6 SHOT ABLATION STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
