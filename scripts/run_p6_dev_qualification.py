#!/usr/bin/env python
"""P6-01..P6-05 development qualification.

Mainline:
- WavLM-large hidden_states[15]
- Attention DR 64D
- representative checkpoint selected before P6 scoring

Control / E0b baseline:
- same WavLM layer
- Mean DR 64D
- representative checkpoint selected by same rule

Uses generic_enrollment + generic_dev_cal + generic_dev_score only.
generic_test is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WavLMModel

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import (
    FrameCacheDataset,
    frame_collate,
)
from papr_ssl.training.teacher.real_experiment import (
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)


MODEL_ID = "microsoft/wavlm-large"
MODEL_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
LAYER = 15
DIM = 1024
SR = 16000

GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}
E0B_MAX_DROP = 0.02


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_checkpoint_dr(method: str, checkpoint_path: Path, device) -> torch.nn.Module:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if obj.get("method") != method:
        raise RuntimeError(
            f"checkpoint method mismatch: expected {method}, got {obj.get('method')}"
        )
    dr = build_dr(method)
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)
    for p in dr.parameters():
        p.requires_grad_(False)
    return dr


@torch.inference_mode()
def embed_known(
    dataset: FrameCacheDataset,
    drs: dict[str, torch.nn.Module],
    device,
) -> dict[str, dict[str, torch.Tensor]]:
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )
    out = {method: {} for method in drs}
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        m = batch["frame_mask"].to(device, non_blocking=True)
        for method, dr in drs.items():
            z = dr(x, m).cpu()
            for utt_id, emb in zip(batch["utt_ids"], z):
                out[method][utt_id] = emb.contiguous()
    return out


def load_wave(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SR:
        raise RuntimeError(f"expected 16 kHz, got {sr}: {path}")
    x = torch.from_numpy(data).to(torch.float32).mean(dim=1)
    if x.numel() <= 0 or not torch.isfinite(x).all():
        raise RuntimeError(f"invalid audio: {path}")
    return x.contiguous()


@torch.inference_mode()
def embed_unknown_dev(
    *,
    open_index: Path,
    mdsc_root: Path,
    wanted_utt_ids: set[str],
    backbone: WavLMModel,
    drs: dict[str, torch.nn.Module],
    device,
) -> dict[str, dict[str, torch.Tensor]]:
    rows = load_real_task_rows(open_index, splits=("dev",))
    rows = [r for r in rows if r.utt_id in wanted_utt_ids]
    if len(rows) != len(wanted_utt_ids):
        missing = wanted_utt_ids - {r.utt_id for r in rows}
        raise RuntimeError(f"missing open-set rows: {len(missing)}")

    legacy = [
        r for r in rows
        if r.audio_relpath is None and qualified_utt_relpath(r) is None
    ]
    catalog = build_mdsc_audio_catalog(mdsc_root) if legacy else None

    out = {method: {} for method in drs}
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=catalog,
        )
        wav = load_wave(path).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long, device=device)
        model_out = backbone(
            input_values=wav,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = model_out.hidden_states[LAYER].to(torch.float32)
        frame_mask = backbone._get_feature_vector_attention_mask(
            hs.shape[1],
            mask,
        ).to(torch.bool)
        for method, dr in drs.items():
            z = dr(hs, frame_mask).squeeze(0).cpu().contiguous()
            out[method][row.utt_id] = z
        if i % 100 == 0 or i == len(rows):
            print(f"[unknown dev] {i}/{len(rows)}")
    return out


def build_prototypes(
    enrollment_rows: list[dict],
    embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    by_class: dict[int, list[torch.Tensor]] = {}
    for row in enrollment_rows:
        by_class.setdefault(int(row["label_index"]), []).append(
            embeddings[row["utt_id"]]
        )
    if sorted(by_class) != list(range(30)):
        raise RuntimeError("prototype classes are not exactly 0..29")
    protos = []
    for c in range(30):
        p = torch.stack(by_class[c], dim=0).mean(dim=0)
        protos.append(F.normalize(p, p=2, dim=0))
    return torch.stack(protos, dim=0)


def scores_for(
    rows: list[dict],
    embeddings: dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    known: bool,
) -> dict[str, np.ndarray]:
    z = torch.stack([embeddings[r["utt_id"]] for r in rows], dim=0)
    sim = F.normalize(z, p=2, dim=-1) @ F.normalize(
        prototypes, p=2, dim=-1
    ).T
    top2 = torch.topk(sim, k=2, dim=1)
    pred = top2.indices[:, 0]
    score = top2.values[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]
    result = {
        "score": score.numpy(),
        "margin": margin.numpy(),
        "pred": pred.numpy(),
    }
    if known:
        result["true"] = np.asarray(
            [int(r["label_index"]) for r in rows],
            dtype=np.int64,
        )
    return result


def macro_f1_after_reject(
    true: np.ndarray,
    pred: np.ndarray,
    accepted: np.ndarray,
) -> float:
    f1s = []
    for c in range(30):
        pred_c = accepted & (pred == c)
        true_c = true == c
        tp = int(np.sum(pred_c & true_c))
        fp = int(np.sum(pred_c & ~true_c))
        fn = int(np.sum(~pred_c & true_c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1s.append(f1)
    return float(np.mean(f1s))


def metrics_at(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
    score_threshold: float,
    margin_threshold: float,
) -> dict:
    ka = (
        (known["score"] >= score_threshold)
        & (known["margin"] >= margin_threshold)
    )
    ua = (
        (unknown["score"] >= score_threshold)
        & (unknown["margin"] >= margin_threshold)
    )
    correct = ka & (known["pred"] == known["true"])
    wrong = ka & (known["pred"] != known["true"])
    reject = ~ka

    return {
        "macro_f1": macro_f1_after_reject(
            known["true"], known["pred"], ka
        ),
        "correct_accept": float(np.mean(correct)),
        "wrong_intent": float(np.mean(wrong)),
        "known_reject": float(np.mean(reject)),
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


def violation(m: dict) -> float:
    terms = [
        max(0.0, GATES["macro_f1_min"] - m["macro_f1"])
        / GATES["macro_f1_min"],
        max(0.0, GATES["correct_accept_min"] - m["correct_accept"])
        / GATES["correct_accept_min"],
        max(0.0, m["wrong_intent"] - GATES["wrong_intent_max"])
        / GATES["wrong_intent_max"],
        max(0.0, m["known_reject"] - GATES["known_reject_max"])
        / GATES["known_reject_max"],
        max(0.0, GATES["unknown_reject_min"] - m["unknown_reject"])
        / GATES["unknown_reject_min"],
    ]
    return float(sum(terms))


def candidates(values: np.ndarray) -> list[float]:
    qs = np.linspace(0.0, 1.0, 101)
    vals = np.unique(np.quantile(values, qs))
    eps = 1e-6
    return [float(vals[0] - eps)] + [float(x) for x in vals] + [
        float(vals[-1] + eps)
    ]


def calibrate(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
) -> dict:
    score_values = np.concatenate([known["score"], unknown["score"]])
    margin_values = np.concatenate([known["margin"], unknown["margin"]])
    score_grid = candidates(score_values)
    margin_grid = candidates(margin_values)

    feasible = []
    diagnostic = []

    for ts in score_grid:
        for tm in margin_grid:
            m = metrics_at(known, unknown, ts, tm)
            row = {
                "score_threshold": ts,
                "margin_threshold": tm,
                "metrics": m,
            }
            if gate_pass(m):
                feasible.append(row)
            else:
                row["violation"] = violation(m)
                diagnostic.append(row)

    if feasible:
        best = max(
            feasible,
            key=lambda x: (
                x["metrics"]["macro_f1"],
                x["metrics"]["correct_accept"],
                x["metrics"]["unknown_reject"],
                -x["metrics"]["wrong_intent"],
                -x["metrics"]["known_reject"],
                x["score_threshold"],
                x["margin_threshold"],
            ),
        )
        best["calibration_feasible"] = True
        return best

    best = min(
        diagnostic,
        key=lambda x: (
            x["violation"],
            -x["metrics"]["macro_f1"],
            -x["metrics"]["correct_accept"],
            -x["metrics"]["unknown_reject"],
        ),
    )
    best["calibration_feasible"] = False
    return best


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("artifacts/p6_dev/protocol"),
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
        default=Path("artifacts/p6_dev/qualification"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"qualification output exists; refusing overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(
        (args.protocol_dir / "p6_protocol.json").read_text(encoding="utf-8")
    )
    if protocol["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("P6 protocol test seal violated")

    enrollment = read_jsonl(args.protocol_dir / "generic_enrollment.jsonl")
    known_cal = read_jsonl(args.protocol_dir / "generic_dev_cal_known.jsonl")
    known_score = read_jsonl(args.protocol_dir / "generic_dev_score_known.jsonl")
    unknown_cal = read_jsonl(args.protocol_dir / "generic_dev_cal_unknown.jsonl")
    unknown_score = read_jsonl(args.protocol_dir / "generic_dev_score_unknown.jsonl")

    device = torch.device(args.device)

    drs = {}
    checkpoint_meta = {}
    for method in ("mean", "attention"):
        meta = protocol["representative_checkpoints"][method]
        checkpoint = Path(meta["checkpoint"])
        if sha256(checkpoint) != meta["checkpoint_sha256"]:
            raise RuntimeError(f"{method}: checkpoint hash mismatch")
        drs[method] = load_checkpoint_dr(method, checkpoint, device)
        checkpoint_meta[method] = meta

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

    print("[P6] embedding known train/dev from frozen frame cache...")
    known_train_emb = embed_known(train_ds, drs, device)
    known_dev_emb = embed_known(dev_ds, drs, device)

    wanted_unknown = {
        row["utt_id"] for row in unknown_cal + unknown_score
    }

    print("[P6] embedding open-set dev with frozen WavLM[15]...")
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
        drs=drs,
        device=device,
    )
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {}
    for method in ("mean", "attention"):
        proto = build_prototypes(enrollment, known_train_emb[method])

        cal_known_scores = scores_for(
            known_cal, known_dev_emb[method], proto, known=True
        )
        cal_unknown_scores = scores_for(
            unknown_cal, unknown_emb[method], proto, known=False
        )
        calibrated = calibrate(cal_known_scores, cal_unknown_scores)

        score_known_scores = scores_for(
            known_score, known_dev_emb[method], proto, known=True
        )
        score_unknown_scores = scores_for(
            unknown_score, unknown_emb[method], proto, known=False
        )
        score_metrics = metrics_at(
            score_known_scores,
            score_unknown_scores,
            calibrated["score_threshold"],
            calibrated["margin_threshold"],
        )
        score_pass = (
            calibrated["calibration_feasible"]
            and gate_pass(score_metrics)
        )

        results[method] = {
            "checkpoint": checkpoint_meta[method],
            "prototype_policy": protocol["p6_01"],
            "calibration": calibrated,
            "generic_dev_score_metrics": score_metrics,
            "generic_dev_score_gate_pass": score_pass,
        }

        print("=" * 108)
        print(f"P6 {method.upper()} DEV QUALIFICATION")
        print("=" * 108)
        print(
            f"threshold score/margin: "
            f"{calibrated['score_threshold']:.6f} / "
            f"{calibrated['margin_threshold']:.6f}"
        )
        print(f"cal feasible:    {calibrated['calibration_feasible']}")
        for key, val in score_metrics.items():
            if isinstance(val, float):
                print(f"{key:20s} {val:.6f}")
        print(f"dev score Gate: {score_pass}")

    mainline = results["attention"]
    baseline = results["mean"]
    relative_drop = (
        baseline["generic_dev_score_metrics"]["macro_f1"]
        - mainline["generic_dev_score_metrics"]["macro_f1"]
    )
    e0b_pass = relative_drop <= E0B_MAX_DROP

    output = {
        "schema": "papr_ssl.p6_dev_qualification.v1",
        "phase": "P6",
        "mainline_method": "attention",
        "e0b_baseline_method": "mean",
        "absolute_gates": GATES,
        "results": results,
        "p6_04_mainline_absolute_gate_pass": bool(
            mainline["generic_dev_score_gate_pass"]
        ),
        "p6_05_e0b": {
            "baseline": "mean_64d_head",
            "mainline": "attention_64d_head",
            "baseline_macro_f1": baseline[
                "generic_dev_score_metrics"
            ]["macro_f1"],
            "mainline_macro_f1": mainline[
                "generic_dev_score_metrics"
            ]["macro_f1"],
            "baseline_minus_mainline_drop": relative_drop,
            "max_allowed_drop": E0B_MAX_DROP,
            "pass": bool(e0b_pass),
        },
        "eligible_for_p6_06_freeze": bool(
            mainline["generic_dev_score_gate_pass"] and e0b_pass
        ),
        "generic_test": "sealed_not_accessed",
    }
    out_path = args.output_dir / "p6_dev_gate.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 DEVELOPMENT GATE SUMMARY")
    print("=" * 108)
    print(
        "P6-04 absolute Gate: "
        f"{'PASS' if output['p6_04_mainline_absolute_gate_pass'] else 'FAIL'}"
    )
    print(
        "P6-05 E0b drop Gate: "
        f"{'PASS' if e0b_pass else 'FAIL'} "
        f"(baseline-mainline={relative_drop:+.6f}, max={E0B_MAX_DROP:.6f})"
    )
    print(
        "eligible for P6-06 freeze: "
        f"{output['eligible_for_p6_06_freeze']}"
    )
    print("generic_test accessed: NO")
    return 0 if output["eligible_for_p6_06_freeze"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
