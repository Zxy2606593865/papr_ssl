#!/usr/bin/env python
"""H5-01 — C/W/U Decision Head for unseen-speaker personalized open-set recognition.

This is the first Head that learns the BUSINESS decision event:

C = query is registered and the current top1 phrase is correct
W = query is registered but the current top1 phrase is wrong
U = query is not registered for this user

Backbone/Neck/ranking stay frozen.
The ranking baseline remains the promoted fixed fusion:
    0.5 * Global + 0.5 * DTW

Data protocol
-------------
TRAIN speakers are split speaker-wise into:
    HEAD-FIT speakers       -> fit the linear C/W/U Head
    CALIBRATION speakers    -> choose acceptance threshold

DEV speakers (4 unseen speakers):
    evaluation only

Each episode:
    20 Core30 phrases registered
    10 Core30 phrases unregistered (user-relative Unknown)
    support = 1-shot or 2-shot
    remaining registered recordings = known queries
    all recordings of unregistered phrases = unknown queries

Decision comparison
-------------------
Baseline:
    accept if fixed-fusion top1 score >= scalar threshold

C/W/U Head:
    build query-level evidence
    -> shared linear 3-class Head
    -> softmax decision scores (NOT claimed calibrated probabilities)
    -> accept only when argmax == C and score_C >= threshold

Both thresholds are selected on CALIBRATION speakers under:
    Unknown FAR <= 10%
and maximize Correct Accept.

generic_test remains sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SHOT_LEVELS = (1, 2)
FIT_REPEATS = 3
CAL_REPEATS = 10
DEV_REPEATS = 20
REGISTERED_CLASSES = 20
TOP_K = 3
FUSION_LAMBDA = 0.50
TARGET_UNKNOWN_FAR = 0.10

FEATURE_NAMES = (
    "fused_top1",
    "fused_margin",
    "global_top1",
    "global_margin",
    "dtw_top1",
    "dtw_margin",
    "global_dtw_winner_agree",
    "duration_log_ratio",
    "log_support_n",
    "support_cosine_mean",
    "support_cosine_min",
)

EVENT_TO_INDEX = {"C": 0, "W": 1, "U": 2}
INDEX_TO_EVENT = {0: "C", 1: "W", 2: "U"}


def stable_hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def stable_support(indices, shot, repeat, speaker, label, utt_ids):
    return sorted(
        indices,
        key=lambda i: stable_hash(
            "support", repeat, speaker, label, utt_ids[i]
        ),
    )[:shot]


def choose_registered(labels, repeat, speaker):
    # User-relative vocabulary changes by repeat and speaker.
    ranked = sorted(
        labels,
        key=lambda ph: stable_hash(
            "registered", repeat, speaker, ph
        ),
    )
    return set(ranked[:REGISTERED_CLASSES])


def load_temporal_index(feature_dir: Path):
    index = {}
    path = feature_dir / "temporal" / "index.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index[row["utt_id"]] = {
            **row,
            "path": feature_dir / "temporal" / row["cache_relpath"],
        }
    return index


class TemporalStore:
    def __init__(self, index):
        self.index = index
        self.cache = {}
        self.dtw_cache = {}

    def seq(self, utt_id):
        if utt_id not in self.cache:
            obj = torch.load(
                self.index[utt_id]["path"],
                map_location="cpu",
                weights_only=False,
            )
            x = obj["features"].to(torch.float32).numpy()
            self.cache[utt_id] = l2norm(x)
        return self.cache[utt_id]

    def frames(self, utt_id):
        return int(self.index[utt_id]["frames"])

    def dtw(self, a_id, b_id, band_ratio=0.25):
        key = tuple(sorted((a_id, b_id)))
        if key in self.dtw_cache:
            return self.dtw_cache[key]

        a = self.seq(a_id)
        b = self.seq(b_id)
        n, m = len(a), len(b)
        band = max(abs(n - m), int(math.ceil(band_ratio * max(n, m))))

        inf = np.inf
        prev_cost = np.full(m + 1, inf, dtype=np.float64)
        prev_len = np.zeros(m + 1, dtype=np.int32)
        prev_cost[0] = 0.0

        for i in range(1, n + 1):
            cur_cost = np.full(m + 1, inf, dtype=np.float64)
            cur_len = np.zeros(m + 1, dtype=np.int32)
            j0 = max(1, i - band)
            j1 = min(m, i + band)
            ai = a[i - 1]

            for j in range(j0, j1 + 1):
                local = 1.0 - float(
                    np.clip(np.dot(ai, b[j - 1]), -1.0, 1.0)
                )
                candidates = (
                    (prev_cost[j], prev_len[j]),
                    (cur_cost[j - 1], cur_len[j - 1]),
                    (prev_cost[j - 1], prev_len[j - 1]),
                )
                best_cost, best_len = min(candidates, key=lambda x: x[0])
                if not np.isfinite(best_cost):
                    continue
                cur_cost[j] = best_cost + local
                cur_len[j] = best_len + 1

            prev_cost, prev_len = cur_cost, cur_len

        if not np.isfinite(prev_cost[m]) or prev_len[m] <= 0:
            raise RuntimeError(f"No DTW path for {a_id}, {b_id}")

        value = float(prev_cost[m] / prev_len[m])
        self.dtw_cache[key] = value
        return value


def load_split(npz, split):
    return {
        "utt_id": npz[f"{split}_utt_id"].astype(str),
        "speaker_id": npz[f"{split}_speaker_id"].astype(str),
        "label": npz[f"{split}_label"].astype(str),
        "label_index": npz[f"{split}_label_index"].astype(np.int64),
        "embedding": l2norm(npz[f"{split}_embedding"].astype(np.float64)),
    }


def groups_for_split(data):
    groups = defaultdict(list)
    for i, (speaker, label) in enumerate(
        zip(data["speaker_id"], data["label"])
    ):
        groups[(speaker, label)].append(i)
    return groups


def split_train_speakers(speakers, cal_count=6):
    ranked = sorted(
        speakers,
        key=lambda s: stable_hash("h5_calibration_speaker", s),
    )
    cal = set(ranked[:cal_count])
    fit = set(ranked[cal_count:])
    return fit, cal


def build_episode_features(
    *,
    data,
    groups,
    temporal,
    speaker,
    shot,
    repeat,
):
    labels = sorted(set(data["label"]))
    registered = choose_registered(labels, repeat, speaker)
    registered_sorted = sorted(registered)

    support = {}
    known_queries = []
    unknown_queries = []

    for label in labels:
        idx = groups[(speaker, label)]
        if label in registered:
            if len(idx) < shot + 1:
                # Skip labels that cannot support this shot + query.
                continue
            sup = stable_support(
                idx, shot, repeat, speaker, label, data["utt_id"]
            )
            support[label] = sup
            sup_set = set(sup)
            known_queries.extend(
                (i, label) for i in idx if i not in sup_set
            )
        else:
            unknown_queries.extend((i, label) for i in idx)

    # Keep only registered labels that survived shot feasibility.
    registered_sorted = [l for l in registered_sorted if l in support]
    if len(registered_sorted) < 2:
        return []

    prototypes = []
    for label in registered_sorted:
        p = l2norm(
            data["embedding"][support[label]].mean(axis=0, keepdims=True)
        )[0]
        prototypes.append(p)
    prototypes = np.stack(prototypes, axis=0)

    rows = []

    def process(query_idx, true_label, is_unknown):
        qz = data["embedding"][query_idx]
        q_utt = data["utt_id"][query_idx]
        q_frames = temporal.frames(q_utt)

        global_scores = qz @ prototypes.T
        global_order = np.argsort(-global_scores)
        candidates = global_order[: min(TOP_K, len(global_order))]

        dtw_sims = []
        duration = []
        support_cos_mean = []
        support_cos_min = []

        for local_c in candidates:
            label = registered_sorted[int(local_c)]
            sims = []
            frame_counts = []
            cosines = []

            for sidx in support[label]:
                s_utt = data["utt_id"][sidx]
                d = temporal.dtw(q_utt, s_utt)
                sims.append(1.0 - 0.5 * d)
                frame_counts.append(temporal.frames(s_utt))
                cosines.append(float(qz @ data["embedding"][sidx]))

            dtw_sims.append(float(max(sims)))
            med_frames = float(np.median(frame_counts))
            duration.append(
                abs(math.log(max(q_frames, 1) / max(med_frames, 1.0)))
            )
            support_cos_mean.append(float(np.mean(cosines)))
            support_cos_min.append(float(np.min(cosines)))

        dtw_sims = np.asarray(dtw_sims, dtype=np.float64)
        fused = (
            FUSION_LAMBDA * global_scores[candidates]
            + (1.0 - FUSION_LAMBDA) * dtw_sims
        )
        best_pos = int(np.argmax(fused))
        best_local = int(candidates[best_pos])
        pred_label = registered_sorted[best_local]

        # Top1/top2 evidence in the candidate space.
        fused_order = np.argsort(-fused)
        top1_pos = int(fused_order[0])
        top2_pos = int(fused_order[1]) if len(fused_order) > 1 else top1_pos

        top1_local = int(candidates[top1_pos])
        top2_local = int(candidates[top2_pos])

        global_top1_local = int(candidates[int(np.argmax(global_scores[candidates]))])
        dtw_top1_local = int(candidates[int(np.argmax(dtw_sims))])

        fused_top1 = float(fused[top1_pos])
        fused_top2 = float(fused[top2_pos])
        global_top1 = float(global_scores[top1_local])
        global_top2 = float(global_scores[top2_local])
        dtw_top1 = float(dtw_sims[top1_pos])
        dtw_top2 = float(dtw_sims[top2_pos])

        feat = np.asarray(
            [
                fused_top1,
                fused_top1 - fused_top2,
                global_top1,
                global_top1 - global_top2,
                dtw_top1,
                dtw_top1 - dtw_top2,
                1.0 if global_top1_local == dtw_top1_local else 0.0,
                float(duration[top1_pos]),
                math.log(float(shot)),
                float(support_cos_mean[top1_pos]),
                float(support_cos_min[top1_pos]),
            ],
            dtype=np.float32,
        )

        if is_unknown:
            event = "U"
        elif pred_label == true_label:
            event = "C"
        else:
            event = "W"

        rows.append(
            {
                "speaker_id": speaker,
                "shot": shot,
                "repeat": repeat,
                "query_utt": q_utt,
                "true_label": true_label,
                "pred_label": pred_label,
                "event": event,
                "is_unknown": bool(is_unknown),
                "feature": feat,
                "baseline_top1_score": fused_top1,
            }
        )

    for qidx, label in known_queries:
        process(qidx, label, False)
    for qidx, label in unknown_queries:
        process(qidx, label, True)

    return rows


def collect_rows(
    *,
    data,
    groups,
    temporal,
    speakers,
    shots,
    repeats,
):
    rows = []
    for shot in shots:
        for repeat in range(repeats):
            for speaker in sorted(speakers):
                rows.extend(
                    build_episode_features(
                        data=data,
                        groups=groups,
                        temporal=temporal,
                        speaker=speaker,
                        shot=shot,
                        repeat=repeat,
                    )
                )
            print(
                f"[build] shot={shot} repeat={repeat+1}/{repeats} "
                f"rows={len(rows)}"
            )
    return rows


def rows_to_arrays(rows):
    X = np.stack([r["feature"] for r in rows], axis=0)
    y = np.asarray(
        [EVENT_TO_INDEX[r["event"]] for r in rows],
        dtype=np.int64,
    )
    baseline = np.asarray(
        [r["baseline_top1_score"] for r in rows],
        dtype=np.float64,
    )
    shot = np.asarray([r["shot"] for r in rows], dtype=np.int64)
    return X, y, baseline, shot


def fit_cwu_head(X, y, epochs=600, lr=0.03, weight_decay=2e-3):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xs = (X - mean) / std

    xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    counts = np.bincount(y, minlength=3).astype(np.float64)
    # Mild balancing: inverse sqrt frequency, normalized to mean 1.
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    torch.manual_seed(20260915)
    model = torch.nn.Linear(X.shape[1], 3)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best = None
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(xt)
        loss = F.cross_entropy(
            logits,
            yt,
            weight=class_weights,
        )
        loss.backward()
        opt.step()

        value = float(loss.detach())
        if best is None or value < best["loss"]:
            best = {
                "loss": value,
                "epoch": epoch,
                "state": {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                },
            }

    model.load_state_dict(best["state"])
    return model, mean, std, counts, weights, best


def predict_scores(model, mean, std, X):
    xt = torch.tensor(
        (X - mean) / std,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        logits = model(xt)
        scores = torch.softmax(logits, dim=-1).numpy()
    return scores


def select_baseline_threshold(rows, target_far):
    # shot-specific threshold chosen on calibration speakers.
    known = [r for r in rows if r["event"] != "U"]
    unknown = [r for r in rows if r["event"] == "U"]

    known_scores = np.asarray(
        [r["baseline_top1_score"] for r in known],
        dtype=np.float64,
    )
    known_correct = np.asarray(
        [r["event"] == "C" for r in known],
        dtype=bool,
    )
    unknown_scores = np.asarray(
        [r["baseline_top1_score"] for r in unknown],
        dtype=np.float64,
    )

    candidates = np.unique(
        np.concatenate([known_scores, unknown_scores, [np.inf, -np.inf]])
    )
    best = None
    for t in candidates:
        far = float(np.mean(unknown_scores >= t))
        if far > target_far + 1e-12:
            continue
        ca = float(np.mean((known_scores >= t) & known_correct))
        wi = float(np.mean((known_scores >= t) & ~known_correct))
        key = (ca, -wi, -far, t)
        if best is None or key > best["key"]:
            best = {
                "threshold": float(t),
                "cal_ca": ca,
                "cal_wi": wi,
                "cal_unknown_far": far,
                "key": key,
            }
    if best is None:
        raise RuntimeError("No feasible baseline threshold")
    best.pop("key")
    return best


def select_head_threshold(rows, scores, target_far):
    y = np.asarray(
        [EVENT_TO_INDEX[r["event"]] for r in rows],
        dtype=np.int64,
    )
    c_score = scores[:, EVENT_TO_INDEX["C"]]
    argmax = scores.argmax(axis=1)
    accept_candidate = argmax == EVENT_TO_INDEX["C"]

    unknown_mask = y == EVENT_TO_INDEX["U"]
    known_mask = ~unknown_mask
    known_correct = y == EVENT_TO_INDEX["C"]
    known_wrong = y == EVENT_TO_INDEX["W"]

    candidates = np.unique(
        np.concatenate([c_score, [np.inf, -np.inf]])
    )
    best = None

    for t in candidates:
        accept = accept_candidate & (c_score >= t)
        far = float(np.mean(accept[unknown_mask]))
        if far > target_far + 1e-12:
            continue

        ca = float(np.mean(accept[known_mask] & known_correct[known_mask]))
        wi = float(np.mean(accept[known_mask] & known_wrong[known_mask]))
        key = (ca, -wi, -far, t)
        if best is None or key > best["key"]:
            best = {
                "threshold": float(t),
                "cal_ca": ca,
                "cal_wi": wi,
                "cal_unknown_far": far,
                "key": key,
            }

    if best is None:
        raise RuntimeError("No feasible Head threshold")
    best.pop("key")
    return best


def evaluate_baseline(rows, threshold):
    known = [r for r in rows if r["event"] != "U"]
    unknown = [r for r in rows if r["event"] == "U"]

    kscore = np.asarray([r["baseline_top1_score"] for r in known])
    kcorrect = np.asarray([r["event"] == "C" for r in known])
    uscore = np.asarray([r["baseline_top1_score"] for r in unknown])

    accept_k = kscore >= threshold
    accept_u = uscore >= threshold

    ca = float(np.mean(accept_k & kcorrect))
    wi = float(np.mean(accept_k & ~kcorrect))
    kr = float(np.mean(~accept_k))
    ua = float(np.mean(accept_u))
    ur = 1.0 - ua
    return {
        "correct_accept": ca,
        "wrong_intent": wi,
        "known_reject": kr,
        "unknown_accept": ua,
        "unknown_reject": ur,
    }


def evaluate_head(rows, scores, threshold):
    y = np.asarray(
        [EVENT_TO_INDEX[r["event"]] for r in rows],
        dtype=np.int64,
    )
    c_score = scores[:, EVENT_TO_INDEX["C"]]
    pred_event = scores.argmax(axis=1)

    accept = (
        (pred_event == EVENT_TO_INDEX["C"])
        & (c_score >= threshold)
    )

    unknown_mask = y == EVENT_TO_INDEX["U"]
    known_mask = ~unknown_mask
    correct_mask = y == EVENT_TO_INDEX["C"]
    wrong_mask = y == EVENT_TO_INDEX["W"]

    ca = float(np.mean(accept[known_mask] & correct_mask[known_mask]))
    wi = float(np.mean(accept[known_mask] & wrong_mask[known_mask]))
    kr = float(np.mean(~accept[known_mask]))
    ua = float(np.mean(accept[unknown_mask]))
    ur = 1.0 - ua

    # Event classification diagnostic.
    event_acc = float(np.mean(pred_event == y))
    f1s = []
    for c in range(3):
        tp = np.sum((pred_event == c) & (y == c))
        fp = np.sum((pred_event == c) & (y != c))
        fn = np.sum((pred_event != c) & (y == c))
        den = 2 * tp + fp + fn
        f1s.append(float(2 * tp / den) if den else 0.0)

    return {
        "correct_accept": ca,
        "wrong_intent": wi,
        "known_reject": kr,
        "unknown_accept": ua,
        "unknown_reject": ur,
        "event_accuracy": event_acc,
        "event_macro_f1": float(np.mean(f1s)),
        "event_f1_C": f1s[0],
        "event_f1_W": f1s[1],
        "event_f1_U": f1s[2],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head"),
    )
    p.add_argument("--fit-repeats", type=int, default=FIT_REPEATS)
    p.add_argument("--cal-repeats", type=int, default=CAL_REPEATS)
    p.add_argument("--dev-repeats", type=int, default=DEV_REPEATS)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )
    train = load_split(d, "train")
    dev = load_split(d, "dev")
    train_groups = groups_for_split(train)
    dev_groups = groups_for_split(dev)

    train_speakers = sorted(set(train["speaker_id"]))
    dev_speakers = sorted(set(dev["speaker_id"]))
    if len(train_speakers) != 34 or len(dev_speakers) != 4:
        raise RuntimeError(
            f"Expected 34 TRAIN / 4 DEV speakers, got "
            f"{len(train_speakers)} / {len(dev_speakers)}"
        )

    fit_speakers, cal_speakers = split_train_speakers(
        train_speakers, cal_count=6
    )
    if len(fit_speakers) != 28 or len(cal_speakers) != 6:
        raise RuntimeError("Expected 28 fit / 6 calibration TRAIN speakers")

    temporal_index = load_temporal_index(args.feature_dir)
    temporal = TemporalStore(temporal_index)

    print("=" * 108)
    print("H5-01 BUILD HEAD-FIT EVIDENCE")
    print("=" * 108)
    fit_rows = collect_rows(
        data=train,
        groups=train_groups,
        temporal=temporal,
        speakers=fit_speakers,
        shots=SHOT_LEVELS,
        repeats=args.fit_repeats,
    )

    X_fit, y_fit, _, _ = rows_to_arrays(fit_rows)
    model, mean, std, counts, class_weights, fit_info = fit_cwu_head(
        X_fit, y_fit
    )

    head = {
        "schema": "papr_ssl.h5_01_cwu_linear_head.v1",
        "feature_names": list(FEATURE_NAMES),
        "event_order": ["C", "W", "U"],
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
        "weight": model.weight.detach().cpu().numpy().tolist(),
        "bias": model.bias.detach().cpu().numpy().tolist(),
        "fit_event_counts": {
            INDEX_TO_EVENT[i]: int(counts[i]) for i in range(3)
        },
        "class_weights_inverse_sqrt": {
            INDEX_TO_EVENT[i]: float(class_weights[i]) for i in range(3)
        },
        "fit_loss": fit_info["loss"],
        "fit_epoch": fit_info["epoch"],
        "fit_speakers": sorted(fit_speakers),
        "calibration_speakers": sorted(cal_speakers),
        "scores_are_calibrated_probabilities": False,
        "backbone_neck_updated": False,
    }
    (args.output_dir / "cwu_head.json").write_text(
        json.dumps(head, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H5-01 BUILD CALIBRATION EVIDENCE")
    print("=" * 108)
    cal_rows = collect_rows(
        data=train,
        groups=train_groups,
        temporal=temporal,
        speakers=cal_speakers,
        shots=SHOT_LEVELS,
        repeats=args.cal_repeats,
    )
    X_cal, _, _, cal_shot = rows_to_arrays(cal_rows)
    cal_scores = predict_scores(model, mean, std, X_cal)

    calibration = {}
    for shot in SHOT_LEVELS:
        mask = cal_shot == shot
        rows_s = [r for r, m in zip(cal_rows, mask) if m]
        scores_s = cal_scores[mask]

        baseline_t = select_baseline_threshold(
            rows_s, TARGET_UNKNOWN_FAR
        )
        head_t = select_head_threshold(
            rows_s, scores_s, TARGET_UNKNOWN_FAR
        )
        calibration[str(shot)] = {
            "baseline": baseline_t,
            "head": head_t,
        }

    print("=" * 108)
    print("H5-01 BUILD UNSEEN DEV EVIDENCE")
    print("=" * 108)
    dev_rows = collect_rows(
        data=dev,
        groups=dev_groups,
        temporal=temporal,
        speakers=set(dev_speakers),
        shots=SHOT_LEVELS,
        repeats=args.dev_repeats,
    )
    X_dev, _, _, dev_shot = rows_to_arrays(dev_rows)
    dev_scores = predict_scores(model, mean, std, X_dev)

    summary = {}
    detailed = []

    for shot in SHOT_LEVELS:
        mask = dev_shot == shot
        rows_s = [r for r, m in zip(dev_rows, mask) if m]
        scores_s = dev_scores[mask]

        base_m = evaluate_baseline(
            rows_s,
            calibration[str(shot)]["baseline"]["threshold"],
        )
        head_m = evaluate_head(
            rows_s,
            scores_s,
            calibration[str(shot)]["head"]["threshold"],
        )

        summary[str(shot)] = {
            "baseline": base_m,
            "cwu_head": head_m,
            "delta_head_minus_baseline": {
                "correct_accept": (
                    head_m["correct_accept"] - base_m["correct_accept"]
                ),
                "wrong_intent": (
                    head_m["wrong_intent"] - base_m["wrong_intent"]
                ),
                "known_reject": (
                    head_m["known_reject"] - base_m["known_reject"]
                ),
                "unknown_reject": (
                    head_m["unknown_reject"] - base_m["unknown_reject"]
                ),
            },
            "thresholds": calibration[str(shot)],
        }

    out = {
        "schema": "papr_ssl.h5_01_cwu_decision_head_eval.v1",
        "task": "unseen-speaker personalized open-set C/W/U decision",
        "protocol": {
            "train_head_fit_speakers": 28,
            "train_calibration_speakers": 6,
            "dev_unseen_speakers": 4,
            "registered_classes_per_episode": REGISTERED_CLASSES,
            "unregistered_classes_per_episode": 30 - REGISTERED_CLASSES,
            "shot_levels": list(SHOT_LEVELS),
            "fit_repeats": args.fit_repeats,
            "cal_repeats": args.cal_repeats,
            "dev_repeats": args.dev_repeats,
        },
        "ranking": {
            "system": "fixed Global+DTW fusion",
            "lambda": FUSION_LAMBDA,
            "top_k": TOP_K,
        },
        "head": {
            "type": "shared linear 3-class C/W/U",
            "event_definitions": {
                "C": "registered query and current top1 phrase is correct",
                "W": "registered query but current top1 phrase is wrong",
                "U": "query phrase is not registered for current user",
            },
            "scores_are_calibrated_probabilities": False,
        },
        "acceptance": {
            "baseline": "top1 fixed-fusion score >= calibrated threshold",
            "cwu_head": (
                "argmax event == C AND C-score >= calibrated threshold"
            ),
            "target_calibration_unknown_far": TARGET_UNKNOWN_FAR,
        },
        "summary": summary,
        "backbone_neck_updated": False,
        "learned_h4_ranker_used": False,
        "unknown_exposure_training_used": False,
        "loo_shrinkage_used": False,
        "cross_session_claim": False,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "h5_01_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H5-01 C/W/U DECISION HEAD SUMMARY")
    print("=" * 108)
    for shot in SHOT_LEVELS:
        s = summary[str(shot)]
        b = s["baseline"]
        h = s["cwu_head"]
        dlt = s["delta_head_minus_baseline"]
        print(
            f"{shot}-shot BASE | "
            f"CA={b['correct_accept']:.6f} "
            f"WI={b['wrong_intent']:.6f} "
            f"KR={b['known_reject']:.6f} "
            f"UR={b['unknown_reject']:.6f}"
        )
        print(
            f"{shot}-shot C/W/U| "
            f"CA={h['correct_accept']:.6f} "
            f"WI={h['wrong_intent']:.6f} "
            f"KR={h['known_reject']:.6f} "
            f"UR={h['unknown_reject']:.6f} "
            f"EventF1={h['event_macro_f1']:.6f}"
        )
        print(
            f"{shot}-shot DELTA | "
            f"dCA={dlt['correct_accept']:+.6f} "
            f"dWI={dlt['wrong_intent']:+.6f} "
            f"dKR={dlt['known_reject']:+.6f} "
            f"dUR={dlt['unknown_reject']:+.6f}"
        )

    print("-" * 108)
    print(f"Head-fit TRAIN speakers:  {len(fit_speakers)}")
    print(f"Calibration speakers:     {len(cal_speakers)}")
    print(f"Unseen DEV speakers:      {len(dev_speakers)}")
    print("H4 learned ranker used:   NO")
    print("Backbone/Neck updated:    NO")
    print("Unknown Exposure train:   NO")
    print("generic_test accessed:    NO")
    print("H5-01 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
