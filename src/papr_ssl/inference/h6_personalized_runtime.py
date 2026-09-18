"""Feature-level personalized runtime for PAPR-SSL H6.

This module is intentionally independent of raw-audio extraction.

Input:
    frozen global 256D embedding
    frozen temporal 256D sequence
    user enrollment memory

Output:
    ACCEPT / CONFIRM / REJECT
    intent_id / canonical_text
    decision_scores (C/W/U; NOT calibrated probabilities)

Raw WAV -> frozen features is added as a separate adapter so that the research
runtime can first be verified without changing the validated WavLM pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


EVENT_ORDER = ("C", "W", "U")


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _dtw_distance(a: np.ndarray, b: np.ndarray, band_ratio: float = 0.25) -> float:
    a = _l2(a)
    b = _l2(b)
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
            local = 1.0 - float(np.clip(np.dot(ai, b[j - 1]), -1.0, 1.0))
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
        raise RuntimeError("No valid DTW path")
    return float(prev_cost[m] / prev_len[m])


@dataclass
class EnrollmentExample:
    global_embedding: np.ndarray
    temporal_sequence: np.ndarray


@dataclass
class IntentMemory:
    intent_id: str
    canonical_text: str
    examples: list[EnrollmentExample] = field(default_factory=list)


@dataclass
class UserMemory:
    user_id: str
    intents: dict[str, IntentMemory] = field(default_factory=dict)

    def enroll(
        self,
        *,
        intent_id: str,
        canonical_text: str,
        global_embedding: np.ndarray,
        temporal_sequence: np.ndarray,
    ) -> None:
        g = _l2(np.asarray(global_embedding, dtype=np.float64).reshape(1, -1))[0]
        t = _l2(np.asarray(temporal_sequence, dtype=np.float64))

        if g.shape != (256,):
            raise ValueError(f"global_embedding must be [256], got {g.shape}")
        if t.ndim != 2 or t.shape[1] != 256:
            raise ValueError(f"temporal_sequence must be [T,256], got {t.shape}")

        if intent_id not in self.intents:
            self.intents[intent_id] = IntentMemory(
                intent_id=intent_id,
                canonical_text=canonical_text,
            )
        existing = self.intents[intent_id]
        if existing.canonical_text != canonical_text:
            raise ValueError(
                f"canonical_text mismatch for intent_id={intent_id}: "
                f"{existing.canonical_text!r} vs {canonical_text!r}"
            )

        existing.examples.append(
            EnrollmentExample(
                global_embedding=g.astype(np.float32),
                temporal_sequence=t.astype(np.float32),
            )
        )

    def shot_count(self) -> int:
        counts = {len(v.examples) for v in self.intents.values()}
        if not counts:
            raise RuntimeError("No enrolled intents")
        if len(counts) != 1:
            raise RuntimeError(
                f"Current frozen H6 policy requires equal shot count per intent; got {sorted(counts)}"
            )
        shot = next(iter(counts))
        if shot not in (1, 2):
            raise RuntimeError(
                f"Frozen H6 policy supports 1-shot or 2-shot only; got {shot}"
            )
        return shot

    def save(self, path: Path) -> None:
        payload = {
            "schema": "papr_ssl.h6_02_user_memory.v1",
            "user_id": self.user_id,
            "intents": {
                key: {
                    "intent_id": value.intent_id,
                    "canonical_text": value.canonical_text,
                    "examples": [
                        {
                            "global_embedding": ex.global_embedding,
                            "temporal_sequence": ex.temporal_sequence,
                        }
                        for ex in value.examples
                    ],
                }
                for key, value in self.intents.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "UserMemory":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "papr_ssl.h6_02_user_memory.v1":
            raise RuntimeError("Unsupported user-memory schema")

        obj = cls(user_id=str(payload["user_id"]))
        for key, value in payload["intents"].items():
            for ex in value["examples"]:
                obj.enroll(
                    intent_id=str(value["intent_id"]),
                    canonical_text=str(value["canonical_text"]),
                    global_embedding=np.asarray(ex["global_embedding"]),
                    temporal_sequence=np.asarray(ex["temporal_sequence"]),
                )
        return obj


class CWUDecisionHead:
    def __init__(self, head_json: Path, policy_json: Path):
        h = json.loads(head_json.read_text(encoding="utf-8"))
        p = json.loads(policy_json.read_text(encoding="utf-8"))

        self.feature_names = list(h["feature_names"])
        self.mean = np.asarray(h["standardize_mean"], dtype=np.float64)
        self.std = np.asarray(h["standardize_std"], dtype=np.float64)
        self.weight = np.asarray(h["weight"], dtype=np.float64)
        self.bias = np.asarray(h["bias"], dtype=np.float64)

        if self.weight.shape != (3, len(self.feature_names)):
            raise RuntimeError(
                f"Unexpected C/W/U weight shape: {self.weight.shape}"
            )
        if p.get("generic_test") != "sealed_not_accessed":
            raise RuntimeError("H6 policy generic_test seal is not intact")

        self.thresholds = {}
        for shot in ("1", "2"):
            info = p["summary"][shot]["thresholds"]
            self.thresholds[int(shot)] = {
                "accept": float(info["accept"]["threshold"]),
                "reject": float(info["reject"]["threshold"]),
            }

    def scores(self, feature: np.ndarray) -> dict[str, float]:
        x = np.asarray(feature, dtype=np.float64)
        if x.shape != (len(self.feature_names),):
            raise ValueError(
                f"Expected feature shape {(len(self.feature_names),)}, got {x.shape}"
            )
        z = (x - self.mean) / self.std
        logits = self.weight @ z + self.bias
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        s = exp / exp.sum()
        return {name: float(s[i]) for i, name in enumerate(EVENT_ORDER)}

    def decide(self, feature: np.ndarray, shot: int) -> tuple[str, dict[str, float]]:
        if shot not in self.thresholds:
            raise ValueError(f"Unsupported shot={shot}")
        s = self.scores(feature)
        argmax = max(s, key=s.get)
        t = self.thresholds[shot]

        if argmax == "C" and s["C"] >= t["accept"]:
            return "ACCEPT", s
        if argmax == "U" and s["U"] >= t["reject"]:
            return "REJECT", s
        return "CONFIRM", s


class PersonalizedRuntime:
    """Frozen research runtime for enrolled phrase recognition."""

    def __init__(
        self,
        *,
        head_json: Path,
        policy_json: Path,
        fusion_lambda: float = 0.50,
        top_k: int = 3,
    ):
        self.head = CWUDecisionHead(head_json, policy_json)
        self.fusion_lambda = float(fusion_lambda)
        self.top_k = int(top_k)

    def _candidate_evidence(
        self,
        memory: UserMemory,
        query_global: np.ndarray,
        query_temporal: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        shot = memory.shot_count()
        qg = _l2(np.asarray(query_global, dtype=np.float64).reshape(1, -1))[0]
        qt = _l2(np.asarray(query_temporal, dtype=np.float64))
        if qg.shape != (256,):
            raise ValueError(f"query_global must be [256], got {qg.shape}")
        if qt.ndim != 2 or qt.shape[1] != 256:
            raise ValueError(f"query_temporal must be [T,256], got {qt.shape}")

        intent_ids = sorted(memory.intents)
        prototypes = []
        for intent_id in intent_ids:
            ex = memory.intents[intent_id].examples
            p = _l2(
                np.mean(
                    np.stack([e.global_embedding for e in ex], axis=0),
                    axis=0,
                    keepdims=True,
                )
            )[0]
            prototypes.append(p)
        prototypes = np.stack(prototypes, axis=0)

        global_scores_all = qg @ prototypes.T
        global_order = np.argsort(-global_scores_all)
        candidate_local = global_order[: min(self.top_k, len(global_order))]

        dtw_scores = []
        duration_log_ratio = []
        support_cos_mean = []
        support_cos_min = []

        for local_idx in candidate_local:
            intent_id = intent_ids[int(local_idx)]
            ex = memory.intents[intent_id].examples

            dtw_sims = []
            frame_counts = []
            cosines = []
            for e in ex:
                d = _dtw_distance(qt, e.temporal_sequence)
                dtw_sims.append(1.0 - 0.5 * d)
                frame_counts.append(len(e.temporal_sequence))
                cosines.append(float(qg @ e.global_embedding))

            dtw_scores.append(float(max(dtw_sims)))
            med_frames = float(np.median(frame_counts))
            duration_log_ratio.append(
                abs(math.log(max(len(qt), 1) / max(med_frames, 1.0)))
            )
            support_cos_mean.append(float(np.mean(cosines)))
            support_cos_min.append(float(np.min(cosines)))

        dtw_scores = np.asarray(dtw_scores, dtype=np.float64)
        candidate_global = global_scores_all[candidate_local]
        fused = (
            self.fusion_lambda * candidate_global
            + (1.0 - self.fusion_lambda) * dtw_scores
        )
        order = np.argsort(-fused)
        top1_pos = int(order[0])
        top2_pos = int(order[1]) if len(order) > 1 else top1_pos

        global_winner_pos = int(np.argmax(candidate_global))
        dtw_winner_pos = int(np.argmax(dtw_scores))

        feature = np.asarray(
            [
                float(fused[top1_pos]),
                float(fused[top1_pos] - fused[top2_pos]),
                float(candidate_global[top1_pos]),
                float(candidate_global[top1_pos] - candidate_global[top2_pos]),
                float(dtw_scores[top1_pos]),
                float(dtw_scores[top1_pos] - dtw_scores[top2_pos]),
                1.0 if global_winner_pos == dtw_winner_pos else 0.0,
                float(duration_log_ratio[top1_pos]),
                math.log(float(shot)),
                float(support_cos_mean[top1_pos]),
                float(support_cos_min[top1_pos]),
            ],
            dtype=np.float64,
        )

        candidates = []
        for rank, pos in enumerate(order, start=1):
            local_idx = int(candidate_local[int(pos)])
            intent_id = intent_ids[local_idx]
            im = memory.intents[intent_id]
            candidates.append(
                {
                    "rank": rank,
                    "intent_id": intent_id,
                    "canonical_text": im.canonical_text,
                    "global_score": float(candidate_global[int(pos)]),
                    "dtw_score": float(dtw_scores[int(pos)]),
                    "fused_score": float(fused[int(pos)]),
                }
            )

        return feature, candidates

    def predict_feature(
        self,
        *,
        memory: UserMemory,
        query_global: np.ndarray,
        query_temporal: np.ndarray,
    ) -> dict[str, Any]:
        shot = memory.shot_count()
        feature, candidates = self._candidate_evidence(
            memory,
            query_global,
            query_temporal,
        )
        status, scores = self.head.decide(feature, shot)

        top1 = candidates[0]
        if status == "REJECT":
            intent_id = None
            canonical_text = None
        else:
            intent_id = top1["intent_id"]
            canonical_text = top1["canonical_text"]

        return {
            "schema": "papr_ssl.h6_02_prediction.v1",
            "user_id": memory.user_id,
            "status": status,
            "intent_id": intent_id,
            "canonical_text": canonical_text,
            "shot": shot,
            "decision_scores": scores,
            "scores_are_calibrated_probabilities": False,
            "top_candidate": top1,
            "candidates": candidates,
        }
