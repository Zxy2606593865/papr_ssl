"""Continuous personalized phrase spotting for the frozen PAPR H6 runtime.

This is an engineering extension, not a replacement for H6 and not open-vocabulary
ASR. It turns a continuous session into a timeline of *registered* expressions.

Design:
1) detect short acoustic boundary proposals;
2) run the existing RawWavFeatureAdapter + PersonalizedRuntime on candidates;
3) optionally merge adjacent atomic spans when a phrase was over-segmented;
4) resolve overlapping accepted candidates as an ordered non-overlapping sequence.

C/W/U decision scores remain uncalibrated. Continuous-mode use needs its own DEV
calibration before any scientific claims.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TimeSpan:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class SpotDetection:
    start_sec: float
    end_sec: float
    status: str
    intent_id: str | None
    canonical_text: str | None
    ranking_score: float
    cwu_c_score: float
    fused_score: float
    fused_margin: float
    merged_atomic_spans: int
    diagnostic: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["scores_are_calibrated_probabilities"] = False
        return out


@dataclass
class SpotterConfig:
    boundary_mode: str = "auto"  # auto | zero | energy
    min_boundary_sec: float = 0.12
    min_phrase_sec: float = 0.25
    pad_sec: float = 0.02
    max_merge_spans: int = 3
    max_merge_gap_sec: float = 0.65

    # Experimental microphone/real-audio fallback.
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    energy_margin_db: float = 10.0
    energy_peak_floor_db: float = 36.0
    fill_gap_ms: float = 50.0
    min_speech_ms: float = 100.0

    # Sequence decoder.
    allow_confirm: bool = False
    overlap_tolerance_sec: float = 0.08


class BoundaryDetector:
    def __init__(self, config: SpotterConfig | None = None):
        self.cfg = config or SpotterConfig()

    @staticmethod
    def _zero_runs(x: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
        zero = np.asarray(x) == 0
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, flag in enumerate(zero):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= min_samples:
                    runs.append((start, i))
                start = None
        if start is not None and len(x) - start >= min_samples:
            runs.append((start, len(x)))
        return runs

    @staticmethod
    def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, flag in enumerate(mask):
            match = bool(flag) == value
            if match and start is None:
                start = i
            elif not match and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    def _zero_boundary_spans(self, waveform: np.ndarray, sr: int) -> list[TimeSpan]:
        x = np.asarray(waveform)
        min_samples = max(1, round(self.cfg.min_boundary_sec * sr))
        runs = self._zero_runs(x, min_samples)
        if not runs:
            return [TimeSpan(0.0, len(x) / sr)] if len(x) else []

        spans: list[TimeSpan] = []
        cursor = 0
        for a, b in runs:
            if a > cursor:
                spans.append(TimeSpan(cursor / sr, a / sr))
            cursor = b
        if cursor < len(x):
            spans.append(TimeSpan(cursor / sr, len(x) / sr))
        return [s for s in spans if s.duration_sec >= self.cfg.min_phrase_sec]

    def _energy_speech_spans(self, waveform: np.ndarray, sr: int) -> list[TimeSpan]:
        x = np.asarray(waveform, dtype=np.float32)
        frame = max(1, round(self.cfg.frame_ms * sr / 1000))
        hop = max(1, round(self.cfg.hop_ms * sr / 1000))
        if len(x) < frame:
            return [TimeSpan(0.0, len(x) / sr)] if len(x) else []

        rms = []
        for s in range(0, len(x) - frame + 1, hop):
            v = x[s : s + frame]
            rms.append(float(np.sqrt(np.mean(v * v) + 1e-12)))
        rms = np.asarray(rms, dtype=np.float64)
        db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        noise = float(np.percentile(db, 20))
        peak = float(np.percentile(db, 95))
        threshold = max(
            noise + self.cfg.energy_margin_db,
            peak - self.cfg.energy_peak_floor_db,
        )
        speech = db > threshold

        # Fill micro holes so plosives / tiny pauses do not explode one phrase
        # into many islands.
        fill_frames = max(1, round(self.cfg.fill_gap_ms / self.cfg.hop_ms))
        for a, b in self._runs(speech, False):
            if a > 0 and b < len(speech) and (b - a) <= fill_frames:
                speech[a:b] = True

        min_frames = max(1, round(self.cfg.min_speech_ms / self.cfg.hop_ms))
        spans: list[TimeSpan] = []
        for a, b in self._runs(speech, True):
            if b - a < min_frames:
                continue
            start_sec = (a * hop) / sr
            end_sec = min(len(x) / sr, ((b - 1) * hop + frame) / sr)
            spans.append(TimeSpan(start_sec, end_sec))
        return spans

    def detect(self, waveform: np.ndarray, sr: int) -> tuple[str, list[TimeSpan]]:
        mode = self.cfg.boundary_mode
        if mode not in {"auto", "zero", "energy"}:
            raise ValueError(f"Unsupported boundary_mode={mode}")

        if mode in {"auto", "zero"}:
            zero_spans = self._zero_boundary_spans(waveform, sr)
            # Fixed demo contains explicit short zero gaps. Auto only commits to
            # this mode if at least two plausible phrase spans are found.
            if mode == "zero" or len(zero_spans) >= 2:
                return "zero", zero_spans

        return "energy", self._energy_speech_spans(waveform, sr)


class ContinuousPhraseSpotter:
    """Continuous registered-phrase spotter on top of frozen H6 components.

    Parameters are the already validated runtime objects:
      adapter: RawWavFeatureAdapter
      runtime: PersonalizedRuntime
      memory: UserMemory
    """

    def __init__(
        self,
        *,
        adapter: Any,
        runtime: Any,
        memory: Any,
        config: SpotterConfig | None = None,
    ):
        self.adapter = adapter
        self.runtime = runtime
        self.memory = memory
        self.cfg = config or SpotterConfig()
        self.boundary = BoundaryDetector(self.cfg)

    @staticmethod
    def _ranking(pred: dict[str, Any]) -> tuple[float, float, float, float]:
        c_score = float(pred.get("decision_scores", {}).get("C", 0.0))
        candidates = pred.get("candidates") or []
        fused = float(candidates[0].get("fused_score", 0.0)) if candidates else 0.0
        second = (
            float(candidates[1].get("fused_score", fused))
            if len(candidates) > 1
            else fused
        )
        margin = fused - second
        # Engineering ranking scalar only. NOT a probability.
        rank = c_score + 0.20 * fused + 0.10 * margin
        return float(rank), float(c_score), float(fused), float(margin)

    def _predict_span(
        self,
        waveform: np.ndarray,
        sr: int,
        span: TimeSpan,
        merged_atomic_spans: int,
    ) -> SpotDetection:
        a = max(0, int(round((span.start_sec - self.cfg.pad_sec) * sr)))
        b = min(len(waveform), int(round((span.end_sec + self.cfg.pad_sec) * sr)))
        if b <= a:
            raise ValueError("Empty candidate span")

        feat = self.adapter.extract_waveform(
            np.asarray(waveform[a:b], dtype=np.float32),
            sr,
        )
        pred = self.runtime.predict_feature(
            memory=self.memory,
            query_global=feat["global_embedding"],
            query_temporal=feat["temporal_sequence"],
        )
        rank, c_score, fused, margin = self._ranking(pred)
        top = pred.get("top_candidate") or {}

        return SpotDetection(
            start_sec=a / sr,
            end_sec=b / sr,
            status=str(pred["status"]),
            intent_id=pred.get("intent_id"),
            canonical_text=pred.get("canonical_text"),
            ranking_score=rank,
            cwu_c_score=c_score,
            fused_score=fused,
            fused_margin=margin,
            merged_atomic_spans=int(merged_atomic_spans),
            diagnostic={
                "top_candidate": top,
                "decision_scores": pred.get("decision_scores"),
                "scores_are_calibrated_probabilities": False,
            },
        )

    def _candidate_spans(
        self,
        atoms: list[TimeSpan],
    ) -> Iterable[tuple[TimeSpan, int]]:
        """Generate single-atom spans plus short adjacent merges.

        This is the practical substitute for brute-force raw-waveform sliding
        windows, which would invoke WavLM-large hundreds of times.
        """
        n = len(atoms)
        for i in range(n):
            yield TimeSpan(atoms[i].start_sec, atoms[i].end_sec), 1
            for count in range(2, self.cfg.max_merge_spans + 1):
                j = i + count - 1
                if j >= n:
                    break
                gap = atoms[j].start_sec - atoms[j - 1].end_sec
                if gap > self.cfg.max_merge_gap_sec:
                    break
                yield TimeSpan(atoms[i].start_sec, atoms[j].end_sec), count

    def _decode_non_overlapping(
        self,
        detections: list[SpotDetection],
    ) -> list[SpotDetection]:
        """Weighted interval scheduling for an ordered phrase sequence."""
        if not detections:
            return []

        dets = sorted(detections, key=lambda d: (d.end_sec, d.start_sec))
        predecessor: list[int] = []
        for i, d in enumerate(dets):
            j = i - 1
            while (
                j >= 0
                and dets[j].end_sec
                > d.start_sec + self.cfg.overlap_tolerance_sec
            ):
                j -= 1
            predecessor.append(j)

        dp = [0.0] * (len(dets) + 1)
        for i, d in enumerate(dets, start=1):
            include = max(0.0, d.ranking_score) + dp[predecessor[i - 1] + 1]
            exclude = dp[i - 1]
            dp[i] = max(include, exclude)

        out: list[SpotDetection] = []
        i = len(dets) - 1
        while i >= 0:
            include = max(0.0, dets[i].ranking_score) + dp[predecessor[i] + 1]
            exclude = dp[i]
            if include > exclude:
                out.append(dets[i])
                i = predecessor[i]
            else:
                i -= 1

        return sorted(out, key=lambda d: d.start_sec)

    def spot_waveform(self, waveform: np.ndarray, sr: int) -> dict[str, Any]:
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("Expected non-empty mono waveform")

        boundary_mode, atoms = self.boundary.detect(x, sr)
        accepted: list[SpotDetection] = []

        for span, merged_count in self._candidate_spans(atoms):
            detection = self._predict_span(x, sr, span, merged_count)
            if detection.status == "ACCEPT":
                accepted.append(detection)
            elif self.cfg.allow_confirm and detection.status == "CONFIRM":
                accepted.append(detection)

        selected = self._decode_non_overlapping(accepted)

        return {
            "schema": "papr_ssl.continuous_phrase_spotting.v1",
            "mode": "registered_phrase_spotting",
            "boundary_mode_used": boundary_mode,
            "duration_sec": round(len(x) / sr, 4),
            "atomic_span_count": len(atoms),
            "candidate_accept_count": len(accepted),
            "detections": [d.to_dict() for d in selected],
            "continuous_mode_calibration": (
                "engineering_demo_not_scientifically_recalibrated"
            ),
            "scores_are_calibrated_probabilities": False,
            "open_vocabulary_asr": False,
        }

    def spot_wav(self, path: str, load_fn: Any) -> dict[str, Any]:
        waveform, sr = load_fn(path)
        return self.spot_waveform(waveform, sr)
