from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "src/papr_ssl/inference/h6_continuous_phrase_spotter.py"
spec = importlib.util.spec_from_file_location("papr_continuous_test_module", MODULE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


class FakeAdapter:
    def extract_waveform(self, waveform, sr):
        dur = len(waveform) / sr
        g = np.zeros(256, np.float32)
        g[0] = dur
        t = np.zeros((max(1, round(dur * 10)), 256), np.float32)
        t[:, 0] = dur
        return {"global_embedding": g, "temporal_sequence": t}


class FakeMemory:
    pass


class FakeRuntime:
    def predict_feature(self, *, memory, query_global, query_temporal):
        dur = float(query_global[0])
        ok = 0.45 <= dur <= 1.2
        return {
            "status": "ACCEPT" if ok else "REJECT",
            "intent_id": "x" if ok else None,
            "canonical_text": "测试表达" if ok else None,
            "decision_scores": {
                "C": 0.9 if ok else 0.1,
                "W": 0.05,
                "U": 0.05 if ok else 0.85,
            },
            "top_candidate": {"fused_score": 0.8 if ok else 0.2},
            "candidates": [
                {"fused_score": 0.8 if ok else 0.2},
                {"fused_score": 0.3},
            ],
        }


class ContinuousSpotterTests(unittest.TestCase):
    def test_short_zero_boundaries_detected(self):
        sr = 16000
        a = np.ones(int(0.8 * sr), np.float32)
        z = np.zeros(int(0.18 * sr), np.float32)
        b = np.ones(int(0.7 * sr), np.float32)
        cfg = module.SpotterConfig(
            boundary_mode="zero",
            min_boundary_sec=0.12,
            pad_sec=0,
            max_merge_spans=1,
        )
        mode, spans = module.BoundaryDetector(cfg).detect(
            np.concatenate([a, z, b]),
            sr,
        )
        self.assertEqual(mode, "zero")
        self.assertEqual(len(spans), 2)

    def test_spotter_outputs_registered_timeline(self):
        sr = 16000
        a = np.ones(int(0.8 * sr), np.float32)
        z = np.zeros(int(0.18 * sr), np.float32)
        b = np.ones(int(0.7 * sr), np.float32)
        cfg = module.SpotterConfig(
            boundary_mode="zero",
            min_boundary_sec=0.12,
            pad_sec=0,
            max_merge_spans=1,
        )
        spotter = module.ContinuousPhraseSpotter(
            adapter=FakeAdapter(),
            runtime=FakeRuntime(),
            memory=FakeMemory(),
            config=cfg,
        )
        out = spotter.spot_waveform(np.concatenate([a, z, b]), sr)
        self.assertFalse(out["open_vocabulary_asr"])
        self.assertEqual(len(out["detections"]), 2)
        self.assertTrue(
            all(x["canonical_text"] == "测试表达" for x in out["detections"])
        )
        self.assertFalse(out["scores_are_calibrated_probabilities"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
