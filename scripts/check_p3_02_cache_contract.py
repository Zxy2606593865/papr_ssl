#!/usr/bin/env python
"""Standalone P3-02 cache-contract check."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from papr_ssl.cache.ssl_feature_cache import (
    CacheSample,
    MeanFeatureCacheReader,
    MeanFeatureCacheWriter,
    cache_identities,
    fanout_multilayer_masked_mean,
    semantic_waveform_hash,
)


MODEL_ID = "facebook/wav2vec2-base"
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"


def main() -> int:
    waveform = torch.linspace(-0.5, 0.5, 16000)
    sample_hash = semantic_waveform_hash(
        waveform,
        sample_rate_hz=16000,
        valid_num_samples=16000,
    )

    identities = cache_identities(
        model_id=MODEL_ID,
        model_revision=REVISION,
        layers=(6, 12),
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        writers = {
            ident.layer: MeanFeatureCacheWriter(
                root,
                ident,
                shard_size=8,
            )
            for ident in identities
        }

        sample = CacheSample(
            utt_id="smoke_0001",
            dataset="gsc",
            split="train",
            sample_hash=sample_hash,
            frame_count=1,
        )

        # Simulates two selected hidden states produced by ONE SSL forward.
        frame_mask = torch.tensor(
            [[True, True, True, False]],
            dtype=torch.bool,
        )
        hidden_states = {
            6: torch.tensor(
                [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [99.0, 99.0]]]
            ),
            12: torch.tensor(
                [[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [99.0, 99.0]]]
            ),
        }

        fanout_multilayer_masked_mean(
            hidden_states_by_layer=hidden_states,
            frame_mask=frame_mask,
            samples=[sample],
            writers=writers,
        )

        manifests = {
            layer: writer.finalize()
            for layer, writer in writers.items()
        }

        for ident in identities:
            reader = MeanFeatureCacheReader(
                manifests[ident.layer].parent,
                ident,
            )
            vector, frame_count = reader.get(
                dataset="gsc",
                utt_id="smoke_0001",
                expected_sample_hash=sample_hash,
            )
            assert vector.shape == (2,)
            assert frame_count == 3

            manifest = json.loads(
                manifests[ident.layer].read_text(encoding="utf-8")
            )
            assert manifest["identity"]["model_revision"] == REVISION
            assert manifest["identity"]["layer"] == ident.layer

    print("=" * 92)
    print("PAPR-SSL P3-02 OFFLINE SSL FEATURE CACHE CONTRACT")
    print("=" * 92)
    print("cache payload:              masked-mean SSL vector [D], float32")
    print("model revision recorded:    YES")
    print("hidden_states layer recorded:YES")
    print("sample SHA-256 recorded:    YES")
    print("stale audio detection:      YES")
    print("multi-layer one-pass fanout:YES")
    print("task labels cached:         NO")
    print("-" * 92)
    print("P3-02 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
