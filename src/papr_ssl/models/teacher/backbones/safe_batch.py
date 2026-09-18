"""Per-backbone safe batching policy for PAPR-SSL.

Measured P2-06B policy:
- WavLM: native right-padded batching is numerically aligned with the exact-
  length singleton reference.
- W2v-BERT 2.0: native batching is aligned when raw rows are trimmed before the
  model-specific feature extractor.
- Wav2Vec2-base: right-padding changes valid-frame feature values substantially
  for shorter rows, despite structurally correct frame masks.

Therefore Wav2Vec2 uses exact-length grouping:
  1) group batch rows by semantic waveform length;
  2) slice every group to that exact raw length;
  3) run one backbone forward per length group;
  4) repad feature sequences and restore original row order.

This preserves correctness while still batching samples that share a length.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from papr_ssl.models.teacher.backbones.base import SSLBackboneOutput


_NATIVE_KINDS = {"wavlm", "w2v_bert2"}
_GROUPED_EXACT_LENGTH_KINDS = {"wav2vec2"}


def _normalize_kind(kind: str) -> str:
    return str(kind).strip().lower().replace("-", "_")


def _validate_audio_batch(batch: Any) -> None:
    for name in ("waveforms", "waveform_mask", "lengths"):
        if not hasattr(batch, name):
            raise TypeError(f"Audio batch misses required field {name!r}")

    waveforms = batch.waveforms
    mask = batch.waveform_mask
    lengths = batch.lengths

    if not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 2:
        raise ValueError("batch.waveforms must be Tensor [B,T]")
    if waveforms.dtype != torch.float32:
        raise TypeError("batch.waveforms must be float32")

    if not isinstance(mask, torch.Tensor) or mask.shape != waveforms.shape:
        raise ValueError("batch.waveform_mask must have shape [B,T]")
    if mask.dtype != torch.bool:
        raise TypeError("batch.waveform_mask must be bool")

    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise ValueError("batch.lengths must be Tensor [B]")
    if lengths.shape[0] != waveforms.shape[0]:
        raise ValueError("batch.lengths batch size mismatch")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch.lengths must be integer")

    derived = mask.sum(dim=1, dtype=torch.int64)
    if not torch.equal(
        derived.to(device=lengths.device),
        lengths.to(dtype=torch.int64),
    ):
        raise ValueError("batch.lengths and waveform_mask disagree")


def _grouped_exact_length_forward(
    backbone,
    batch,
) -> SSLBackboneOutput:
    """Exact-length grouped forward, primarily for Wav2Vec2-base."""
    _validate_audio_batch(batch)

    batch_size = int(batch.waveforms.shape[0])
    lengths_cpu = [int(v) for v in batch.lengths.detach().cpu().tolist()]

    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths_cpu):
        if length <= 0:
            raise ValueError("all semantic waveform lengths must be positive")
        groups[length].append(index)

    feature_rows: list[torch.Tensor | None] = [None] * batch_size
    hidden_layer: int | None = None
    feature_dim: int | None = None

    for length in sorted(groups):
        indices = groups[length]
        index_t = torch.tensor(
            indices,
            dtype=torch.long,
            device=batch.waveforms.device,
        )
        group_waveforms = batch.waveforms.index_select(0, index_t)[:, :length]

        # No sample_mask is required: every row in this group ends exactly at
        # its semantic source length.
        output = backbone(group_waveforms)

        if output.features.shape[0] != len(indices):
            raise RuntimeError("backbone changed grouped batch size unexpectedly")
        if output.frame_mask.shape != output.features.shape[:2]:
            raise RuntimeError("backbone frame_mask/features shape mismatch")

        this_layer = int(output.hidden_layer)
        if hidden_layer is None:
            hidden_layer = this_layer
        elif hidden_layer != this_layer:
            raise RuntimeError("backbone hidden_layer changed across length groups")

        for local_i, original_i in enumerate(indices):
            valid_frames = int(output.frame_mask[local_i].sum().item())
            if valid_frames <= 0:
                raise RuntimeError("backbone produced zero valid feature frames")

            row = output.features[local_i, :valid_frames].to(torch.float32)
            this_dim = int(row.shape[-1])
            if feature_dim is None:
                feature_dim = this_dim
            elif feature_dim != this_dim:
                raise RuntimeError("feature dimension changed across length groups")

            feature_rows[original_i] = row

    if hidden_layer is None or any(row is None for row in feature_rows):
        raise RuntimeError("failed to produce all grouped feature rows")

    rows = [row for row in feature_rows if row is not None]
    features = pad_sequence(
        rows,
        batch_first=True,
        padding_value=0.0,
    ).to(dtype=torch.float32)

    frame_lengths = torch.tensor(
        [int(row.shape[0]) for row in rows],
        dtype=torch.int64,
        device=features.device,
    )
    t = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
    frame_mask = t < frame_lengths.unsqueeze(1)

    return SSLBackboneOutput(
        features=features,
        frame_mask=frame_mask,
        hidden_layer=hidden_layer,
    )


def safe_forward_audio_batch(
    *,
    backbone_kind: str,
    backbone,
    batch,
) -> SSLBackboneOutput:
    """Run one AudioBatch using the measured-safe policy for this backbone."""
    _validate_audio_batch(batch)
    kind = _normalize_kind(backbone_kind)

    if kind in _NATIVE_KINDS:
        return backbone(
            batch.waveforms,
            sample_mask=batch.waveform_mask,
        )

    if kind in _GROUPED_EXACT_LENGTH_KINDS:
        return _grouped_exact_length_forward(backbone, batch)

    raise ValueError(
        f"No frozen safe batching policy for backbone_kind={backbone_kind!r}"
    )


def batching_policy_name(backbone_kind: str) -> str:
    kind = _normalize_kind(backbone_kind)
    if kind in _NATIVE_KINDS:
        return "native_padded_batch"
    if kind in _GROUPED_EXACT_LENGTH_KINDS:
        return "grouped_exact_waveform_length"
    raise ValueError(f"Unknown backbone kind: {backbone_kind!r}")
