"""Correctness-first length-aware adapter for PAPR-SSL SSL backbones.

P2-06A converts a right-padded waveform batch into exact-length singleton
forwards, then repads the resulting VALID feature prefixes.

Important:
- the underlying backbone's own frame_mask is authoritative;
- an exact-length singleton input may still produce invalid tail frames because
  some SSL frontends internally pad/align feature sequences (e.g. W2v-BERT 2.0);
- therefore we keep only the prefix marked valid by output.frame_mask before
  cross-sample feature repadding.

This is a correctness reference path, not the final throughput optimization.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from papr_ssl.models.teacher.backbones.base import SSLBackboneOutput


def _validate_prefix_mask(mask: torch.Tensor) -> None:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise TypeError("mask must be bool [B,T]")
    if mask.shape[1] > 1:
        false_to_true = (~mask[:, :-1]) & mask[:, 1:]
        if false_to_true.any():
            raise ValueError(
                "mask must be prefix-contiguous: valid positions first, "
                "invalid/padded positions second"
            )


def _resolve_lengths(
    waveforms: torch.Tensor,
    *,
    waveform_lengths: torch.Tensor | None,
    waveform_mask: torch.Tensor | None,
) -> torch.Tensor:
    if waveforms.ndim != 2:
        raise ValueError(
            f"waveforms must be float32 [B,T], got {tuple(waveforms.shape)}"
        )
    if waveforms.dtype != torch.float32:
        raise TypeError(f"waveforms must be float32, got {waveforms.dtype}")
    if not torch.isfinite(waveforms).all():
        raise ValueError("waveforms contain NaN or Inf")

    batch_size, max_len = waveforms.shape
    if batch_size <= 0 or max_len <= 0:
        raise ValueError("waveforms cannot contain an empty batch/time axis")

    from_lengths = None
    if waveform_lengths is not None:
        if waveform_lengths.ndim != 1:
            raise ValueError("waveform_lengths must be [B]")
        if waveform_lengths.shape[0] != batch_size:
            raise ValueError("waveform_lengths batch size mismatch")
        if waveform_lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("waveform_lengths must be an integer tensor")

        from_lengths = waveform_lengths.to(
            device=waveforms.device,
            dtype=torch.int64,
        )
        if (from_lengths <= 0).any() or (from_lengths > max_len).any():
            raise ValueError(
                f"waveform_lengths must be within [1,{max_len}]"
            )

    from_mask = None
    if waveform_mask is not None:
        if waveform_mask.shape != waveforms.shape:
            raise ValueError(
                "waveform_mask must have the same [B,T] shape as waveforms"
            )

        waveform_mask = waveform_mask.to(device=waveforms.device)
        _validate_prefix_mask(waveform_mask)
        from_mask = waveform_mask.sum(dim=1, dtype=torch.int64)

        if (from_mask <= 0).any():
            raise ValueError("Every sample must have at least one valid sample")

    if from_lengths is None and from_mask is None:
        return torch.full(
            (batch_size,),
            max_len,
            dtype=torch.int64,
            device=waveforms.device,
        )

    if from_lengths is None:
        return from_mask
    if from_mask is None:
        return from_lengths

    if not torch.equal(from_lengths, from_mask):
        raise ValueError(
            "waveform_lengths and waveform_mask describe different valid lengths"
        )
    return from_lengths


def _validate_singleton_output(output: Any, sample_index: int) -> int:
    """Validate one backbone output and return its valid feature-frame count."""
    for name in ("features", "frame_mask", "hidden_layer"):
        if not hasattr(output, name):
            raise TypeError(
                f"Backbone output for sample {sample_index} misses {name!r}"
            )

    features = output.features
    mask = output.frame_mask

    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise TypeError("output.features must be Tensor [B,T,D]")
    if features.shape[0] != 1:
        raise ValueError(
            "LengthAwareSSLAdapter expects singleton backbone output B=1, "
            f"got {tuple(features.shape)}"
        )
    if features.dtype != torch.float32:
        raise TypeError(
            f"Backbone features must be float32, got {features.dtype}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("Backbone features contain NaN/Inf")

    if not isinstance(mask, torch.Tensor):
        raise TypeError("output.frame_mask must be Tensor")
    if mask.dtype != torch.bool:
        raise TypeError("output.frame_mask must be bool")
    if tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(
            "output.frame_mask must be [B,T] aligned with features"
        )

    _validate_prefix_mask(mask)

    valid_frames = int(mask[0].sum().item())
    if valid_frames <= 0:
        raise ValueError(
            f"Backbone produced zero valid feature frames for sample {sample_index}"
        )
    if valid_frames > features.shape[1]:
        raise ValueError("frame_mask valid count exceeds feature length")

    return valid_frames


class LengthAwareSSLAdapter(nn.Module):
    """Wrap an existing PAPR SSL backbone with explicit waveform validity.

    Reference behavior:

    1. crop each padded waveform to its semantic source length;
    2. run the existing backbone as a singleton;
    3. trust the backbone's own output.frame_mask;
    4. keep only the valid feature prefix;
    5. repad valid feature sequences across samples;
    6. emit a new strict frame_mask for the repadded batch.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor | None = None,
        waveform_mask: torch.Tensor | None = None,
    ) -> SSLBackboneOutput:
        lengths = _resolve_lengths(
            waveforms,
            waveform_lengths=waveform_lengths,
            waveform_mask=waveform_mask,
        )

        feature_rows: list[torch.Tensor] = []
        feature_lengths: list[int] = []

        hidden_layer: int | None = None
        feature_dim: int | None = None

        for i, length in enumerate(lengths.detach().cpu().tolist()):
            sample = waveforms[i : i + 1, : int(length)]

            output = self.backbone(sample)
            valid_frames = _validate_singleton_output(output, i)

            this_layer = int(output.hidden_layer)
            if hidden_layer is None:
                hidden_layer = this_layer
            elif this_layer != hidden_layer:
                raise ValueError(
                    "Underlying backbone returned inconsistent hidden_layer "
                    f"across samples: {hidden_layer} vs {this_layer}"
                )

            # The model-provided frame_mask is authoritative.
            # Prefix-contiguity was already validated, so [:valid_frames]
            # is equivalent to boolean selection while preserving time order.
            row = output.features[0, :valid_frames, :]

            this_dim = int(row.shape[-1])
            if feature_dim is None:
                feature_dim = this_dim
            elif this_dim != feature_dim:
                raise ValueError(
                    "Underlying backbone returned inconsistent feature dimension "
                    f"across samples: {feature_dim} vs {this_dim}"
                )

            feature_rows.append(row)
            feature_lengths.append(valid_frames)

        assert hidden_layer is not None

        features = pad_sequence(
            feature_rows,
            batch_first=True,
            padding_value=0.0,
        ).to(dtype=torch.float32)

        max_frames = int(features.shape[1])
        feature_lengths_t = torch.tensor(
            feature_lengths,
            dtype=torch.int64,
            device=features.device,
        )
        frame_index = torch.arange(
            max_frames,
            device=features.device,
        ).unsqueeze(0)
        frame_mask = frame_index < feature_lengths_t.unsqueeze(1)

        result = SSLBackboneOutput(
            features=features,
            frame_mask=frame_mask,
            hidden_layer=hidden_layer,
        )

        validate = getattr(result, "validate", None)
        if callable(validate):
            validate()

        return result

    def forward_audio_batch(self, batch: Any) -> SSLBackboneOutput:
        """Convenience bridge for the P2-05B AudioBatch contract."""
        for name in ("waveforms", "lengths", "waveform_mask"):
            if not hasattr(batch, name):
                raise TypeError(f"Audio batch misses required field {name!r}")

        return self.forward(
            batch.waveforms,
            waveform_lengths=batch.lengths,
            waveform_mask=batch.waveform_mask,
        )
