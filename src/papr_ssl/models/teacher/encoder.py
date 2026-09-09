"""Minimal composition of a Teacher backbone and representation head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from papr_ssl.contracts.teacher import TeacherOutput, WINDOW_SAMPLES


class TeacherEncoder(nn.Module):
    """Compose frame extraction and DR without coupling inference to SCAF."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, waveforms: Tensor) -> TeacherOutput:
        self._validate_waveforms(waveforms)
        backbone_output = self.backbone(waveforms)
        features = getattr(backbone_output, "features", None)
        frame_mask = getattr(backbone_output, "frame_mask", None)
        if not isinstance(features, Tensor) or not isinstance(frame_mask, Tensor):
            raise TypeError(
                "backbone output must expose Tensor features and frame_mask"
            )

        embedding = self.head(features, frame_mask)
        if embedding.dtype != torch.float32:
            embedding = embedding.to(dtype=torch.float32)
        frame_count = frame_mask.sum(dim=1).to(dtype=torch.int64)
        return TeacherOutput(embedding=embedding, frame_count=frame_count)

    @torch.no_grad()
    def encode(self, waveforms: Tensor) -> TeacherOutput:
        """Inference entry point; call `eval()` before deterministic encoding."""

        return self.forward(waveforms)

    @staticmethod
    def _validate_waveforms(waveforms: Tensor) -> None:
        if not isinstance(waveforms, Tensor):
            raise TypeError("waveforms must be a torch.Tensor")
        if waveforms.ndim != 2 or waveforms.shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"waveforms must have shape [B,{WINDOW_SAMPLES}]")
        if waveforms.shape[0] == 0:
            raise ValueError("waveforms batch must be non-empty")
        if waveforms.dtype != torch.float32:
            raise TypeError("waveforms must use torch.float32")
        if not bool(torch.isfinite(waveforms).all()):
            raise ValueError("waveforms contain NaN or Inf")
        if float(waveforms.abs().max()) > 1.0:
            raise ValueError("waveform samples must lie in [-1, 1]")
