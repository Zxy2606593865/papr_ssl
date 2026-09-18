"""Raw WAV -> frozen H6 feature adapter.

Pipeline
--------
WAV
-> mono float32 / 16 kHz
-> pinned microsoft/wavlm-large
-> hidden_states[15]
-> frozen Attention DR -> global 256D
-> same frozen 1024->256 frame projection
-> L2 -> temporal downsample x3 -> L2
-> H6-02 PersonalizedRuntime

This module does not train or update any parameter.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

from papr_ssl.training.teacher.p5_dr import build_dr


WAVLM_MODEL_ID = "microsoft/wavlm-large"
WAVLM_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
HIDDEN_STATE_INDEX = 15
TARGET_SAMPLE_RATE = 16000
TEMPORAL_DOWNSAMPLE = 3


def _replace_projection(model: nn.Module, dim: int = 256) -> nn.Module:
    model = copy.deepcopy(model)
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == 64
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one 64-output Linear in Attention DR, got "
            f"{[(n, m.in_features, m.out_features) for n, m in candidates]}"
        )
    name, old = candidates[0]
    model.set_submodule(
        name,
        nn.Linear(old.in_features, dim, bias=(old.bias is not None)),
    )
    return model


def _load_teacher_dr(checkpoint: Path, device: torch.device) -> nn.Module:
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(obj.get("embedding_dim", -1)) != 256:
        raise RuntimeError(
            f"Expected 256D checkpoint, got embedding_dim={obj.get('embedding_dim')}"
        )

    dr = _replace_projection(build_dr("attention"), 256)
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)

    for p in dr.parameters():
        p.requires_grad_(False)

    return dr


def _get_frame_projection(dr: nn.Module) -> tuple[str, nn.Linear]:
    candidates = [
        (name, module)
        for name, module in dr.named_modules()
        if isinstance(module, nn.Linear)
        and module.in_features == 1024
        and module.out_features == 256
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one 1024->256 projection, got "
            f"{[(n, m.in_features, m.out_features) for n, m in candidates]}"
        )
    return candidates[0]


def _temporal_project(
    hidden: torch.Tensor,
    projection: nn.Linear,
    downsample: int = TEMPORAL_DOWNSAMPLE,
) -> torch.Tensor:
    z = projection(hidden.to(torch.float32))
    z = F.normalize(z, dim=-1)

    chunks = []
    for start in range(0, z.shape[0], downsample):
        pooled = z[start:start + downsample].mean(dim=0, keepdim=True)
        chunks.append(F.normalize(pooled, dim=-1))

    return torch.cat(chunks, dim=0).contiguous()


def resolve_selected_checkpoint(embedding_manifest: Path) -> Path:
    manifest = json.loads(embedding_manifest.read_text(encoding="utf-8"))
    if manifest.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("Teacher embedding manifest generic_test seal is not intact")
    selected = manifest.get("selected_checkpoint")
    if not isinstance(selected, dict) or "checkpoint" not in selected:
        raise RuntimeError("selected_checkpoint missing from embedding manifest")
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_wav_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono/stereo waveform, got shape={audio.shape}")

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    if int(sr) != TARGET_SAMPLE_RATE:
        x = torch.from_numpy(audio)
        x = torchaudio.functional.resample(
            x,
            orig_freq=int(sr),
            new_freq=TARGET_SAMPLE_RATE,
        )
        audio = x.numpy().astype(np.float32, copy=False)
        sr = TARGET_SAMPLE_RATE

    return audio, int(sr)


class RawWavFeatureAdapter:
    """Frozen WAV -> (global256, temporal256) adapter."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: str | None = None,
        model_id: str = WAVLM_MODEL_ID,
        revision: str = WAVLM_REVISION,
        hidden_state_index: int = HIDDEN_STATE_INDEX,
        temporal_downsample: int = TEMPORAL_DOWNSAMPLE,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_id = model_id
        self.revision = revision
        self.hidden_state_index = int(hidden_state_index)
        self.temporal_downsample = int(temporal_downsample)

        # IMPORTANT: P5 frame cache was generated from the raw waveform tensor.
        # Do NOT use AutoFeatureExtractor here: its default zero-mean/unit-variance
        # normalization changes WavLM representations and breaks parity.
        self.wavlm = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self.wavlm.eval().to(self.device)

        for p in self.wavlm.parameters():
            p.requires_grad_(False)

        self.dr = _load_teacher_dr(Path(checkpoint), self.device)
        self.projection_name, self.frame_projection = _get_frame_projection(self.dr)

    @torch.inference_mode()
    def extract_waveform(
        self,
        waveform: np.ndarray,
        sampling_rate: int,
    ) -> dict[str, np.ndarray | int | str]:
        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError(f"Expected [N] waveform, got {waveform.shape}")

        if int(sampling_rate) != TARGET_SAMPLE_RATE:
            x = torch.from_numpy(waveform)
            x = torchaudio.functional.resample(
                x,
                orig_freq=int(sampling_rate),
                new_freq=TARGET_SAMPLE_RATE,
            )
            waveform = x.numpy().astype(np.float32, copy=False)
            sampling_rate = TARGET_SAMPLE_RATE

        # Exact P5-compatible input path:
        #   raw mono float32 waveform -> [1, N] -> WavLM
        # No HF waveform normalization and no attention mask for a single,
        # unpadded utterance.
        input_values = torch.from_numpy(
            np.ascontiguousarray(waveform, dtype=np.float32)
        )[None].to(self.device)

        outputs = self.wavlm(
            input_values=input_values,
            output_hidden_states=True,
            return_dict=True,
        )

        if outputs.hidden_states is None:
            raise RuntimeError("WavLM did not return hidden_states")
        if self.hidden_state_index >= len(outputs.hidden_states):
            raise RuntimeError(
                f"hidden_state_index={self.hidden_state_index}, "
                f"available={len(outputs.hidden_states)}"
            )

        hidden = outputs.hidden_states[self.hidden_state_index]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 1024:
            raise RuntimeError(f"Unexpected WavLM hidden shape: {tuple(hidden.shape)}")

        frame_mask = torch.ones(
            (1, hidden.shape[1]),
            dtype=torch.bool,
            device=self.device,
        )

        global256 = F.normalize(
            self.dr(hidden, frame_mask).float(),
            dim=-1,
        )[0]

        temporal256 = _temporal_project(
            hidden[0],
            self.frame_projection,
            downsample=self.temporal_downsample,
        )

        return {
            "global_embedding": global256.cpu().numpy().astype(np.float32),
            "temporal_sequence": temporal256.cpu().numpy().astype(np.float32),
            "hidden_frames": int(hidden.shape[1]),
            "temporal_frames": int(temporal256.shape[0]),
            "sampling_rate": TARGET_SAMPLE_RATE,
            "model_id": self.model_id,
            "revision": self.revision,
            "hidden_state_index": self.hidden_state_index,
            "projection_name": self.projection_name,
            "waveform_input_mode": "raw_float32_no_hf_normalization",
        }

    def extract_wav(self, path: str | Path) -> dict[str, np.ndarray | int | str]:
        path = Path(path)
        waveform, sr = load_wav_mono_16k(path)
        out = self.extract_waveform(waveform, sr)
        out["wav_path"] = str(path)
        return out
