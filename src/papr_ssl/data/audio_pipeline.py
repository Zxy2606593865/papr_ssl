"""PAPR-SSL P2-05B audio loading and length-aware batching.

Policy
------
GSC v2
    - source must be 16 kHz
    - source must be mono
    - speech waveform is represented in a 1-second / 16000-sample container
    - shorter source speech is right-zero-padded
    - valid_num_samples remains the ORIGINAL source length

MDSC / AISHELL-6B
    - source must be 16 kHz
    - full utterance is preserved
    - stereo/multi-channel source is averaged to mono in memory
    - no crop, truncation, or source-file rewrite

Batching
    - examples are right-zero-padded to the longest stored waveform in batch
    - lengths stores semantic valid sample counts
    - waveform_mask is bool [B, T] and marks valid samples only
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

from papr_ssl.data.manifest_schema import AudioManifestRecord


TARGET_SAMPLE_RATE_HZ = 16_000
GSC_FIXED_NUM_SAMPLES = 16_000


@dataclass(frozen=True)
class AudioExample:
    """One loaded, mono, float32 waveform plus its manifest record."""

    record: AudioManifestRecord
    waveform: torch.Tensor  # float32 [T]
    valid_num_samples: int  # semantic length before in-memory right padding

    def validate(self) -> None:
        self.record.validate()

        if self.waveform.ndim != 1:
            raise ValueError(
                f"waveform must be rank-1 [T], got {tuple(self.waveform.shape)}"
            )
        if self.waveform.dtype != torch.float32:
            raise ValueError(
                f"waveform dtype must be float32, got {self.waveform.dtype}"
            )
        if self.waveform.numel() <= 0:
            raise ValueError("waveform must be non-empty")
        if self.valid_num_samples <= 0:
            raise ValueError("valid_num_samples must be > 0")
        if self.valid_num_samples > self.waveform.numel():
            raise ValueError(
                "valid_num_samples cannot exceed stored waveform length"
            )

        tail = self.waveform[self.valid_num_samples :]
        if tail.numel() and not torch.count_nonzero(tail).eq(0):
            raise ValueError(
                "samples after valid_num_samples must be zero padding"
            )


@dataclass(frozen=True)
class AudioBatch:
    """Length-aware waveform batch before SSL-specific frontend processing."""

    waveforms: torch.Tensor      # float32 [B, T]
    waveform_mask: torch.Tensor  # bool [B, T], True = valid source sample
    lengths: torch.Tensor        # int64 [B]
    records: tuple[AudioManifestRecord, ...]

    def validate(self) -> None:
        if self.waveforms.ndim != 2:
            raise ValueError(
                f"waveforms must be [B,T], got {tuple(self.waveforms.shape)}"
            )
        if self.waveform_mask.shape != self.waveforms.shape:
            raise ValueError(
                "waveform_mask must have the same [B,T] shape as waveforms"
            )
        if self.waveform_mask.dtype != torch.bool:
            raise ValueError("waveform_mask must be bool")
        if self.lengths.ndim != 1:
            raise ValueError("lengths must be rank-1 [B]")
        if self.lengths.dtype != torch.int64:
            raise ValueError("lengths must be int64")

        batch_size, max_len = self.waveforms.shape
        if self.lengths.shape[0] != batch_size:
            raise ValueError("lengths batch dimension mismatch")
        if len(self.records) != batch_size:
            raise ValueError("records batch dimension mismatch")
        if batch_size == 0:
            raise ValueError("empty batch is not allowed")
        if self.waveforms.dtype != torch.float32:
            raise ValueError("waveforms must be float32")

        for i, length in enumerate(self.lengths.tolist()):
            if length <= 0 or length > max_len:
                raise ValueError(
                    f"invalid length at row {i}: {length}, max_len={max_len}"
                )
            expected = torch.arange(
                max_len, device=self.waveform_mask.device
            ) < length
            if not torch.equal(self.waveform_mask[i], expected):
                raise ValueError(
                    f"waveform_mask does not match length at row {i}"
                )
            padded = self.waveforms[i, length:]
            if padded.numel() and not torch.count_nonzero(padded).eq(0):
                raise ValueError(
                    f"non-zero padded samples found at row {i}"
                )

    def to(self, device: torch.device | str) -> "AudioBatch":
        moved = AudioBatch(
            waveforms=self.waveforms.to(device),
            waveform_mask=self.waveform_mask.to(device),
            lengths=self.lengths.to(device),
            records=self.records,
        )
        moved.validate()
        return moved


def _resolve_audio_path(
    record: AudioManifestRecord,
    dataset_roots: Mapping[str, Path | str],
) -> Path:
    try:
        root = Path(dataset_roots[record.dataset]).resolve()
    except KeyError as exc:
        raise KeyError(
            f"Missing dataset root for {record.dataset!r}"
        ) from exc

    path = (root / record.audio_relpath).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Resolved audio path escapes dataset root: {path}"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_mono_float32(path: Path) -> tuple[np.ndarray, int, int]:
    """Return mono float32 waveform, sample rate, original channel count."""
    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    # soundfile shape: [frames, channels]
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(f"Invalid decoded audio shape for {path}: {audio.shape}")

    original_channels = int(audio.shape[1])

    if original_channels == 1:
        mono = audio[:, 0]
    else:
        # Policy: deterministic equal-weight mean across channels.
        mono = audio.mean(axis=1, dtype=np.float32)

    mono = np.asarray(mono, dtype=np.float32)
    return mono, int(sr), original_channels


def load_audio_example(
    record: AudioManifestRecord,
    dataset_roots: Mapping[str, Path | str],
) -> AudioExample:
    """Load one manifest record under the frozen P2-05B policy."""
    record.validate()
    if record.record_type != "speech":
        raise ValueError(
            "P2-05B speech loader does not consume record_type='noise'"
        )

    path = _resolve_audio_path(record, dataset_roots)
    mono, sr, source_channels = _read_mono_float32(path)

    if sr != TARGET_SAMPLE_RATE_HZ:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE_HZ} Hz, got {sr} Hz for {path}"
        )

    # Cross-check manifest raw facts against the real file.
    if sr != record.sample_rate_hz:
        raise ValueError(
            f"Manifest sample_rate mismatch for {record.utt_id}: "
            f"manifest={record.sample_rate_hz}, file={sr}"
        )
    if source_channels != record.num_channels:
        raise ValueError(
            f"Manifest channel mismatch for {record.utt_id}: "
            f"manifest={record.num_channels}, file={source_channels}"
        )
    if mono.shape[0] != record.num_frames:
        raise ValueError(
            f"Manifest frame-count mismatch for {record.utt_id}: "
            f"manifest={record.num_frames}, file={mono.shape[0]}"
        )

    valid_num_samples = int(mono.shape[0])

    if record.dataset == "gsc_v2":
        if valid_num_samples > GSC_FIXED_NUM_SAMPLES:
            raise ValueError(
                "GSC speech exceeds frozen 1-second contract: "
                f"{record.utt_id} has {valid_num_samples} samples"
            )
        if valid_num_samples < GSC_FIXED_NUM_SAMPLES:
            padded = np.zeros((GSC_FIXED_NUM_SAMPLES,), dtype=np.float32)
            padded[:valid_num_samples] = mono
            mono = padded
    elif record.dataset == "mdsc":
        # Preserve the complete utterance. No crop/pad at example level.
        pass
    else:
        raise ValueError(
            f"P2-05B has no audio policy for dataset={record.dataset!r}"
        )

    waveform = torch.from_numpy(np.ascontiguousarray(mono)).to(torch.float32)

    example = AudioExample(
        record=record,
        waveform=waveform,
        valid_num_samples=valid_num_samples,
    )
    example.validate()
    return example


def collate_audio_examples(
    examples: Sequence[AudioExample],
) -> AudioBatch:
    """Right-pad examples and emit explicit semantic lengths/mask."""
    if not examples:
        raise ValueError("Cannot collate an empty sequence")

    for example in examples:
        example.validate()

    max_len = max(int(x.waveform.numel()) for x in examples)
    batch_size = len(examples)

    waveforms = torch.zeros(
        (batch_size, max_len),
        dtype=torch.float32,
    )
    lengths = torch.empty((batch_size,), dtype=torch.int64)

    for i, example in enumerate(examples):
        stored_len = int(example.waveform.numel())
        waveforms[i, :stored_len] = example.waveform
        lengths[i] = int(example.valid_num_samples)

    time = torch.arange(max_len, dtype=torch.int64).unsqueeze(0)
    waveform_mask = time < lengths.unsqueeze(1)

    batch = AudioBatch(
        waveforms=waveforms,
        waveform_mask=waveform_mask,
        lengths=lengths,
        records=tuple(x.record for x in examples),
    )
    batch.validate()
    return batch
