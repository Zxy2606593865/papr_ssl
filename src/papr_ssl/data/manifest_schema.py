"""Unified public-audio manifest contract for PAPR-SSL.

P2-03 freezes the interchange format between dataset-specific adapters and
downstream PAPR-SSL data/model code.

Important:
- paths are dataset-root-relative, not machine-specific absolute paths;
- source metadata is preserved without forcing all datasets into identical
  raw directory semantics;
- this module does not read audio or dataset metadata files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = "papr_ssl.audio_manifest.v1"

SUPPORTED_DATASETS = {"gsc_v2", "mdsc"}
SUPPORTED_LANGUAGES = {"en", "zh-CN"}
SUPPORTED_SPLITS = {"train", "dev", "test"}
SUPPORTED_DOMAINS = {"standard", "control", "dysarthria"}
SUPPORTED_ROLES = {"train", "enrollment", "eval", "none"}
SUPPORTED_RECORD_TYPES = {"speech", "noise"}


@dataclass(frozen=True)
class AudioManifestRecord:
    """One immutable audio-record entry in the PAPR-SSL public-data manifest."""

    # Stable identity / source
    utt_id: str
    dataset: str
    audio_relpath: str

    # Task/source semantics
    speaker_id: str | None
    label: str | None
    transcript: str | None
    language: str
    split: str
    domain: str
    role: str
    record_type: str = "speech"

    # Raw audio header facts
    sample_rate_hz: int = 16_000
    num_channels: int = 1
    num_frames: int = 0
    duration_sec: float = 0.0

    # Dataset-specific metadata that should not expand the common schema.
    source_meta: Mapping[str, Any] = field(default_factory=dict)

    # Explicit versioning for reproducibility.
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        """Raise ValueError if the record violates the P2-03 contract."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={self.schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION!r}"
            )

        if not self.utt_id.strip():
            raise ValueError("utt_id must be non-empty")

        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, "
                f"got {self.dataset!r}"
            )

        if not self.audio_relpath.strip():
            raise ValueError("audio_relpath must be non-empty")

        normalized_path = self.audio_relpath.replace("\\", "/")
        if normalized_path.startswith("/") or ":" in normalized_path.split("/")[0]:
            raise ValueError(
                "audio_relpath must be relative to the dataset root, not absolute"
            )
        if ".." in normalized_path.split("/"):
            raise ValueError("audio_relpath must not escape the dataset root")

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {sorted(SUPPORTED_LANGUAGES)}, "
                f"got {self.language!r}"
            )

        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(SUPPORTED_SPLITS)}, "
                f"got {self.split!r}"
            )

        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"domain must be one of {sorted(SUPPORTED_DOMAINS)}, "
                f"got {self.domain!r}"
            )

        if self.role not in SUPPORTED_ROLES:
            raise ValueError(
                f"role must be one of {sorted(SUPPORTED_ROLES)}, "
                f"got {self.role!r}"
            )

        if self.record_type not in SUPPORTED_RECORD_TYPES:
            raise ValueError(
                f"record_type must be one of {sorted(SUPPORTED_RECORD_TYPES)}, "
                f"got {self.record_type!r}"
            )

        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if self.duration_sec <= 0:
            raise ValueError("duration_sec must be > 0")

        if self.record_type == "speech":
            if self.speaker_id is None or not self.speaker_id.strip():
                raise ValueError("speech record requires speaker_id")
            if self.label is None and self.transcript is None:
                raise ValueError(
                    "speech record requires at least one of label/transcript"
                )

        if self.record_type == "noise":
            # GSC background-noise recordings are source assets, not speakers.
            if self.role != "none":
                raise ValueError("noise record must use role='none'")

        # Dataset-specific semantic guards.
        if self.dataset == "gsc_v2":
            if self.language != "en":
                raise ValueError("gsc_v2 language must be 'en'")
            if self.domain != "standard":
                raise ValueError("gsc_v2 domain must be 'standard'")
            if self.record_type == "speech" and self.role != "train":
                # role describes enrollment/eval semantics, not official split.
                # GSC has no enrollment/eval hierarchy.
                raise ValueError("gsc_v2 speech records must use role='train'")

        if self.dataset == "mdsc":
            if self.language != "zh-CN":
                raise ValueError("mdsc language must be 'zh-CN'")
            if self.domain not in {"control", "dysarthria"}:
                raise ValueError(
                    "mdsc domain must be 'control' or 'dysarthria'"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["audio_relpath"] = self.audio_relpath.replace("\\", "/")
        payload["source_meta"] = dict(self.source_meta)
        return payload


def record_from_dict(payload: Mapping[str, Any]) -> AudioManifestRecord:
    """Construct and validate one record from decoded JSON."""
    record = AudioManifestRecord(**dict(payload))
    record.validate()
    return record
