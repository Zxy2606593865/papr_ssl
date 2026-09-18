"""Google Speech Commands v2 -> PAPR-SSL unified manifest adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import soundfile as sf

from papr_ssl.data.manifest_schema import AudioManifestRecord


@dataclass(frozen=True)
class GSCAdapter:
    root: Path
    include_background_noise: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        if not self.root.is_dir():
            raise FileNotFoundError(f"GSC root not found: {self.root}")

    @staticmethod
    def _speaker_id(path: Path) -> str:
        stem = path.stem
        marker = "_nohash_"
        if marker not in stem:
            raise ValueError(
                f"Unexpected GSC filename without {marker!r}: {path.name}"
            )
        return stem.split(marker, 1)[0]

    def _official_split_sets(self) -> tuple[set[str], set[str]]:
        validation_file = self.root / "validation_list.txt"
        testing_file = self.root / "testing_list.txt"
        if not validation_file.is_file() or not testing_file.is_file():
            raise FileNotFoundError(
                "GSC v2 root must contain validation_list.txt and testing_list.txt"
            )

        def read_list(path: Path) -> set[str]:
            with path.open("r", encoding="utf-8") as f:
                return {
                    line.strip().replace("\\", "/")
                    for line in f
                    if line.strip()
                }

        validation = read_list(validation_file)
        testing = read_list(testing_file)
        overlap = validation & testing
        if overlap:
            raise ValueError(
                f"GSC official validation/test lists overlap: {len(overlap)} items"
            )
        return validation, testing

    def iter_records(self) -> Iterator[AudioManifestRecord]:
        validation, testing = self._official_split_sets()

        wavs = sorted(self.root.rglob("*.wav"))
        for wav in wavs:
            rel = wav.relative_to(self.root).as_posix()

            if rel.startswith("_background_noise_/"):
                if not self.include_background_noise:
                    continue
                info = sf.info(str(wav))
                # Noise assets do not belong to a speech-class split. We use
                # train only as a storage partition while role='none' carries
                # the "not an utterance" semantics.
                yield AudioManifestRecord(
                    utt_id=f"gsc_v2:{rel}",
                    dataset="gsc_v2",
                    audio_relpath=rel,
                    speaker_id=None,
                    label=None,
                    transcript=None,
                    language="en",
                    split="train",
                    domain="standard",
                    role="none",
                    record_type="noise",
                    sample_rate_hz=int(info.samplerate),
                    num_channels=int(info.channels),
                    num_frames=int(info.frames),
                    duration_sec=float(info.frames) / float(info.samplerate),
                    source_meta={
                        "source_group": "_background_noise_",
                        "official_split": False,
                    },
                )
                continue

            if rel in validation:
                split = "dev"
                official_split_name = "validation"
            elif rel in testing:
                split = "test"
                official_split_name = "test"
            else:
                split = "train"
                official_split_name = "train"

            label = wav.parent.name
            speaker_id = self._speaker_id(wav)
            info = sf.info(str(wav))

            yield AudioManifestRecord(
                utt_id=f"gsc_v2:{rel}",
                dataset="gsc_v2",
                audio_relpath=rel,
                speaker_id=speaker_id,
                label=label,
                transcript=label,
                language="en",
                split=split,
                domain="standard",
                role="train",
                record_type="speech",
                sample_rate_hz=int(info.samplerate),
                num_channels=int(info.channels),
                num_frames=int(info.frames),
                duration_sec=float(info.frames) / float(info.samplerate),
                source_meta={
                    "official_split": True,
                    "official_split_name": official_split_name,
                },
            )
