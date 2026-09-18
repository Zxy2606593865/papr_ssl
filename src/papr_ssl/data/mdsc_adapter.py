"""MDSC / AISHELL-6B -> PAPR-SSL unified manifest adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import soundfile as sf

from papr_ssl.data.manifest_schema import AudioManifestRecord


def _read_utf8_sig_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_label_line(line: str, path: Path) -> tuple[str, str]:
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Malformed MDSC label row in {path}: {line!r}")

    utt_key, transcript = parts
    utt_key = Path(utt_key).stem
    transcript = transcript.strip()

    if not utt_key or not transcript:
        raise ValueError(f"Malformed MDSC label row in {path}: {line!r}")

    return utt_key, transcript


@dataclass(frozen=True)
class MDSCAdapter:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        if not self.root.is_dir():
            raise FileNotFoundError(f"MDSC root not found: {self.root}")

    @staticmethod
    def _speaker_from_utt_key(utt_key: str) -> str:
        parts = utt_key.split("_", 1)
        if len(parts) != 2 or not parts[0]:
            raise ValueError(f"Unexpected MDSC utterance key: {utt_key!r}")
        return parts[0]

    def _wav_for_label(
        self,
        label_path: Path,
        utt_key: str,
    ) -> Path:
        """Resolve one transcript key to its WAV path.

        MDSC exposes two transcript layouts:

        1) Centralized label file:
             Control/train/transcript/label.txt
             Uncontrol/train/transcript/label.txt

           while WAVs remain speaker-grouped:
             .../wav/<speaker_id>/<utt_key>.wav

        2) Per-speaker label file:
             Uncontrol/dev/enrollment/transcript/DF0014/label.txt

           with matching WAV subtree:
             Uncontrol/dev/enrollment/wav/DF0014/<utt_key>.wav

        This resolver supports both without recursive per-utterance searches.
        """
        rel_parts = label_path.relative_to(self.root).parts
        low = [x.lower() for x in rel_parts]

        if "transcript" not in low:
            raise ValueError(f"Unexpected MDSC transcript path: {label_path}")

        transcript_idx = low.index("transcript")
        prefix = self.root.joinpath(*rel_parts[:transcript_idx])
        transcript_suffix = rel_parts[transcript_idx + 1 : -1]

        wav_root = prefix / "wav"
        speaker_id = self._speaker_from_utt_key(utt_key)

        candidates: list[Path] = []

        # Per-speaker transcript tree:
        # transcript/DF0014/label.txt -> wav/DF0014/<utt>.wav
        if transcript_suffix:
            candidates.append(
                wav_root.joinpath(*transcript_suffix) / f"{utt_key}.wav"
            )

        # Centralized transcript tree:
        # transcript/label.txt -> wav/<speaker>/<utt>.wav
        candidates.append(wav_root / speaker_id / f"{utt_key}.wav")

        # Defensive fallback for any flat WAV layout.
        candidates.append(wav_root / f"{utt_key}.wav")

        for wav in candidates:
            if wav.is_file():
                return wav

        attempted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No WAV found for label key {utt_key!r}; attempted: {attempted}"
        )

    def _semantics_from_path(self, wav: Path) -> tuple[str, str, str]:
        parts = wav.relative_to(self.root).parts
        low = [x.lower() for x in parts]

        if not parts:
            raise ValueError(f"Invalid MDSC WAV path: {wav}")

        if low[0] == "control":
            domain = "control"
        elif low[0] == "uncontrol":
            domain = "dysarthria"
        else:
            raise ValueError(f"Unexpected MDSC top-level group: {parts[0]!r}")

        split = next(
            (x for x in ("train", "dev", "test") if x in low),
            None,
        )
        if split is None:
            raise ValueError(f"Cannot infer MDSC split from {wav}")

        if "enrollment" in low:
            role = "enrollment"
        elif "eval" in low:
            role = "eval"
        elif split == "train":
            role = "train"
        else:
            # Control dev/test has no explicit enrollment/eval subtree.
            role = "eval"

        return domain, split, role

    def iter_records(self) -> Iterator[AudioManifestRecord]:
        label_files = sorted(self.root.rglob("label.txt"))
        if not label_files:
            raise FileNotFoundError(f"No label.txt files found under {self.root}")

        seen_utt_ids: set[str] = set()
        seen_wavs: set[Path] = set()

        for label_path in label_files:
            for line in _read_utf8_sig_lines(label_path):
                utt_key, transcript = _parse_label_line(line, label_path)
                wav = self._wav_for_label(label_path, utt_key)
                rel = wav.relative_to(self.root).as_posix()

                domain, split, role = self._semantics_from_path(wav)
                speaker_id = self._speaker_from_utt_key(utt_key)

                # Cross-check that the speaker directory agrees with utt_key
                # when the WAV is speaker-grouped.
                wav_parts = wav.relative_to(self.root).parts
                if len(wav_parts) >= 2:
                    parent_name = wav.parent.name
                    if parent_name.upper().startswith(("CF", "CM", "DF", "DM")):
                        if parent_name != speaker_id:
                            raise ValueError(
                                "MDSC speaker mismatch: "
                                f"utt_key={utt_key!r}, wav_parent={parent_name!r}"
                            )

                info = sf.info(str(wav))

                utt_id = f"mdsc:{rel}"
                if utt_id in seen_utt_ids:
                    raise ValueError(f"Duplicate MDSC utt_id: {utt_id}")

                seen_utt_ids.add(utt_id)
                seen_wavs.add(wav.resolve())

                # In the raw public-data manifest, source label and transcript
                # intentionally remain identical. Canonical intent mapping is
                # a later task-layer concern.
                yield AudioManifestRecord(
                    utt_id=utt_id,
                    dataset="mdsc",
                    audio_relpath=rel,
                    speaker_id=speaker_id,
                    label=transcript,
                    transcript=transcript,
                    language="zh-CN",
                    split=split,
                    domain=domain,
                    role=role,
                    record_type="speech",
                    sample_rate_hz=int(info.samplerate),
                    num_channels=int(info.channels),
                    num_frames=int(info.frames),
                    duration_sec=float(info.frames) / float(info.samplerate),
                    source_meta={
                        "source_group": (
                            "Control" if domain == "control" else "Uncontrol"
                        ),
                        "label_file": label_path.relative_to(self.root).as_posix(),
                        "source_utt_key": utt_key,
                    },
                )

        all_wavs = {p.resolve() for p in self.root.rglob("*.wav")}
        missing_from_transcripts = all_wavs - seen_wavs

        if missing_from_transcripts:
            examples = sorted(str(p) for p in missing_from_transcripts)[:10]
            raise ValueError(
                "MDSC WAV files missing from parsed transcripts: "
                f"{len(missing_from_transcripts)}; examples={examples}"
            )
