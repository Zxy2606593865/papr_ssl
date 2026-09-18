"""Helpers for real P3 experiments.

P2 task-view rows use a globally unique utterance id. For current MDSC
artifacts, that id can be dataset-qualified and path-like, e.g.:

    mdsc:Control/dev/wav/CF0010/CF0010_0001.wav

That identifier is already sufficient to resolve the immutable physical WAV.
Legacy/plain utterance ids are supported through a stem catalog fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ALLOWED_REAL_SPLITS = {"train", "dev"}


@dataclass(frozen=True)
class RealTaskRow:
    dataset: str
    utt_id: str
    split: str
    task_label: str
    audio_relpath: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.utt_id:
            raise ValueError("utt_id must be non-empty")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"unsupported split: {self.split!r}")
        if not self.task_label:
            raise ValueError("task_label must be non-empty")


def _first_present(
    row: Mapping[str, Any],
    keys: Sequence[str],
) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(
        "none of the required alternative fields are present: "
        + ", ".join(keys)
    )


def _optional_first_present(
    row: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def parse_real_task_row(raw: Mapping[str, Any]) -> RealTaskRow:
    return RealTaskRow(
        dataset=str(raw.get("dataset") or "mdsc").strip(),
        utt_id=_first_present(
            raw,
            ("utt_id", "utterance_id", "id"),
        ),
        split=str(raw["split"]).strip().lower(),
        task_label=_first_present(
            raw,
            ("task_label", "label", "phrase_id", "transcript"),
        ),
        audio_relpath=_optional_first_present(
            raw,
            ("audio_relpath", "path", "audio_path", "relpath"),
        ),
    )


def load_real_task_rows(
    path: str | Path,
    *,
    splits: Sequence[str] = ("train", "dev"),
) -> list[RealTaskRow]:
    path = Path(path)
    wanted = {str(x).strip().lower() for x in splits}
    if not wanted:
        raise ValueError("splits must be non-empty")
    if not wanted <= ALLOWED_REAL_SPLITS:
        raise ValueError(
            "real P3 execution may load train/dev only; "
            f"requested={sorted(wanted)}"
        )

    rows: list[RealTaskRow] = []
    seen: set[tuple[str, str]] = set()

    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, Mapping):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        row = parse_real_task_row(obj)

        # The source task index can contain test rows, but P3 development never
        # materializes or evaluates test audio.
        if row.split not in wanted:
            continue

        key = (row.dataset, row.utt_id)
        if key in seen:
            raise ValueError(f"duplicate task key: {key}")
        seen.add(key)
        rows.append(row)

    if not rows:
        raise ValueError(
            f"no rows found for splits={sorted(wanted)} in {path}"
        )
    return rows


def summarize_real_rows(
    rows: Sequence[RealTaskRow],
) -> dict[str, Any]:
    return {
        "count": len(rows),
        "split_counts": dict(Counter(row.split for row in rows)),
        "class_count": len({row.task_label for row in rows}),
        "dataset_counts": dict(Counter(row.dataset for row in rows)),
        "rows_with_explicit_path": sum(
            row.audio_relpath is not None for row in rows
        ),
        "rows_with_qualified_utt_id": sum(
            qualified_utt_relpath(row) is not None for row in rows
        ),
    }


def build_mdsc_audio_catalog(
    mdsc_root: str | Path,
) -> dict[str, Path]:
    """Legacy/plain-id fallback: map unique WAV stem -> physical path."""
    root = Path(mdsc_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MDSC root does not exist: {root}")

    catalog: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in root.rglob("*.wav"):
        resolved = path.resolve()
        key = path.stem
        previous = catalog.get(key)
        if previous is None:
            catalog[key] = resolved
        else:
            duplicates.setdefault(key, [previous]).append(resolved)

    if duplicates:
        preview = []
        for utt_id, paths in list(sorted(duplicates.items()))[:5]:
            preview.append(
                f"{utt_id}: " + " | ".join(str(x) for x in paths[:3])
            )
        raise RuntimeError(
            "MDSC WAV catalog has duplicate filename stems; "
            "legacy plain-id fallback is ambiguous:\n"
            + "\n".join(preview)
        )

    if not catalog:
        raise RuntimeError(f"no WAV files found under MDSC root: {root}")
    return catalog


def resolve_audio_path(
    *,
    mdsc_root: str | Path,
    audio_relpath: str,
) -> Path:
    """Resolve a physical path safely inside the declared MDSC root."""
    root = Path(mdsc_root).expanduser().resolve()
    raw = Path(audio_relpath).expanduser()

    if raw.is_absolute():
        candidates = [raw.resolve()]
    else:
        candidates = [(root / raw).resolve()]
        cwd_candidate = (Path.cwd() / raw).resolve()
        if cwd_candidate not in candidates:
            candidates.append(cwd_candidate)

    safe_candidates: list[Path] = []
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        safe_candidates.append(candidate)

    if not safe_candidates:
        raise ValueError(
            f"audio path escapes declared MDSC root: {audio_relpath!r}"
        )

    for candidate in safe_candidates:
        if candidate.exists():
            return candidate
    return safe_candidates[0]


def qualified_utt_relpath(row: RealTaskRow) -> str | None:
    """Decode `dataset:relative/path.wav` utterance ids.

    Returns None for legacy/plain ids.
    """
    text = row.utt_id.strip()
    if ":" not in text:
        return None

    prefix, rel = text.split(":", 1)
    if prefix.strip().casefold() != row.dataset.strip().casefold():
        return None
    rel = rel.strip().replace("\\", "/")
    if not rel:
        raise ValueError(f"qualified utt_id has empty path: {row.utt_id!r}")

    # Treat it as POSIX-style logical relative path independent of host OS.
    logical = PurePosixPath(rel)
    if logical.is_absolute() or ".." in logical.parts:
        raise ValueError(
            f"unsafe qualified utt_id path: {row.utt_id!r}"
        )
    if logical.suffix.lower() != ".wav":
        raise ValueError(
            f"qualified MDSC utt_id must point to .wav: {row.utt_id!r}"
        )
    return logical.as_posix()


def resolve_task_audio_path(
    *,
    row: RealTaskRow,
    mdsc_root: str | Path,
    audio_catalog: Mapping[str, Path] | None = None,
) -> Path:
    """Resolve one task row to a physical WAV with integrity checks.

    Resolution order:
    1. explicit audio path field, if present;
    2. current canonical dataset-qualified utt_id;
    3. legacy/plain-id stem catalog fallback.
    """
    resolution_kind: str

    if row.audio_relpath is not None:
        path = resolve_audio_path(
            mdsc_root=mdsc_root,
            audio_relpath=row.audio_relpath,
        )
        resolution_kind = "explicit_path"
    else:
        encoded_rel = qualified_utt_relpath(row)
        if encoded_rel is not None:
            path = resolve_audio_path(
                mdsc_root=mdsc_root,
                audio_relpath=encoded_rel,
            )
            resolution_kind = "qualified_utt_id"
        else:
            if audio_catalog is None:
                raise ValueError(
                    "legacy/plain utt_id requires an audio catalog"
                )
            lookup_keys = [row.utt_id, Path(row.utt_id).stem]
            path = None
            for key in lookup_keys:
                candidate = audio_catalog.get(key)
                if candidate is not None:
                    path = Path(candidate).resolve()
                    break
            if path is None:
                raise FileNotFoundError(
                    "task utt_id not found in MDSC WAV catalog: "
                    f"{row.utt_id}"
                )
            resolution_kind = "legacy_catalog"

    if not path.is_file():
        raise FileNotFoundError(path)

    # Prevent accidental train/dev cross-join.
    parts_lower = {part.lower() for part in path.parts}
    if row.split not in parts_lower:
        raise RuntimeError(
            "task/audio split mismatch: "
            f"utt_id={row.utt_id}, task_split={row.split}, path={path}"
        )

    if resolution_kind == "qualified_utt_id":
        encoded_rel = qualified_utt_relpath(row)
        assert encoded_rel is not None
        actual_rel = (
            path.resolve()
            .relative_to(Path(mdsc_root).expanduser().resolve())
            .as_posix()
        )
        if actual_rel != encoded_rel:
            raise RuntimeError(
                "qualified utt_id/path mismatch: "
                f"{encoded_rel!r} != {actual_rel!r}"
            )
    elif resolution_kind == "legacy_catalog":
        if path.stem != Path(row.utt_id).stem:
            raise RuntimeError(
                "legacy utt_id/path stem mismatch: "
                f"{row.utt_id!r} != {path.stem!r}"
            )

    return path
