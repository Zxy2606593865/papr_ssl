"""JSONL I/O helpers for PAPR-SSL audio manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from papr_ssl.data.manifest_schema import AudioManifestRecord, record_from_dict


def write_manifest_jsonl(
    records: Iterable[AudioManifestRecord],
    path: Path,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            payload = record.to_dict()
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_manifest_jsonl(path: Path) -> list[AudioManifestRecord]:
    path = Path(path)
    records: list[AudioManifestRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records.append(record_from_dict(payload))
            except Exception as exc:
                raise ValueError(
                    f"Invalid manifest row at {path}:{line_no}: {exc}"
                ) from exc
    return records
