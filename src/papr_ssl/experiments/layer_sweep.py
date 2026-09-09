"""Expand layer-sweep configuration without loading models or training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import yaml

from papr_ssl.models.teacher.backbones.factory import SUPPORTED_BACKBONES


@dataclass(frozen=True)
class LayerSweepCandidate:
    backbone_kind: str
    base_config: str
    hidden_layer: int


def load_layer_sweep(path: Path | str) -> Tuple[LayerSweepCandidate, ...]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"layer sweep config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return expand_layer_sweep(payload)


def expand_layer_sweep(payload: Mapping[str, Any]) -> Tuple[LayerSweepCandidate, ...]:
    """Expand and validate unique `(backbone_kind, hidden_layer)` candidates."""

    if not isinstance(payload, Mapping):
        raise TypeError("layer sweep payload must be a mapping")
    entries = payload.get("backbones")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("backbones must be a sequence")

    candidates = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("each backbone entry must be a mapping")
        kind = str(entry.get("kind", "")).strip().lower()
        if kind not in SUPPORTED_BACKBONES:
            raise ValueError(f"unknown backbone kind in layer sweep: {kind!r}")
        base_config = str(entry.get("base_config", "")).strip()
        if not base_config:
            raise ValueError("base_config must be non-empty")
        if Path(base_config).is_absolute():
            raise ValueError("base_config must be project-relative")
        hidden_layers = entry.get("hidden_layers")
        if not isinstance(hidden_layers, Sequence) or isinstance(
            hidden_layers, (str, bytes)
        ):
            raise ValueError("hidden_layers must be a sequence")
        for hidden_layer in hidden_layers:
            if not isinstance(hidden_layer, int) or isinstance(hidden_layer, bool):
                raise TypeError("hidden_layer candidates must be integers")
            if hidden_layer < 0:
                raise ValueError("hidden_layer candidates must be non-negative")
            key = (kind, hidden_layer)
            if key in seen:
                raise ValueError(f"duplicate layer sweep candidate: {key}")
            seen.add(key)
            candidates.append(
                LayerSweepCandidate(
                    backbone_kind=kind,
                    base_config=base_config,
                    hidden_layer=hidden_layer,
                )
            )
    if not candidates:
        raise ValueError("layer sweep must contain at least one candidate")
    return tuple(candidates)
