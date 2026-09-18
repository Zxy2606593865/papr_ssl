"""P3-03 frozen Teacher training configuration contract.

This module defines the minimal machine-readable contract for a P3 Teacher run:
frozen SSL backbone -> masked-mean cache/Mean DR -> 64D unit embedding -> SCAF.

The config deliberately contains only training semantics that must be frozen
before P3-04/P3-05. Dataset/sampler details are added in P3-04; run outputs and
hashes are recorded later in P3-08.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


KNOWN_BACKBONES: dict[str, dict[str, Any]] = {
    "wav2vec2_base": {
        "model_id": "facebook/wav2vec2-base",
        "model_revision": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
        "hidden_state_count": 13,
        "default_layer": 12,
        "hidden_dim": 768,
    },
    "wavlm_large": {
        "model_id": "microsoft/wavlm-large",
        "model_revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
        "hidden_state_count": 25,
        "default_layer": 24,
        "hidden_dim": 1024,
    },
    "w2v_bert2": {
        "model_id": "facebook/w2v-bert-2.0",
        "model_revision": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        "hidden_state_count": 25,
        "default_layer": 24,
        "hidden_dim": 1024,
    },
}


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    model_id: str
    model_revision: str
    layer: int
    frozen: bool
    force_eval: bool
    use_offline_cache: bool

    def validate(self) -> None:
        if self.name not in KNOWN_BACKBONES:
            raise ValueError(f"unknown backbone name: {self.name}")

        expected = KNOWN_BACKBONES[self.name]
        if self.model_id != expected["model_id"]:
            raise ValueError(
                f"{self.name}: model_id mismatch: "
                f"{self.model_id!r} != {expected['model_id']!r}"
            )
        if self.model_revision != expected["model_revision"]:
            raise ValueError(
                f"{self.name}: model_revision must be the frozen P1 commit SHA"
            )

        max_index = int(expected["hidden_state_count"]) - 1
        if not 0 <= self.layer <= max_index:
            raise ValueError(
                f"{self.name}: layer must be outputs.hidden_states index "
                f"in [0,{max_index}], got {self.layer}"
            )
        if self.frozen is not True:
            raise ValueError("P3 requires backbone.frozen=true")
        if self.force_eval is not True:
            raise ValueError("P3 requires backbone.force_eval=true")
        if self.use_offline_cache is not True:
            raise ValueError(
                "P3 baseline requires backbone.use_offline_cache=true"
            )


@dataclass(frozen=True)
class HeadConfig:
    kind: str
    embedding_dim: int
    l2_normalize: bool

    def validate(self) -> None:
        if self.kind != "mean_dr":
            raise ValueError("P3-03 freezes head.kind='mean_dr'")
        if self.embedding_dim != 64:
            raise ValueError("P3-03 freezes embedding_dim=64")
        if self.l2_normalize is not True:
            raise ValueError("P3 requires 64D unit-norm embeddings")


@dataclass(frozen=True)
class SCAFConfig:
    k: int
    margin: float
    scale: float

    def validate(self) -> None:
        if self.k != 3:
            raise ValueError("P3-03 freezes SCAF K=3")
        if abs(self.margin - 0.2) > 1e-12:
            raise ValueError("P3-03 freezes SCAF margin=0.2 rad")
        if abs(self.scale - 30.0) > 1e-12:
            raise ValueError("P3-03 freezes SCAF scale=30")


@dataclass(frozen=True)
class TeacherTrainingConfig:
    schema: str
    phase: str
    seed: int
    backbone: BackboneConfig
    head: HeadConfig
    scaf: SCAFConfig

    def validate(self) -> None:
        if self.schema != "papr_ssl.teacher_training_config.v1":
            raise ValueError(f"unsupported config schema: {self.schema}")
        if self.phase != "P3":
            raise ValueError("Teacher training config phase must be P3")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.backbone.validate()
        self.head.validate()
        self.scaf.validate()


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required config field: {key}")
    return mapping[key]


def teacher_training_config_from_dict(
    data: Mapping[str, Any],
) -> TeacherTrainingConfig:
    backbone = _required(data, "backbone")
    head = _required(data, "head")
    scaf = _required(data, "scaf")

    cfg = TeacherTrainingConfig(
        schema=str(_required(data, "schema")),
        phase=str(_required(data, "phase")),
        seed=int(_required(data, "seed")),
        backbone=BackboneConfig(
            name=str(_required(backbone, "name")),
            model_id=str(_required(backbone, "model_id")),
            model_revision=str(_required(backbone, "model_revision")),
            layer=int(_required(backbone, "layer")),
            frozen=bool(_required(backbone, "frozen")),
            force_eval=bool(_required(backbone, "force_eval")),
            use_offline_cache=bool(
                _required(backbone, "use_offline_cache")
            ),
        ),
        head=HeadConfig(
            kind=str(_required(head, "kind")),
            embedding_dim=int(_required(head, "embedding_dim")),
            l2_normalize=bool(_required(head, "l2_normalize")),
        ),
        scaf=SCAFConfig(
            k=int(_required(scaf, "k")),
            margin=float(_required(scaf, "margin")),
            scale=float(_required(scaf, "scale")),
        ),
    )
    cfg.validate()
    return cfg


def load_teacher_training_config(path: str | Path) -> TeacherTrainingConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("teacher config root must be a mapping")
    return teacher_training_config_from_dict(data)


def config_signature(cfg: TeacherTrainingConfig) -> tuple[Any, ...]:
    """Stable semantic signature excluding backbone identity/layer/seed.

    Used to prove all P3 backbone configs share the same Mean-DR/SCAF contract.
    """
    return (
        cfg.head.kind,
        cfg.head.embedding_dim,
        cfg.head.l2_normalize,
        cfg.scaf.k,
        cfg.scaf.margin,
        cfg.scaf.scale,
        cfg.backbone.frozen,
        cfg.backbone.force_eval,
        cfg.backbone.use_offline_cache,
    )
