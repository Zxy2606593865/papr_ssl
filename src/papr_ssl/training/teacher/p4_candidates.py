"""P4-01: frozen Backbone / hidden-state sweep candidate registry.

Important semantic rule:
    `hidden_state_index` means the exact index into Hugging Face
    `outputs.hidden_states`.

It is NOT a paper layer number and is never inferred from names such as
"Layer 16".  The candidate ranges below come only from P1 real-checkpoint
validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


P4_CANDIDATE_SCHEMA = "papr_ssl.p4_layer_sweep_candidates.v1"


@dataclass(frozen=True)
class P4BackboneCandidateSet:
    name: str
    model_id: str
    model_revision: str
    hidden_state_count: int
    valid_hidden_state_indices: tuple[int, ...]
    embedding_dim: int

    def validate(self) -> None:
        expected = tuple(range(self.hidden_state_count))
        if self.valid_hidden_state_indices != expected:
            raise ValueError(
                f"{self.name}: candidates must exactly cover P1-validated "
                f"hidden_states indices {expected}, got "
                f"{self.valid_hidden_state_indices}"
            )
        if self.hidden_state_count <= 0:
            raise ValueError("hidden_state_count must be > 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if len(self.model_revision) != 40:
            raise ValueError(
                f"{self.name}: model_revision must be exact 40-char commit SHA"
            )

    @property
    def final_hidden_state_index(self) -> int:
        return self.hidden_state_count - 1


P4_BACKBONES: tuple[P4BackboneCandidateSet, ...] = (
    P4BackboneCandidateSet(
        name="wav2vec2_base",
        model_id="facebook/wav2vec2-base",
        model_revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
        hidden_state_count=13,
        valid_hidden_state_indices=tuple(range(13)),
        embedding_dim=768,
    ),
    P4BackboneCandidateSet(
        name="wavlm_large",
        model_id="microsoft/wavlm-large",
        model_revision="c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
        hidden_state_count=25,
        valid_hidden_state_indices=tuple(range(25)),
        embedding_dim=1024,
    ),
    P4BackboneCandidateSet(
        name="w2v_bert2",
        model_id="facebook/w2v-bert-2.0",
        model_revision="da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        hidden_state_count=25,
        valid_hidden_state_indices=tuple(range(25)),
        embedding_dim=1024,
    ),
)


def validate_p4_candidates() -> None:
    names: set[str] = set()
    model_ids: set[str] = set()

    for item in P4_BACKBONES:
        item.validate()
        if item.name in names:
            raise ValueError(f"duplicate backbone name: {item.name}")
        if item.model_id in model_ids:
            raise ValueError(f"duplicate model_id: {item.model_id}")
        names.add(item.name)
        model_ids.add(item.model_id)


def get_p4_backbone(name: str) -> P4BackboneCandidateSet:
    validate_p4_candidates()
    for item in P4_BACKBONES:
        if item.name == name:
            return item
    raise KeyError(f"unknown P4 backbone: {name!r}")


def total_candidate_count() -> int:
    validate_p4_candidates()
    return sum(len(x.valid_hidden_state_indices) for x in P4_BACKBONES)


def as_machine_readable_dict() -> dict:
    validate_p4_candidates()
    return {
        "schema": P4_CANDIDATE_SCHEMA,
        "phase": "P4-01",
        "index_semantics": (
            "exact outputs.hidden_states tuple index; "
            "not a paper layer number"
        ),
        "candidate_policy": "all P1-validated hidden-state indices",
        "selection_metric": "generic_dev_score",
        "selection_metric_definition": "prototype_macro_f1",
        "generic_test": "sealed_not_accessed",
        "seeds": [17, 29, 43],
        "backbones": [
            {
                "name": x.name,
                "model_id": x.model_id,
                "model_revision": x.model_revision,
                "hidden_state_count": x.hidden_state_count,
                "valid_hidden_state_indices": list(
                    x.valid_hidden_state_indices
                ),
                "final_hidden_state_index": x.final_hidden_state_index,
                "embedding_dim": x.embedding_dim,
            }
            for x in P4_BACKBONES
        ],
        "total_candidate_count": total_candidate_count(),
    }
