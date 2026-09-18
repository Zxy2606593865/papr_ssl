"""P2-07D descriptive phrase inventory audit for MDSC."""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.task_views import (
    canonical_mdsc_wake_word,
    normalize_mdsc_task_text,
)

@dataclass(frozen=True)
class PhraseInventoryRow:
    phrase: str
    utterance_count: int
    speaker_count: int
    control_utterance_count: int
    dysarthria_utterance_count: int
    control_speaker_count: int
    dysarthria_speaker_count: int
    train_utterance_count: int
    train_speaker_count: int
    control_train_speaker_count: int
    dysarthria_train_speaker_count: int
    dev_utterance_count: int
    test_utterance_count: int
    enrollment_count: int
    eval_count: int
    raw_variant_count: int
    raw_variants: tuple[tuple[str, int], ...]
    is_canonical_wake_word: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "utterance_count": self.utterance_count,
            "speaker_count": self.speaker_count,
            "control_utterance_count": self.control_utterance_count,
            "dysarthria_utterance_count": self.dysarthria_utterance_count,
            "control_speaker_count": self.control_speaker_count,
            "dysarthria_speaker_count": self.dysarthria_speaker_count,
            "train_utterance_count": self.train_utterance_count,
            "train_speaker_count": self.train_speaker_count,
            "control_train_speaker_count": self.control_train_speaker_count,
            "dysarthria_train_speaker_count": self.dysarthria_train_speaker_count,
            "dev_utterance_count": self.dev_utterance_count,
            "test_utterance_count": self.test_utterance_count,
            "enrollment_count": self.enrollment_count,
            "eval_count": self.eval_count,
            "raw_variant_count": self.raw_variant_count,
            "raw_variants": [
                {"text": text, "count": count}
                for text, count in self.raw_variants
            ],
            "is_canonical_wake_word": self.is_canonical_wake_word,
        }

def build_mdsc_phrase_inventory(
    records: Iterable[AudioManifestRecord],
) -> list[PhraseInventoryRow]:
    utterance_counts = Counter()
    speakers = defaultdict(set)
    domain_utterance = defaultdict(Counter)
    domain_speakers = defaultdict(set)
    split_utterance = defaultdict(Counter)
    role_utterance = defaultdict(Counter)
    train_speakers = defaultdict(set)
    train_domain_speakers = defaultdict(set)
    raw_variants = defaultdict(Counter)

    for r in records:
        if r.dataset != "mdsc" or r.record_type != "speech":
            continue
        raw = r.transcript if r.transcript is not None else r.label
        if raw is None:
            raise ValueError(f"Missing transcript/label: {r.utt_id}")
        phrase = normalize_mdsc_task_text(raw)
        if not phrase:
            raise ValueError(f"Empty normalized phrase: {r.utt_id}")
        if r.speaker_id is None:
            raise ValueError(f"Missing speaker_id: {r.utt_id}")

        utterance_counts[phrase] += 1
        speakers[phrase].add(r.speaker_id)
        domain_utterance[phrase][r.domain] += 1
        domain_speakers[(phrase, r.domain)].add(r.speaker_id)
        split_utterance[phrase][r.split] += 1
        role_utterance[phrase][r.role] += 1
        raw_variants[phrase][raw] += 1

        if r.split == "train":
            train_speakers[phrase].add(r.speaker_id)
            train_domain_speakers[(phrase, r.domain)].add(r.speaker_id)

    out = []
    for phrase in utterance_counts:
        variants = tuple(sorted(
            raw_variants[phrase].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ))
        out.append(PhraseInventoryRow(
            phrase=phrase,
            utterance_count=utterance_counts[phrase],
            speaker_count=len(speakers[phrase]),
            control_utterance_count=domain_utterance[phrase]["control"],
            dysarthria_utterance_count=domain_utterance[phrase]["dysarthria"],
            control_speaker_count=len(domain_speakers[(phrase, "control")]),
            dysarthria_speaker_count=len(domain_speakers[(phrase, "dysarthria")]),
            train_utterance_count=split_utterance[phrase]["train"],
            train_speaker_count=len(train_speakers[phrase]),
            control_train_speaker_count=len(
                train_domain_speakers[(phrase, "control")]
            ),
            dysarthria_train_speaker_count=len(
                train_domain_speakers[(phrase, "dysarthria")]
            ),
            dev_utterance_count=split_utterance[phrase]["dev"],
            test_utterance_count=split_utterance[phrase]["test"],
            enrollment_count=role_utterance[phrase]["enrollment"],
            eval_count=role_utterance[phrase]["eval"],
            raw_variant_count=len(variants),
            raw_variants=variants,
            is_canonical_wake_word=canonical_mdsc_wake_word(phrase) is not None,
        ))

    out.sort(key=lambda x: (
        -x.train_speaker_count,
        -min(x.control_train_speaker_count, x.dysarthria_train_speaker_count),
        -x.train_utterance_count,
        x.phrase,
    ))
    return out

DEFAULT_THRESHOLD_SWEEP = (
    (1, 1), (3, 3), (5, 3), (5, 5), (8, 5),
    (10, 5), (10, 8), (15, 8), (20, 10),
)

def summarize_phrase_inventory(
    rows: list[PhraseInventoryRow],
    thresholds=DEFAULT_THRESHOLD_SWEEP,
) -> dict[str, Any]:
    sweep = []
    for min_control, min_dys in thresholds:
        selected = [
            r for r in rows
            if (
                r.control_train_speaker_count >= min_control
                and r.dysarthria_train_speaker_count >= min_dys
            )
        ]
        sweep.append({
            "min_control_train_speakers": min_control,
            "min_dysarthria_train_speakers": min_dys,
            "phrase_count": len(selected),
            "non_wake_phrase_count": sum(
                not r.is_canonical_wake_word for r in selected
            ),
            "train_utterance_count": sum(
                r.train_utterance_count for r in selected
            ),
        })

    hist = Counter(r.train_speaker_count for r in rows)
    return {
        "schema": "papr_ssl.mdsc_phrase_inventory.v1",
        "phase": "P2-07D",
        "policy": {
            "raw_manifest_rewrite": False,
            "task_normalization_only": True,
            "semantic_synonym_merging": False,
            "candidate_support_metrics_use_train_only": True,
            "dev_test_used_only_for_descriptive_counts": True,
        },
        "inventory": {
            "phrase_count": len(rows),
            "cross_domain_phrase_count": sum(
                r.control_utterance_count > 0
                and r.dysarthria_utterance_count > 0
                for r in rows
            ),
            "wake_phrase_count": sum(r.is_canonical_wake_word for r in rows),
            "non_wake_phrase_count": sum(
                not r.is_canonical_wake_word for r in rows
            ),
        },
        "train_speaker_coverage_histogram": {
            str(k): v for k, v in sorted(hist.items())
        },
        "threshold_sweep": sweep,
        "top_by_train_support": [r.to_dict() for r in rows[:100]],
    }
