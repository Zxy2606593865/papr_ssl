"""P2-07E revised MDSC task policy.

MDSC is treated as a multi-use dysarthria speech resource rather than a
wake-word-only dataset.

Frozen views:
1. Core30 exact-phrase benchmark:
   10 canonical wake phrases + 20 high-coverage commands.
2. Command20 exact-phrase benchmark:
   the 20 non-wake commands only.
3. Core30 open-set evaluation:
   dev/test phrases outside Core30 are rejection trials, NOT one SCAF class.
4. Existing WWS10 personalized protocol remains an auxiliary benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.task_views import (
    MDSC_COMMON_30,
    MDSC_WAKE_WORDS,
    canonical_mdsc_wake_word,
    normalize_mdsc_task_text,
)


OPEN_SET_LABEL = "__OPEN_SET__"

_NORMALIZED_CORE30_TO_CANONICAL = {
    normalize_mdsc_task_text(x): x for x in MDSC_COMMON_30
}
_NORMALIZED_WAKE_TO_CANONICAL = {
    normalize_mdsc_task_text(x): x for x in MDSC_WAKE_WORDS
}

MDSC_COMMAND_20 = tuple(
    phrase
    for phrase in MDSC_COMMON_30
    if normalize_mdsc_task_text(phrase) not in _NORMALIZED_WAKE_TO_CANONICAL
)

_NORMALIZED_COMMAND20_TO_CANONICAL = {
    normalize_mdsc_task_text(x): x for x in MDSC_COMMAND_20
}


@dataclass(frozen=True)
class MDSCTaskRecord:
    utt_id: str
    task_view: str
    task_label: str
    raw_transcript: str
    split: str
    domain: str
    role: str
    speaker_id: str
    is_wake_word: bool
    is_open_set: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_text(r: AudioManifestRecord) -> str:
    text = r.transcript if r.transcript is not None else r.label
    if text is None:
        raise ValueError(f"Missing transcript/label for {r.utt_id}")
    return text


def _base_record(
    r: AudioManifestRecord,
    *,
    task_view: str,
    task_label: str,
    is_open_set: bool,
) -> MDSCTaskRecord:
    if r.speaker_id is None:
        raise ValueError(f"Missing speaker_id for {r.utt_id}")
    raw = _raw_text(r)
    return MDSCTaskRecord(
        utt_id=r.utt_id,
        task_view=task_view,
        task_label=task_label,
        raw_transcript=raw,
        split=r.split,
        domain=r.domain,
        role=r.role,
        speaker_id=r.speaker_id,
        is_wake_word=canonical_mdsc_wake_word(raw) is not None,
        is_open_set=is_open_set,
    )


def mdsc_core30_index(
    records: Iterable[AudioManifestRecord],
) -> list[MDSCTaskRecord]:
    """All exact Core30 phrases across official splits/domains."""
    out = []
    for r in records:
        if r.dataset != "mdsc" or r.record_type != "speech":
            continue
        key = normalize_mdsc_task_text(_raw_text(r))
        canonical = _NORMALIZED_CORE30_TO_CANONICAL.get(key)
        if canonical is None:
            continue
        out.append(_base_record(
            r,
            task_view="mdsc_core30_exact_phrase",
            task_label=canonical,
            is_open_set=False,
        ))
    return out


def mdsc_command20_index(
    records: Iterable[AudioManifestRecord],
) -> list[MDSCTaskRecord]:
    """The 20 non-wake high-coverage commands."""
    out = []
    for r in records:
        if r.dataset != "mdsc" or r.record_type != "speech":
            continue
        key = normalize_mdsc_task_text(_raw_text(r))
        canonical = _NORMALIZED_COMMAND20_TO_CANONICAL.get(key)
        if canonical is None:
            continue
        out.append(_base_record(
            r,
            task_view="mdsc_command20_exact_phrase",
            task_label=canonical,
            is_open_set=False,
        ))
    return out


def mdsc_core30_open_set_eval_index(
    records: Iterable[AudioManifestRecord],
) -> list[MDSCTaskRecord]:
    """Dev/test non-Core30 utterances for rejection evaluation.

    These records are deliberately NOT a supervised SCAF class.
    """
    out = []
    for r in records:
        if (
            r.dataset != "mdsc"
            or r.record_type != "speech"
            or r.split not in {"dev", "test"}
        ):
            continue
        key = normalize_mdsc_task_text(_raw_text(r))
        if key in _NORMALIZED_CORE30_TO_CANONICAL:
            continue
        out.append(_base_record(
            r,
            task_view="mdsc_core30_open_set_eval",
            task_label=OPEN_SET_LABEL,
            is_open_set=True,
        ))
    return out


def summarize_mdsc_policy(
    core30: list[MDSCTaskRecord],
    command20: list[MDSCTaskRecord],
    open_set: list[MDSCTaskRecord],
) -> dict[str, Any]:
    def labels(rows):
        return sorted({r.task_label for r in rows})

    def counts(rows, field):
        out = {}
        for r in rows:
            value = getattr(r, field)
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    core30_labels = labels(core30)
    command20_labels = labels(command20)

    core30_train_labels = {
        r.task_label for r in core30 if r.split == "train"
    }
    core30_dev_labels = {
        r.task_label for r in core30 if r.split == "dev"
    }
    core30_test_labels = {
        r.task_label for r in core30 if r.split == "test"
    }

    core30_control_labels = {
        r.task_label for r in core30 if r.domain == "control"
    }
    core30_dys_labels = {
        r.task_label for r in core30 if r.domain == "dysarthria"
    }

    gate = {
        "core30_has_30_classes": len(core30_labels) == 30,
        "command20_has_20_classes": len(command20_labels) == 20,
        "core30_all_classes_have_train_support": len(core30_train_labels) == 30,
        "core30_all_classes_have_dev_support": len(core30_dev_labels) == 30,
        "core30_all_classes_have_test_support": len(core30_test_labels) == 30,
        "core30_all_classes_have_control_support": len(core30_control_labels) == 30,
        "core30_all_classes_have_dysarthria_support": len(core30_dys_labels) == 30,
        "open_set_dev_test_nonempty": len(open_set) > 0,
        "open_set_not_supervised_class": all(
            r.task_label == OPEN_SET_LABEL and r.is_open_set
            for r in open_set
        ),
    }
    gate["overall"] = (
        "PASS"
        if all(v is True for k, v in gate.items() if k != "overall")
        else "FAIL"
    )

    return {
        "schema": "papr_ssl.mdsc_task_policy.v2",
        "phase": "P2-07E",
        "policy": {
            "primary_public_phrase_benchmark": "mdsc_core30_exact_phrase",
            "command_only_diagnostic": "mdsc_command20_exact_phrase",
            "personalized_auxiliary_benchmark": "mdsc_personalized_wws_10",
            "open_set_evaluation": "mdsc_core30_open_set_eval",
            "long_tail_policy": (
                "Long-tail phrases outside Core30 are not collapsed into one "
                "SCAF training class. They are reserved for open-set/rejection "
                "evaluation or future auxiliary objectives."
            ),
            "semantic_merging": False,
            "raw_manifest_rewrite": False,
        },
        "views": {
            "mdsc_core30_exact_phrase": {
                "count": len(core30),
                "class_count": len(core30_labels),
                "labels": core30_labels,
                "splits": counts(core30, "split"),
                "domains": counts(core30, "domain"),
                "wake_count": sum(r.is_wake_word for r in core30),
                "non_wake_command_count": sum(
                    not r.is_wake_word for r in core30
                ),
            },
            "mdsc_command20_exact_phrase": {
                "count": len(command20),
                "class_count": len(command20_labels),
                "labels": command20_labels,
                "splits": counts(command20, "split"),
                "domains": counts(command20, "domain"),
            },
            "mdsc_core30_open_set_eval": {
                "count": len(open_set),
                "splits": counts(open_set, "split"),
                "domains": counts(open_set, "domain"),
                "task_label": OPEN_SET_LABEL,
                "use_as_scaf_training_class": False,
            },
        },
        "gate": gate,
    }
