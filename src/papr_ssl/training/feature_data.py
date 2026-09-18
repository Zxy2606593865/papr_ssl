"""P3-04 cached-feature Dataset / class-balanced batch sampler.

Frozen P3-04 policy:
- training consumes P3-02 offline SSL masked-mean vectors [D];
- class ids are defined from TRAIN only and reused by dev;
- train batches use explicit C-way x K-shot composition;
- class exposure over an epoch is balanced by construction;
- dev is deterministic, complete, and never rebalanced;
- test is not constructed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from papr_ssl.cache.ssl_feature_cache import (
    MeanFeatureCacheReader,
    SSLCacheIdentity,
)


@dataclass(frozen=True)
class CachedTaskRow:
    dataset: str
    utt_id: str
    split: str
    task_label: str
    sample_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.utt_id:
            raise ValueError("utt_id must be non-empty")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"unsupported split: {self.split!r}")
        if not self.task_label:
            raise ValueError("task_label must be non-empty")


@dataclass(frozen=True)
class ClassBalancedSamplerConfig:
    classes_per_batch: int = 8
    samples_per_class: int = 4
    batches_per_epoch: int | None = None
    seed: int = 17

    @property
    def batch_size(self) -> int:
        return self.classes_per_batch * self.samples_per_class

    def validate(self, *, num_classes: int, num_samples: int) -> None:
        if self.classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be > 0")
        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class must be > 0")
        if self.classes_per_batch > num_classes:
            raise ValueError(
                "classes_per_batch cannot exceed number of train classes"
            )
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.batches_per_epoch is not None and self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be > 0 when specified")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

    def resolved_batches_per_epoch(self, *, num_samples: int) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        return max(1, math.ceil(num_samples / self.batch_size))


def _task_value(
    obj: Mapping[str, Any],
    names: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    for name in names:
        value = obj.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    if required:
        raise KeyError("/".join(names))
    return None


def _dataset_from_task_row(
    obj: Mapping[str, Any],
    *,
    utt_id: str,
) -> str:
    """Resolve dataset without forcing task views to duplicate the field.

    Current P2 task views may use globally-qualified utterance ids such as:

        mdsc:Control/dev/wav/CF0010/CF0010_0001.wav

    In that case `dataset` is derivable from the prefix and need not be copied
    into every task-view row. Generic task views that do not use a qualified id
    must still provide an explicit dataset field.
    """
    explicit = obj.get("dataset")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    if ":" in utt_id:
        prefix, remainder = utt_id.split(":", 1)
        prefix = prefix.strip()
        if prefix and remainder.strip():
            return prefix

    raise KeyError("dataset (or dataset-qualified utt_id)")


def _load_task_rows(path: str | Path) -> list[CachedTaskRow]:
    """Load a P2 task view for joining with the audio-centric P3 cache.

    The P3 cache stores physical/audio identity. The task view stores task
    membership/labels. It is therefore valid for a task-view row to omit
    `dataset` and `sample_hash` when those are recoverable from the qualified
    utterance id / cache index.
    """
    path = Path(path)
    rows: list[CachedTaskRow] = []

    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue

        obj = json.loads(raw)
        if not isinstance(obj, Mapping):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")

        try:
            utt_id = _task_value(
                obj,
                ("utt_id", "utterance_id", "id"),
            )
            assert utt_id is not None

            task_label = _task_value(
                obj,
                (
                    "task_label",
                    "label",
                    "phrase_id",
                    "normalized_phrase",
                    "transcript",
                ),
            )
            assert task_label is not None

            split = _task_value(obj, ("split",))
            assert split is not None

            row = CachedTaskRow(
                dataset=_dataset_from_task_row(
                    obj,
                    utt_id=utt_id,
                ),
                utt_id=utt_id,
                split=split.lower(),
                task_label=task_label,
                sample_hash=_task_value(
                    obj,
                    ("sample_hash",),
                    required=False,
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"{path}:{line_no}: missing required task-view identity "
                f"{exc.args[0]!r}"
            ) from exc

        rows.append(row)

    if not rows:
        raise ValueError(f"task index is empty: {path}")
    return rows


def _load_cache_index(cache_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index_path = cache_dir / "index.jsonl"
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line_no, raw in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        key = (str(obj["dataset"]), str(obj["utt_id"]))
        if key in rows:
            raise RuntimeError(
                f"{index_path}:{line_no}: duplicate cache key {key}"
            )
        rows[key] = obj
    if not rows:
        raise RuntimeError(f"cache index is empty: {index_path}")
    return rows


def class_map_from_train_rows(
    rows: Sequence[CachedTaskRow],
) -> dict[str, int]:
    train_labels = sorted(
        {row.task_label for row in rows if row.split == "train"}
    )
    if not train_labels:
        raise ValueError("no train labels found")
    return {label: i for i, label in enumerate(train_labels)}


class OfflineFeatureTaskDataset(Dataset[dict[str, Any]]):
    """Join a task-view index with one P3-02 cache identity."""

    def __init__(
        self,
        *,
        task_index: str | Path,
        cache_dir: str | Path,
        cache_identity: SSLCacheIdentity,
        split: str,
        class_to_index: Mapping[str, int] | None = None,
    ) -> None:
        if split not in {"train", "dev"}:
            raise ValueError(
                "P3-04 may construct only train/dev datasets; test stays sealed"
            )

        all_rows = _load_task_rows(task_index)
        if class_to_index is None:
            if split != "train":
                raise ValueError(
                    "dev dataset must reuse class_to_index defined by train"
                )
            class_to_index = class_map_from_train_rows(all_rows)

        self.class_to_index = dict(class_to_index)
        self.class_names = tuple(
            label
            for label, _ in sorted(
                self.class_to_index.items(),
                key=lambda kv: kv[1],
            )
        )
        expected_ids = set(range(len(self.class_to_index)))
        if set(self.class_to_index.values()) != expected_ids:
            raise ValueError(
                "class_to_index values must be contiguous integers 0..C-1"
            )

        self.rows = [row for row in all_rows if row.split == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r}")

        unknown = sorted(
            {row.task_label for row in self.rows}
            - set(self.class_to_index)
        )
        if unknown:
            raise ValueError(
                "split contains labels absent from train class map: "
                + ", ".join(unknown[:8])
            )

        self.cache_dir = Path(cache_dir)
        self.cache_identity = cache_identity
        self.reader = MeanFeatureCacheReader(
            self.cache_dir,
            cache_identity,
        )
        self.cache_index = _load_cache_index(self.cache_dir)

        missing: list[tuple[str, str]] = []
        for row in self.rows:
            key = (row.dataset, row.utt_id)
            if key not in self.cache_index:
                missing.append(key)
                continue

            cache_row = self.cache_index[key]
            if str(cache_row["split"]) != row.split:
                raise RuntimeError(
                    f"task/cache split mismatch for {key}: "
                    f"{row.split!r} != {cache_row['split']!r}"
                )
            if (
                row.sample_hash is not None
                and row.sample_hash != str(cache_row["sample_hash"])
            ):
                raise RuntimeError(
                    f"task/cache sample_hash mismatch for {key}"
                )

        if missing:
            preview = ", ".join(map(str, missing[:8]))
            raise RuntimeError(
                f"{len(missing)} task rows missing from feature cache: "
                f"{preview}"
            )

        self.labels = tuple(
            self.class_to_index[row.task_label]
            for row in self.rows
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        key = (row.dataset, row.utt_id)
        cache_row = self.cache_index[key]
        sample_hash = str(cache_row["sample_hash"])

        vector, frame_count = self.reader.get(
            dataset=row.dataset,
            utt_id=row.utt_id,
            expected_sample_hash=sample_hash,
        )
        if vector.ndim != 1:
            raise RuntimeError("cached feature vector must have shape [D]")
        if not torch.isfinite(vector).all():
            raise RuntimeError("cached feature contains NaN/Inf")

        return {
            "features": vector.to(dtype=torch.float32),
            "label": int(self.class_to_index[row.task_label]),
            "task_label": row.task_label,
            "dataset": row.dataset,
            "utt_id": row.utt_id,
            "split": row.split,
            "sample_hash": sample_hash,
            "frame_count": int(frame_count),
        }


class ClassBalancedBatchSampler(BatchSampler):
    """Generate deterministic C-way x K-shot batches.

    Global class exposure is balanced greedily: at each batch we choose distinct
    classes with the currently smallest exposure counts, randomizing only ties.
    This keeps max/min class appearances within one count over an epoch.

    Within a selected class:
    - if class size >= K, K distinct samples are drawn without replacement
      inside that batch;
    - if class size < K, sampling with replacement is allowed explicitly.
    """

    def __init__(
        self,
        labels: Sequence[int],
        config: ClassBalancedSamplerConfig,
    ) -> None:
        self.labels = tuple(int(x) for x in labels)
        if not self.labels:
            raise ValueError("labels must be non-empty")

        classes = sorted(set(self.labels))
        if classes != list(range(len(classes))):
            raise ValueError(
                "labels must be contiguous class ids 0..C-1"
            )

        self.num_classes = len(classes)
        self.config = config
        config.validate(
            num_classes=self.num_classes,
            num_samples=len(self.labels),
        )
        self.batches_per_epoch = config.resolved_batches_per_epoch(
            num_samples=len(self.labels)
        )
        self.epoch = 0

        pools: dict[int, list[int]] = {
            c: [] for c in range(self.num_classes)
        }
        for idx, label in enumerate(self.labels):
            pools[label].append(idx)
        if any(not pool for pool in pools.values()):
            raise ValueError("every class must contain at least one sample")
        self.pools = pools

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self.epoch = int(epoch)

    def _class_schedule(
        self,
        rng: random.Random,
    ) -> list[list[int]]:
        exposure = [0] * self.num_classes
        schedule: list[list[int]] = []

        for _ in range(self.batches_per_epoch):
            tie_break = {
                c: rng.random()
                for c in range(self.num_classes)
            }
            ordered = sorted(
                range(self.num_classes),
                key=lambda c: (exposure[c], tie_break[c]),
            )
            chosen = ordered[: self.config.classes_per_batch]
            for c in chosen:
                exposure[c] += 1
            rng.shuffle(chosen)
            schedule.append(chosen)

        if max(exposure) - min(exposure) > 1:
            raise RuntimeError(
                "internal sampler error: class exposure is not balanced"
            )
        return schedule

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.config.seed + self.epoch)
        schedule = self._class_schedule(rng)

        for classes in schedule:
            batch: list[int] = []
            for class_id in classes:
                pool = self.pools[class_id]
                k = self.config.samples_per_class
                if len(pool) >= k:
                    chosen = rng.sample(pool, k)
                else:
                    chosen = [rng.choice(pool) for _ in range(k)]
                batch.extend(chosen)
            rng.shuffle(batch)
            yield batch


def feature_cache_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must be non-empty")

    features = torch.stack(
        [item["features"] for item in batch],
        dim=0,
    ).to(torch.float32)
    labels = torch.tensor(
        [int(item["label"]) for item in batch],
        dtype=torch.long,
    )

    if features.ndim != 2:
        raise RuntimeError("collated cached features must have shape [B,D]")
    if not torch.isfinite(features).all():
        raise RuntimeError("collated features contain NaN/Inf")

    return {
        "features": features,
        "labels": labels,
        "task_labels": [str(x["task_label"]) for x in batch],
        "datasets": [str(x["dataset"]) for x in batch],
        "utt_ids": [str(x["utt_id"]) for x in batch],
        "splits": [str(x["split"]) for x in batch],
        "sample_hashes": [str(x["sample_hash"]) for x in batch],
        "frame_counts": torch.tensor(
            [int(x["frame_count"]) for x in batch],
            dtype=torch.long,
        ),
    }


def build_train_loader(
    dataset: OfflineFeatureTaskDataset,
    *,
    sampler_config: ClassBalancedSamplerConfig,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, ClassBalancedBatchSampler]:
    if any(row.split != "train" for row in dataset.rows):
        raise ValueError("build_train_loader requires a train dataset")

    sampler = ClassBalancedBatchSampler(
        dataset.labels,
        sampler_config,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_cache_collate,
    )
    return loader, sampler


def build_dev_loader(
    dataset: OfflineFeatureTaskDataset,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if any(row.split != "dev" for row in dataset.rows):
        raise ValueError("build_dev_loader requires a dev dataset")

    # Deterministic full pass: no shuffle, no class rebalancing, no drop_last.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_cache_collate,
    )
