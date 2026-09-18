"""Frame-level cache dataset and class-balanced sampler for P5."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


class FrameCacheDataset(Dataset):
    def __init__(
        self,
        *,
        cache_dir: Path,
        task_index: Path,
        split: str,
        class_to_index: dict[str, int] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split

        index_rows = [
            json.loads(line)
            for line in (self.cache_dir / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        index_by_utt = {row["utt_id"]: row for row in index_rows}

        task_rows = [
            json.loads(line)
            for line in Path(task_index).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        task_rows = [row for row in task_rows if row.get("split") == split]

        def label_of(row: dict) -> str:
            for key in (
                "label",
                "task_label",
                "normalized_phrase",
                "phrase",
                "transcript",
            ):
                value = row.get(key)
                if isinstance(value, str) and value:
                    return value
            raise KeyError(f"no usable label field in task row: {row.keys()}")

        labels = [label_of(row) for row in task_rows]
        if class_to_index is None:
            unique = sorted(set(labels))
            self.class_to_index = {label: i for i, label in enumerate(unique)}
        else:
            self.class_to_index = dict(class_to_index)

        self.rows = []
        for task_row, label in zip(task_rows, labels):
            utt_id = task_row["utt_id"]
            if utt_id not in index_by_utt:
                raise KeyError(f"missing frame cache entry: {utt_id}")
            if label not in self.class_to_index:
                raise KeyError(f"dev label absent from train classes: {label}")
            row = dict(index_by_utt[utt_id])
            row["label_index"] = self.class_to_index[label]
            row["label_text"] = label
            self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"no rows for split={split}")

        self._shard_cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> dict:
        if name not in self._shard_cache:
            self._shard_cache[name] = torch.load(
                self.cache_dir / name,
                map_location="cpu",
                weights_only=False,
            )
        return self._shard_cache[name]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        shard = self._load_shard(row["shard"])
        feat = shard["features"][
            int(row["start_frame"]):int(row["end_frame"])
        ].to(torch.float32)
        return {
            "utt_id": row["utt_id"],
            "features": feat,
            "label": int(row["label_index"]),
        }


def frame_collate(batch: list[dict]) -> dict:
    b = len(batch)
    max_t = max(int(x["features"].shape[0]) for x in batch)
    dim = int(batch[0]["features"].shape[1])
    features = torch.zeros((b, max_t, dim), dtype=torch.float32)
    mask = torch.zeros((b, max_t), dtype=torch.bool)
    labels = torch.empty((b,), dtype=torch.long)
    utt_ids = []
    for i, item in enumerate(batch):
        x = item["features"]
        t = int(x.shape[0])
        features[i, :t] = x
        mask[i, :t] = True
        labels[i] = int(item["label"])
        utt_ids.append(item["utt_id"])
    return {
        "utt_ids": utt_ids,
        "features": features,
        "frame_mask": mask,
        "labels": labels,
    }


class ClassBalancedBatchSampler(Sampler[list[int]]):
    """8-way x 4-shot style deterministic class-balanced batch sampler."""

    def __init__(
        self,
        dataset: FrameCacheDataset,
        *,
        classes_per_batch: int = 8,
        samples_per_class: int = 4,
        seed: int = 17,
    ):
        self.dataset = dataset
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed

        by_class: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(dataset.rows):
            by_class[int(row["label_index"])].append(i)
        self.by_class = dict(by_class)
        self.classes = sorted(self.by_class)
        if len(self.classes) < classes_per_batch:
            raise RuntimeError("not enough classes for balanced sampler")

        self.batch_size = classes_per_batch * samples_per_class
        self.batches_per_epoch = max(1, len(dataset) // self.batch_size)
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + 100003 * self.epoch)
        for _ in range(self.batches_per_epoch):
            chosen_classes = rng.sample(self.classes, self.classes_per_batch)
            batch = []
            for c in chosen_classes:
                pool = self.by_class[c]
                if len(pool) >= self.samples_per_class:
                    batch.extend(rng.sample(pool, self.samples_per_class))
                else:
                    batch.extend(
                        rng.choice(pool)
                        for _ in range(self.samples_per_class)
                    )
            rng.shuffle(batch)
            yield batch
