"""P3-07 dev-only checkpoint selection.

Frozen rule:
    select checkpoint by MAX(generic_dev_score)
where:
    generic_dev_score == prototype_macro_f1

No generic_test information is accepted or consulted.

If multiple epochs tie on generic_dev_score, select the earliest epoch. This
deterministic tie-break is fixed before real experiments and avoids preferring a
later checkpoint without dev evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from papr_ssl.training.teacher.evaluate import GENERIC_DEV_SCORE_NAME


SELECTION_SCHEMA = "papr_ssl.p3_checkpoint_selection.v1"
SELECTION_METRIC = "generic_dev_score"
TIE_BREAK = "earliest_epoch"


@dataclass(frozen=True)
class SelectedCheckpoint:
    epoch: int
    checkpoint: str
    checkpoint_sha256: str
    generic_dev_score: float
    generic_dev_score_name: str
    selection_metric: str = SELECTION_METRIC
    tie_break: str = TIE_BREAK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_sha256(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")
        rows.append(obj)
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    return rows


def _reject_test_fields(obj: Any, *, path: str = "root") -> None:
    """Fail closed if a selection input contains generic-test information."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_text = str(key).lower()
            if "test" in key_text:
                raise RuntimeError(
                    "P3-07 sealed-test violation: selection input contains "
                    f"test-related field {path}.{key}"
                )
            _reject_test_fields(value, path=f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _reject_test_fields(value, path=f"{path}[{i}]")


def _validate_row(row: Mapping[str, Any]) -> None:
    _reject_test_fields(row)

    required = {
        "epoch",
        "dev",
        "checkpoint",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            "metrics row missing required fields: " + ", ".join(missing)
        )

    dev = row["dev"]
    if not isinstance(dev, Mapping):
        raise ValueError("metrics row dev field must be an object")

    if "generic_dev_score" not in dev:
        raise ValueError("dev.generic_dev_score is required")
    if dev.get("generic_dev_score_name") != GENERIC_DEV_SCORE_NAME:
        raise ValueError(
            "P3-07 requires generic_dev_score_name="
            f"{GENERIC_DEV_SCORE_NAME!r}"
        )

    epoch = int(row["epoch"])
    if epoch <= 0:
        raise ValueError("epoch must be > 0")

    score = float(dev["generic_dev_score"])
    if not (score == score):  # NaN
        raise ValueError("generic_dev_score must be finite")
    if score in (float("inf"), float("-inf")):
        raise ValueError("generic_dev_score must be finite")

    sha = str(row["checkpoint_sha256"])
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("checkpoint_sha256 must be lowercase SHA-256 hex")


def select_checkpoint_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    verify_hash: bool = True,
) -> SelectedCheckpoint:
    if not rows:
        raise ValueError("rows must be non-empty")

    validated: list[Mapping[str, Any]] = []
    seen_epochs: set[int] = set()
    for row in rows:
        _validate_row(row)
        epoch = int(row["epoch"])
        if epoch in seen_epochs:
            raise ValueError(f"duplicate epoch in selection rows: {epoch}")
        seen_epochs.add(epoch)

        if verify_hash:
            checkpoint = Path(str(row["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"checkpoint does not exist: {checkpoint}"
                )
            actual = checkpoint_sha256(checkpoint)
            expected = str(row["checkpoint_sha256"])
            if actual != expected:
                raise RuntimeError(
                    "checkpoint hash mismatch: "
                    f"{checkpoint}; expected={expected}, actual={actual}"
                )

        validated.append(row)

    # Maximum dev score. On an exact tie, earliest epoch wins.
    best = min(
        validated,
        key=lambda row: (
            -float(row["dev"]["generic_dev_score"]),
            int(row["epoch"]),
        ),
    )
    return SelectedCheckpoint(
        epoch=int(best["epoch"]),
        checkpoint=str(best["checkpoint"]),
        checkpoint_sha256=str(best["checkpoint_sha256"]),
        generic_dev_score=float(best["dev"]["generic_dev_score"]),
        generic_dev_score_name=str(
            best["dev"]["generic_dev_score_name"]
        ),
    )


def select_run_checkpoint(
    run_dir: str | Path,
    *,
    output_name: str = "selected_checkpoint.json",
    verify_hash: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    output_path = run_dir / output_name

    if output_path.exists():
        raise FileExistsError(
            f"selection output already exists: {output_path}"
        )

    rows = load_metrics_jsonl(metrics_path)
    selected = select_checkpoint_from_rows(
        rows,
        verify_hash=verify_hash,
    )
    payload = {
        "schema": SELECTION_SCHEMA,
        "phase": "P3-07",
        "run_dir": run_dir.as_posix(),
        "selection_policy": {
            "metric": SELECTION_METRIC,
            "metric_definition": GENERIC_DEV_SCORE_NAME,
            "direction": "maximize",
            "tie_break": TIE_BREAK,
            "generic_test_accessed": False,
        },
        "selected": selected.to_dict(),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def select_multiseed_checkpoints(
    experiment_dir: str | Path,
    *,
    seeds: Sequence[int] = (17, 29, 43),
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Select exactly one checkpoint per seed using DEV only."""
    experiment_dir = Path(experiment_dir)

    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = experiment_dir / f"seed_{int(seed):04d}"
        payload = select_run_checkpoint(
            run_dir,
            verify_hash=verify_hash,
        )
        rows.append(
            {
                "seed": int(seed),
                **payload["selected"],
            }
        )

    output_path = experiment_dir / "selected_checkpoints.json"
    if output_path.exists():
        raise FileExistsError(
            f"selection output already exists: {output_path}"
        )

    payload = {
        "schema": "papr_ssl.p3_multiseed_selection.v1",
        "phase": "P3-07",
        "selection_policy": {
            "metric": SELECTION_METRIC,
            "metric_definition": GENERIC_DEV_SCORE_NAME,
            "direction": "maximize",
            "tie_break": TIE_BREAK,
            "generic_test_accessed": False,
        },
        "seeds": [int(x) for x in seeds],
        "selected_checkpoints": rows,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
