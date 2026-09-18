"""P3-06 multi-seed orchestration.

Purpose:
- repeat the *same* experiment with at least seeds 17 / 29 / 43;
- seed BEFORE model/sampler construction so initialization is truly controlled;
- keep one immutable run directory per seed;
- aggregate same-epoch diagnostics without performing checkpoint selection.

Checkpoint selection remains P3-07.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from papr_ssl.training.teacher.train import (
    OptimizationConfig,
    run_training,
    seed_everything,
)


REQUIRED_P3_SEEDS = (17, 29, 43)
SUMMARY_SCHEMA = "papr_ssl.p3_multiseed_summary.v1"


@dataclass(frozen=True)
class SeedRunSpec:
    """Everything needed for one P3-05 training run.

    The factory producing this object is called only AFTER seed_everything(seed),
    so Mean DR / SCAF initialization can depend reproducibly on the seed.
    """

    mean_dr: nn.Module
    scaf: nn.Module
    train_loader: Any
    train_reference_loader: Any
    dev_loader: Any
    num_classes: int
    device: torch.device
    optimization: OptimizationConfig

    def validate(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        self.optimization.validate()


@dataclass(frozen=True)
class MultiSeedConfig:
    seeds: tuple[int, ...] = REQUIRED_P3_SEEDS

    def validate(self) -> None:
        if len(self.seeds) < 3:
            raise ValueError("P3-06 requires at least three seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("P3-06 seeds must be unique")
        if any((not isinstance(x, int)) or x < 0 for x in self.seeds):
            raise ValueError("all seeds must be non-negative integers")

        missing = sorted(set(REQUIRED_P3_SEEDS) - set(self.seeds))
        if missing:
            raise ValueError(
                "P3-06 requires seeds 17 / 29 / 43; missing: "
                + ", ".join(map(str, missing))
            )


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample standard deviation needs >= 2 values")
    return float(stdev(values))


def aggregate_scalar(
    values_by_seed: Mapping[int, float],
) -> dict[str, Any]:
    """Aggregate one scalar across seeds using sample std (ddof=1)."""
    if len(values_by_seed) < 3:
        raise ValueError("P3-06 aggregation requires at least three seeds")

    ordered = {
        str(seed): float(values_by_seed[seed])
        for seed in sorted(values_by_seed)
    }
    values = list(ordered.values())
    return {
        "values_by_seed": ordered,
        "mean": float(mean(values)),
        "sample_std": _sample_std(values),
        "std_ddof": 1,
        "n": len(values),
    }


def summarize_final_epoch(
    histories: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize the same fixed endpoint (last epoch) across seeds.

    This is *not* checkpoint selection. P3-07 will select checkpoints using only
    generic_dev_score after every seed run has completed.
    """
    if len(histories) < 3:
        raise ValueError("need at least three seed histories")

    final_epochs: dict[int, int] = {}
    final_dev_scores: dict[int, float] = {}
    final_train_losses: dict[int, float] = {}

    for seed, history in histories.items():
        if not history:
            raise ValueError(f"seed {seed} has empty training history")
        last = history[-1]
        final_epochs[seed] = int(last["epoch"])
        final_train_losses[seed] = float(last["train_loss"])
        final_dev_scores[seed] = float(
            last["dev"]["generic_dev_score"]
        )

    unique_epochs = set(final_epochs.values())
    if len(unique_epochs) != 1:
        raise ValueError(
            "P3-06 final-epoch aggregation requires equal training budgets "
            f"across seeds, got {final_epochs}"
        )

    return {
        "endpoint": "final_epoch_only_no_checkpoint_selection",
        "epoch": next(iter(unique_epochs)),
        "generic_dev_score": aggregate_scalar(final_dev_scores),
        "train_loss": aggregate_scalar(final_train_losses),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_multiseed_experiment(
    *,
    experiment_dir: str | Path,
    build_seed_run: Callable[[int, Path], SeedRunSpec],
    config: MultiSeedConfig = MultiSeedConfig(),
) -> dict[str, Any]:
    """Run the same experiment for every frozen P3 seed.

    Critical ordering:
        seed_everything(seed)
        -> build model / SCAF / seed-aware sampler
        -> P3-05 run_training(...)

    This prevents the common bug where the model is initialized before the seed
    is set.
    """
    config.validate()
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary_path = experiment_dir / "multiseed_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            "multiseed_summary.json already exists; use a new immutable "
            "experiment directory"
        )

    histories: dict[int, list[dict[str, Any]]] = {}
    run_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        run_dir = experiment_dir / f"seed_{seed:04d}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"seed run directory is not empty: {run_dir}"
            )

        # MUST happen before build_seed_run().
        seed_everything(seed)

        spec = build_seed_run(seed, run_dir)
        spec.validate()

        history = run_training(
            mean_dr=spec.mean_dr,
            scaf=spec.scaf,
            train_loader=spec.train_loader,
            train_reference_loader=spec.train_reference_loader,
            dev_loader=spec.dev_loader,
            num_classes=spec.num_classes,
            seed=seed,
            run_dir=run_dir,
            device=spec.device,
            optimization=spec.optimization,
        )
        histories[seed] = history

        last = history[-1]
        run_rows.append(
            {
                "seed": seed,
                "run_dir": run_dir.as_posix(),
                "epochs": len(history),
                "final_epoch": int(last["epoch"]),
                "final_train_loss": float(last["train_loss"]),
                "final_generic_dev_score": float(
                    last["dev"]["generic_dev_score"]
                ),
                "final_checkpoint": str(last["checkpoint"]),
                "final_checkpoint_sha256": str(
                    last["checkpoint_sha256"]
                ),
            }
        )

    endpoint_summary = summarize_final_epoch(histories)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "phase": "P3-06",
        "required_seeds": list(REQUIRED_P3_SEEDS),
        "seeds": list(config.seeds),
        "seed_count": len(config.seeds),
        "selection_policy": (
            "none_in_p3_06; checkpoint selection is deferred to P3-07"
        ),
        "std_definition": "sample_std_ddof_1",
        "runs": run_rows,
        "final_epoch_summary": endpoint_summary,
    }
    _write_json(summary_path, summary)
    return summary
