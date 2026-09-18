"""Pytest controls for explicit PAPR-SSL real-checkpoint integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("papr-ssl real-model integration")
    group.addoption(
        "--run-real-models",
        action="store_true",
        default=False,
        help=(
            "Explicitly enable tests marked `real_model`. These tests may load "
            "multi-GB Hugging Face checkpoints."
        ),
    )
    group.addoption(
        "--hf-mode",
        choices=("cache", "network"),
        default="cache",
        help=(
            "Hugging Face access policy for real-model tests. "
            "`cache` forces offline/cache-only loading; `network` permits "
            "Hugging Face network access. Default: cache."
        ),
    )
    group.addoption(
        "--real-model-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help=(
            "Device used by real-model integration tests. "
            "Default: cpu; use cuda explicitly when desired."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_model: loads a pinned real SSL checkpoint; explicit opt-in only",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config):
    """
    Keep integration tests out of ordinary pytest collection.

    The existing `python -m unittest discover -s tests -v` path is also
    unaffected because these integration tests are pytest functions, not
    unittest.TestCase classes.
    """
    try:
        run_real = bool(config.getoption("--run-real-models"))
    except (ValueError, AttributeError):
        run_real = False

    path = Path(str(collection_path))
    if not run_real and "integration" in {part.lower() for part in path.parts}:
        return True
    return None
