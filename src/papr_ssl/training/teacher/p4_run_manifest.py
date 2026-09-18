"""P4 extension helpers for the existing teacher run manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from papr_ssl.training.teacher.run_manifest import (
    write_run_manifest as _write_run_manifest,
)


def write_p4_run_manifest(**kwargs: Any) -> dict[str, Any]:
    """Write the standard run manifest, then relabel this new artifact as P4-02.

    Existing P3 manifests are untouched.  P4 reuses the same provenance schema
    and integrity checks because the training contract is intentionally the
    same; only Backbone/hidden-state index changes.
    """
    manifest = _write_run_manifest(**kwargs)
    path = Path(kwargs["run_dir"]) / "run_manifest.json"

    # The standard writer has already self-hashed the file.  Do not mutate it
    # afterward.  P4 phase identity is instead recorded in a sidecar.
    sidecar = Path(kwargs["run_dir"]) / "p4_phase.json"
    sidecar.write_text(
        '{\n'
        '  "schema": "papr_ssl.p4_run_phase.v1",\n'
        '  "phase": "P4-02",\n'
        '  "run_manifest_schema": "papr_ssl.teacher_run_manifest.v1"\n'
        '}\n',
        encoding="utf-8",
    )
    return manifest
