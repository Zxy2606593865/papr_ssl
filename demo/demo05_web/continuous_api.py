"""Continuous demo routes for Demo-05.

Integrate from demo/demo05_web/app.py with:

    from demo.demo05_web.continuous_api import install_continuous_api
    install_continuous_api(app=app, engine=engine, static_root=STATIC_ROOT)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

from papr_ssl.inference.h6_continuous_phrase_spotter import (
    ContinuousPhraseSpotter,
    SpotterConfig,
)
from papr_ssl.inference.h6_raw_wav_adapter import load_wav_mono_16k


def install_continuous_api(*, app: Any, engine: Any, static_root: Path) -> None:
    demo_wav = static_root / "audio/continuous/continuous_natural_30s.wav"
    demo_json = static_root / "audio/continuous/continuous_natural_30s.json"

    def build_spotter() -> ContinuousPhraseSpotter:
        if not engine.loaded:
            engine.load()
        if engine.adapter is None or engine.runtime is None or engine.memory is None:
            raise RuntimeError("PAPR DemoEngine is not ready")
        return ContinuousPhraseSpotter(
            adapter=engine.adapter,
            runtime=engine.runtime,
            memory=engine.memory,
            config=SpotterConfig(
                boundary_mode="auto",
                min_boundary_sec=0.12,
                # This frozen demo fixture has explicit zero-gap phrase
                # boundaries, so merged candidates would only add latency.
                max_merge_spans=1,
                allow_confirm=False,
            ),
        )

    @app.get("/continuous")
    def continuous_page() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/continuous/demo-info")
    def continuous_demo_info() -> dict[str, Any]:
        if not demo_json.exists():
            raise HTTPException(status_code=404, detail="Continuous demo metadata missing")
        return json.loads(demo_json.read_text(encoding="utf-8"))

    @app.post("/api/continuous/recognize-demo")
    def continuous_recognize_demo() -> dict[str, Any]:
        if not demo_wav.exists():
            raise HTTPException(status_code=404, detail="Continuous demo WAV missing")
        try:
            result = build_spotter().spot_wav(str(demo_wav), load_wav_mono_16k)
            result["audio_url"] = "/static/audio/continuous/continuous_natural_30s.wav"
            return result
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            )
