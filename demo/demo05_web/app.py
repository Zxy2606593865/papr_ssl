"""PAPR-SSL Demo-05 web application.

Client-facing principle:
    atypical speech -> canonical Chinese text

`intent_id` remains an internal runtime field. The primary UI renders
`canonical_text`, not class labels or uncalibrated C/W/U scores.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
PRESETS_JSON = WEB_ROOT / "demo_presets.json"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from papr_ssl.inference.h6_personalized_runtime import (  # noqa: E402
    PersonalizedRuntime,
    UserMemory,
)
from papr_ssl.inference.h6_raw_wav_adapter import (  # noqa: E402
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


class DemoEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loaded = False
        self.load_error: str | None = None
        self.memory: UserMemory | None = None
        self.runtime: PersonalizedRuntime | None = None
        self.adapter: RawWavFeatureAdapter | None = None

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if self.load_error is not None:
                raise RuntimeError(self.load_error)

            try:
                memory_path = PROJECT_ROOT / "artifacts/demo_04a_wanghao_runtime_v2/wanghao_user_memory.pt"
                head_json = PROJECT_ROOT / "artifacts/h5_01_cwu_decision_head/cwu_head.json"
                policy_json = PROJECT_ROOT / "artifacts/h6_01_three_state_policy/h6_01_eval.json"
                embedding_manifest = PROJECT_ROOT / "artifacts/p6_teacher_256_15shot/embeddings/manifest.json"

                for p in (memory_path, head_json, policy_json, embedding_manifest):
                    if not p.exists():
                        raise FileNotFoundError(p)

                self.memory = UserMemory.load(memory_path)
                self.runtime = PersonalizedRuntime(
                    head_json=head_json,
                    policy_json=policy_json,
                )
                checkpoint = resolve_selected_checkpoint(embedding_manifest)
                self.adapter = RawWavFeatureAdapter(checkpoint=checkpoint)
                self.loaded = True
            except Exception as exc:  # retain startup diagnostics
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise

    def recognize(self, wav_path: Path) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        assert self.memory is not None
        assert self.runtime is not None
        assert self.adapter is not None

        start = time.perf_counter()
        feat = self.adapter.extract_wav(wav_path)
        pred = self.runtime.predict_feature(
            memory=self.memory,
            query_global=feat["global_embedding"],
            query_temporal=feat["temporal_sequence"],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        status = str(pred["status"])
        if status == "ACCEPT":
            display_status = "识别成功"
            display_text = pred["canonical_text"]
            display_level = "success"
        elif status == "CONFIRM":
            display_status = "建议确认"
            display_text = pred["canonical_text"] or "请确认识别结果"
            display_level = "warning"
        else:
            display_status = "未识别到已注册表达"
            display_text = "请重新尝试，或补充该学生的个性化语音样本。"
            display_level = "neutral"

        # Keep the internal intent for diagnostics / future Agent routing,
        # but the client UI is designed around canonical_text.
        return {
            "status": status,
            "display_status": display_status,
            "display_level": display_level,
            "canonical_text": pred["canonical_text"],
            "display_text": display_text,
            "intent_id": pred["intent_id"],
            "latency_ms": round(elapsed_ms, 1),
            "scores_are_calibrated_probabilities": False,
        }


def load_presets() -> list[dict[str, Any]]:
    if not PRESETS_JSON.exists():
        return []
    payload = json.loads(PRESETS_JSON.read_text(encoding="utf-8"))
    return list(payload.get("presets", []))


engine = DemoEngine()

app = FastAPI(
    title="特殊儿童个性化语音识别与智能学伴接入系统",
    version="demo-05",
)

app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.on_event("startup")
def startup() -> None:
    # Load once before the presentation so the first click does not pay the
    # model initialization cost. A startup failure is exposed through health.
    try:
        engine.load()
    except Exception:
        pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    presets = load_presets()
    user_id = engine.memory.user_id if engine.memory is not None else "wanghao"
    registered = len(engine.memory.intents) if engine.memory is not None else None
    return {
        "service": "running",
        "model_loaded": engine.loaded,
        "model_error": engine.load_error,
        "user_id": user_id,
        "registered_expression_count": registered,
        "preset_count": len(presets),
    }


@app.get("/api/profile")
def profile() -> dict[str, Any]:
    if not engine.loaded:
        try:
            engine.load()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    assert engine.memory is not None
    expressions = [
        {
            "canonical_text": im.canonical_text,
            "shot": len(im.examples),
        }
        for _, im in sorted(engine.memory.intents.items())
    ]
    return {
        "student_id": engine.memory.user_id,
        "student_name": "王灏",
        "library_status": "已建立",
        "registered_expression_count": len(expressions),
        "enrollment_per_expression": engine.memory.shot_count(),
        "expressions": expressions,
    }


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    items = load_presets()
    public_items = [
        {
            "id": x["id"],
            "display_name": x["display_name"],
            "description": x.get("description", "王灏真实录音"),
            "audio_url": x["audio_url"],
            "demo_kind": x.get("demo_kind", "recognition"),
        }
        for x in items
    ]
    return {"presets": public_items}


@app.post("/api/recognize/{preset_id}")
def recognize_preset(preset_id: str) -> dict[str, Any]:
    items = load_presets()
    item = next((x for x in items if x["id"] == preset_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Preset not found")

    wav_path = WEB_ROOT / item["wav_relpath"]
    if not wav_path.exists():
        raise HTTPException(status_code=500, detail=f"Preset WAV missing: {wav_path}")

    try:
        result = engine.recognize(wav_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return {
        "preset_id": preset_id,
        "source": "王灏真实录音",
        **result,
    }
