"""
PAPR-SSL P1-11 real-checkpoint integration tests.

These tests are intentionally NOT part of the default offline suite.

Explicit usage:

Cache-only (recommended after P1-10 checkpoints are cached):
    python -m pytest tests/integration/test_real_ssl_backbones.py \
        --run-real-models --hf-mode cache --real-model-device cpu -v

Allow Hugging Face network access:
    python -m pytest tests/integration/test_real_ssl_backbones.py \
        --run-real-models --hf-mode network --real-model-device cuda -v

The tests validate the pinned P1 contract only. They do not train Mean DR,
SCAF, Attention DR, KD, Student, or access datasets.
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import yaml


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000

EXPECTED = {
    "wav2vec2": {
        "config": "configs/teacher/wav2vec2_mean.yaml",
        "checkpoint": "facebook/wav2vec2-base",
        "revision": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
        "configured_hidden_layer": 12,
        "hidden_states_count": 13,
        "feature_dim": 768,
        "frame_count": 49,
    },
    "wavlm": {
        "config": "configs/teacher/wavlm_mean.yaml",
        "checkpoint": "microsoft/wavlm-large",
        "revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
        "configured_hidden_layer": 24,
        "hidden_states_count": 25,
        "feature_dim": 1024,
        "frame_count": 49,
    },
    "w2v_bert2": {
        "config": "configs/teacher/w2v_bert2_mean.yaml",
        "checkpoint": "facebook/w2v-bert-2.0",
        "revision": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        "configured_hidden_layer": 24,
        "hidden_states_count": 25,
        "feature_dim": 1024,
        "frame_count": 49,
    },
}


def _project_root() -> Path:
    # tests/integration/test_real_ssl_backbones.py -> project root
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return payload


def _extract_backbone_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    direct = config.get("backbone")
    if isinstance(direct, Mapping):
        return dict(direct)

    teacher = config.get("teacher")
    if isinstance(teacher, Mapping):
        nested = teacher.get("backbone")
        if isinstance(nested, Mapping):
            return dict(nested)

    if "kind" in config:
        return dict(config)

    raise KeyError("Could not locate backbone configuration.")


def _normalize_kind(value: Any) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def _configured_model_name(backbone_cfg: Mapping[str, Any]) -> str | None:
    """Return the configured Hugging Face model identifier across supported aliases."""
    for key in (
        "model_name",
        "checkpoint",
        "model_name_or_path",
        "pretrained_model_name_or_path",
    ):
        value = backbone_cfg.get(key)
        if value is not None:
            return str(value)
    return None


def _device_from_pytest(request: pytest.FixtureRequest) -> torch.device:
    requested = request.config.getoption("--real-model-device")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        pytest.skip("--real-model-device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _configure_hf_access(request: pytest.FixtureRequest) -> str:
    """
    Configure network policy before importing the PAPR backbone factory.

    cache:
        Hugging Face is forced offline. Missing snapshots cause a clear test
        failure instead of an unexpected download.
    network:
        Network is permitted. Existing user-level environment variables are
        not overwritten unnecessarily.
    """
    mode = request.config.getoption("--hf-mode")
    if mode == "cache":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    return mode


def _deterministic_waveform(device: torch.device) -> torch.Tensor:
    t = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    waveform = 0.05 * torch.sin(2.0 * math.pi * 440.0 * t)
    return waveform.unsqueeze(0).to(device=device, dtype=torch.float32)


def _find_underlying_model(backbone: Any) -> Any:
    for name in ("model", "_model", "ssl_model", "backbone"):
        value = getattr(backbone, name, None)
        if value is not None and value is not backbone:
            return value
    raise AssertionError("Could not locate underlying SSL model.")


def _assert_common_output(
    output: Any,
    *,
    expected_layer: int,
    feature_dim: int,
    frame_count: int,
) -> None:
    assert hasattr(output, "features")
    assert hasattr(output, "frame_mask")
    assert hasattr(output, "hidden_layer")

    assert isinstance(output.features, torch.Tensor)
    assert isinstance(output.frame_mask, torch.Tensor)

    assert output.features.dtype == torch.float32
    assert output.frame_mask.dtype == torch.bool

    assert output.features.ndim == 3
    assert output.frame_mask.ndim == 2
    assert tuple(output.features.shape[:2]) == tuple(output.frame_mask.shape)

    assert output.features.shape == (1, frame_count, feature_dim)
    assert output.frame_mask.shape == (1, frame_count)
    assert int(output.hidden_layer) == expected_layer

    assert torch.isfinite(output.features).all()
    assert output.frame_mask.all()


@pytest.mark.real_model
@pytest.mark.parametrize("kind", ("wav2vec2", "wavlm", "w2v_bert2"))
def test_pinned_real_ssl_backbone_contract(
    kind: str,
    request: pytest.FixtureRequest,
) -> None:
    """
    One real, read-only forward per pinned backbone.

    This is a regression gate for P1:
      - config still names the intended checkpoint and immutable revision;
      - hidden-layer indexing remains structurally valid;
      - real [B,T,D] and frame_mask satisfy SSLBackboneOutput;
      - no silent fallback to a different backbone occurs.
    """
    _configure_hf_access(request)
    device = _device_from_pytest(request)

    spec = EXPECTED[kind]
    config_path = _project_root() / spec["config"]
    config = _load_yaml(config_path)
    backbone_cfg = _extract_backbone_cfg(config)

    assert _normalize_kind(backbone_cfg["kind"]) == _normalize_kind(kind)
    configured_model_name = _configured_model_name(backbone_cfg)
    assert configured_model_name is not None, "Backbone config has no model identifier"
    assert configured_model_name == spec["checkpoint"]
    assert backbone_cfg.get("revision") == spec["revision"]
    assert backbone_cfg.get("revision") != "main"
    assert len(str(backbone_cfg["revision"])) == 40
    assert int(backbone_cfg["hidden_layer"]) == spec["configured_hidden_layer"]

    # Import only after the requested Hugging Face access policy is installed.
    from papr_ssl.models.teacher.backbones.factory import create_ssl_backbone

    factory_kwargs = {k: v for k, v in backbone_cfg.items() if k != "kind"}
    backbone = create_ssl_backbone(backbone_cfg["kind"], **factory_kwargs)
    backbone.to(device)
    backbone.eval()

    underlying = _find_underlying_model(backbone)
    model_cfg = getattr(underlying, "config", None)
    assert model_cfg is not None

    num_hidden_layers = getattr(model_cfg, "num_hidden_layers", None)
    assert isinstance(num_hidden_layers, int)
    assert num_hidden_layers + 1 == spec["hidden_states_count"]
    assert getattr(model_cfg, "hidden_size", None) == spec["feature_dim"]

    waveform = _deterministic_waveform(device)
    with torch.inference_mode():
        output = backbone(waveform)

    _assert_common_output(
        output,
        expected_layer=spec["configured_hidden_layer"],
        feature_dim=spec["feature_dim"],
        frame_count=spec["frame_count"],
    )

    # Explicit kind guard: factory must not silently substitute another model.
    class_name = type(underlying).__name__.lower()
    if kind == "wav2vec2":
        assert "wav2vec2" in class_name and "bert" not in class_name
    elif kind == "wavlm":
        assert "wavlm" in class_name
    else:
        assert "wav2vec2bert" in class_name or "w2vbert" in class_name

    del output, waveform, underlying, backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.real_model
def test_real_model_configs_are_immutably_pinned() -> None:
    """
    Cheap metadata guard inside the explicit integration suite.

    This test performs no model forward, but keeps accidental `revision: main`
    regressions visible whenever P1-11 is intentionally invoked.
    """
    for kind, spec in EXPECTED.items():
        config = _load_yaml(_project_root() / spec["config"])
        backbone_cfg = _extract_backbone_cfg(config)

        assert _normalize_kind(backbone_cfg["kind"]) == _normalize_kind(kind)
        configured_model_name = _configured_model_name(backbone_cfg)
        assert configured_model_name is not None, "Backbone config has no model identifier"
        assert configured_model_name == spec["checkpoint"]
        assert backbone_cfg.get("revision") == spec["revision"]
        assert backbone_cfg.get("revision") != "main"
        assert len(str(backbone_cfg["revision"])) == 40
