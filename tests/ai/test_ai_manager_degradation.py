from __future__ import annotations

import types
from pathlib import Path

import pandas as pd

import pytest

import bot_core.ai.manager as manager_module
from bot_core.ai.inference import DecisionModelInference, ModelRepository
from bot_core.ai.manager import AIManager
from bot_core.ai.pipeline import train_gradient_boosting_model


def test_ai_manager_flags_degradation_on_fallback() -> None:
    manager = AIManager()
    assert manager.is_degraded is True
    with pytest.raises(RuntimeError):
        manager.require_real_models()


def test_ai_manager_clears_degradation_with_real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RealModel:
        feature_names = ["foo"]

    def _load_model(path: object) -> _RealModel:
        return _RealModel()

    stub_backend = types.SimpleNamespace(load_model=_load_model)
    monkeypatch.setattr(manager_module, "_AI_IMPORT_ERROR", None)
    monkeypatch.setattr(manager_module, "_AIModels", stub_backend)
    manager = AIManager()
    assert manager.is_degraded is False
    manager._decision_inferences["stub"] = object()  # type: ignore[attr-defined]
    manager.require_real_models()


def test_ai_manager_degrades_on_quality_thresholds(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "f1": [float(i) for i in range(40)],
        "f2": [float(i % 4) for i in range(40)],
        "target": [float((i % 2) - 0.5) for i in range(40)],
    })
    artifact_path = train_gradient_boosting_model(
        frame,
        ["f1", "f2"],
        "target",
        output_dir=tmp_path,
        model_name="degraded",
        metadata={
            "quality_thresholds": {
                "min_directional_accuracy": 0.99,
                "max_mae": 0.01,
            }
        },
    )
    manager = AIManager()
    manager.configure_decision_repository(tmp_path)
    manager.load_decision_artifact("degraded", artifact_path, set_default=True)
    assert manager.is_degraded is True
    assert manager.degradation_reason is not None
    assert "quality" in manager.degradation_reason.lower()


def test_ai_manager_clears_fallback_degradation_when_backend_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(manager_module, "_AI_IMPORT_ERROR", RuntimeError("missing backend"))
    manager = AIManager()
    assert manager.is_degraded is True
    assert manager.degradation_reason == "fallback_ai_models"

    inference = DecisionModelInference(ModelRepository(tmp_path))
    inference._model = object()  # type: ignore[attr-defined]
    manager._mark_backend_ready(inference)
    assert manager.is_degraded is False
    assert manager.degradation_reason is None
