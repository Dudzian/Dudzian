from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pandas as pd

import pytest


if "lightgbm" not in sys.modules:
    class _StubLightGBMDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class _StubLightGBMBooster:
        def __init__(self, *args: object, model_str: str | None = None, **kwargs: object) -> None:
            self._model_str = model_str or "stub_model"
            self.best_score = {"train": {}}
            self.best_iteration = 0

        def model_to_string(self) -> str:
            return self._model_str

        def predict(self, matrix: object) -> list[float]:
            try:
                length = len(matrix)  # type: ignore[arg-type]
            except Exception:
                length = 0
            return [0.0 for _ in range(length)]

    def _stub_lightgbm_train(*args: object, **kwargs: object) -> _StubLightGBMBooster:
        return _StubLightGBMBooster()

    lightgbm_stub = types.ModuleType("lightgbm")
    lightgbm_stub.Dataset = _StubLightGBMDataset  # type: ignore[attr-defined]
    lightgbm_stub.Booster = _StubLightGBMBooster  # type: ignore[attr-defined]
    lightgbm_stub.train = _stub_lightgbm_train  # type: ignore[attr-defined]
    sys.modules["lightgbm"] = lightgbm_stub


import bot_core.ai.manager as manager_module
from bot_core.ai.inference import DecisionModelInference, ModelRepository
from bot_core.ai.pipeline import train_gradient_boosting_model


def test_ai_manager_flags_degradation_on_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def _raise_for_ai_models(name: str, package: str | None = None):
        if name == "ai_models":
            raise ModuleNotFoundError(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _raise_for_ai_models)
    monkeypatch.setitem(sys.modules, "ai_models", None)
    try:
        reloaded = importlib.reload(manager_module)

        error = reloaded._AI_IMPORT_ERROR  # type: ignore[attr-defined]
        assert error is not None
        if hasattr(error, "exceptions"):
            assert any(isinstance(exc, ModuleNotFoundError) for exc in error.exceptions)  # type: ignore[attr-defined]

        manager = reloaded.AIManager()
        assert manager.is_degraded is True
        assert manager.degradation_reason is not None
        assert manager.degradation_reason.startswith("fallback_ai_models")
        assert "ModuleNotFoundError" in manager.degradation_reason
        with pytest.raises(RuntimeError):
            manager.require_real_models()
    finally:
        importlib.reload(manager_module)


def test_ai_manager_uses_ai_models_backend_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ai_models_module = types.ModuleType("ai_models")

    class _StubAIModels:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        @staticmethod
        def load_model(path: object) -> object:
            return object()

    ai_models_module.AIModels = _StubAIModels  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ai_models", ai_models_module)

    try:
        reloaded_module = importlib.reload(manager_module)

        manager = reloaded_module.AIManager()
        assert manager.is_degraded is False
        manager._decision_inferences["stub"] = object()  # type: ignore[attr-defined]
        manager.require_real_models()
    finally:
        sys.modules.pop("ai_models", None)
        importlib.reload(manager_module)


def test_ai_manager_clears_degradation_with_real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RealModel:
        feature_names = ["foo"]

    def _load_model(path: object) -> _RealModel:
        return _RealModel()

    stub_backend = types.SimpleNamespace(load_model=_load_model)
    monkeypatch.setattr(manager_module, "_AI_IMPORT_ERROR", None)
    monkeypatch.setattr(manager_module, "_AIModels", stub_backend)
    manager = manager_module.AIManager()
    assert manager.is_degraded is False
    assert manager.degradation_details == ()
    assert manager.degradation_exceptions == ()
    assert manager.degradation_exception_types == ()
    assert manager.degradation_exception_diagnostics == ()
    manager._decision_inferences["stub"] = object()  # type: ignore[attr-defined]
    manager.require_real_models()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup dostępne od Pythona 3.11")
def test_collect_exception_messages_preserves_primary_order() -> None:
    primary = ImportError("primary failure")
    secondary = ModuleNotFoundError("secondary failure")
    bundled = manager_module._bundle_import_errors(primary, secondary)
    details = manager_module._collect_exception_messages(bundled)
    assert len(details) >= 3
    assert "primary failure" in details[1]
    assert "secondary failure" in details[2]


def test_collect_exception_types_respects_import_chain() -> None:
    primary = ImportError("primary failure")
    secondary = ModuleNotFoundError("secondary failure")
    bundled = manager_module._bundle_import_errors(primary, secondary)
    types = manager_module._collect_exception_types(bundled)
    assert types
    if sys.version_info >= (3, 11):
        assert types[0].endswith("ExceptionGroup")
        assert any(name.endswith("ModuleNotFoundError") for name in types)
    else:
        assert types[0].endswith("ModuleNotFoundError")
        assert len(types) >= 2
        assert types[1].endswith("ImportError")


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
    manager = manager_module.AIManager()
    manager.configure_decision_repository(tmp_path)
    manager.load_decision_artifact("degraded", artifact_path, set_default=True)
    assert manager.is_degraded is True
    assert manager.degradation_reason is not None
    assert "quality" in manager.degradation_reason.lower()
    assert any("quality" in detail.lower() for detail in manager.degradation_details)
    assert any("directional_accuracy" in detail for detail in manager.degradation_details)
    assert any("mae" in detail for detail in manager.degradation_details)
    assert manager.degradation_exception_types == ("builtins.RuntimeError",)
    assert len(manager.degradation_exceptions) == 1
    assert isinstance(manager.degradation_exceptions[0], RuntimeError)
    diagnostics = manager.degradation_exception_diagnostics
    assert diagnostics
    assert diagnostics[0].type_name == "builtins.RuntimeError"
    assert "quality" in diagnostics[0].formatted.lower()


def test_ai_manager_clears_fallback_degradation_when_backend_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(manager_module, "_AI_IMPORT_ERROR", RuntimeError("missing backend"))
    manager = manager_module.AIManager()
    assert manager.is_degraded is True
    assert manager.degradation_reason is not None
    assert manager.degradation_reason.startswith("fallback_ai_models")

    inference = DecisionModelInference(ModelRepository(tmp_path))
    inference._model = object()  # type: ignore[attr-defined]
    manager._mark_backend_ready(inference)
    assert manager.is_degraded is False
    assert manager.degradation_reason is None
    assert manager.degradation_details == ()
    assert manager.degradation_exception_types == ()
    assert manager.degradation_exceptions == ()
    assert manager.degradation_exception_diagnostics == ()


def test_ai_manager_detects_invalid_packaged_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _StubModels:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    packaged_root = tmp_path / "packaged"
    packaged_root.mkdir()
    manifest_payload = {"versions": {"1.0": {"file": "bundle/model.json"}}}
    (packaged_root / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bundle_dir = packaged_root / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "model.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(manager_module, "_AIModels", _StubModels)
    monkeypatch.setattr(manager_module, "_DefaultAIModels", _StubModels)
    monkeypatch.setattr(manager_module, "_AI_IMPORT_ERROR", None)
    monkeypatch.setattr(manager_module, "_FALLBACK_ACTIVE", False)
    monkeypatch.setattr(manager_module, "_PACKAGED_MODEL_REPOSITORY", packaged_root)

    manager = manager_module.AIManager(model_dir=tmp_path / "runtime")
    assert manager.is_degraded is True
    assert manager.degradation_reason is not None
    assert manager.degradation_reason.startswith("backend_validation_failed")
