from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime, timedelta, timezone
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

    def _raise_for_kryptolowca(name: str, package: str | None = None):
        if name == "KryptoLowca.ai_models":
            raise ModuleNotFoundError(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _raise_for_kryptolowca)
    monkeypatch.setitem(sys.modules, "ai_models", None)
    try:
        reloaded = importlib.reload(manager_module)

        error = reloaded._AI_IMPORT_ERROR  # type: ignore[attr-defined]
        assert error is not None
        if hasattr(error, "exceptions"):
            assert len(error.exceptions) == 2  # type: ignore[attr-defined]
            assert any(isinstance(exc, ModuleNotFoundError) for exc in error.exceptions)  # type: ignore[attr-defined]

        manager = reloaded.AIManager()
        assert manager.is_degraded is True
        assert manager.degradation_reason is not None
        assert manager.degradation_reason.startswith("fallback_ai_models")
        assert "ModuleNotFoundError" in manager.degradation_reason
        assert manager.degradation_details
        assert any("ModuleNotFoundError" in detail for detail in manager.degradation_details)
        assert manager.degradation_exception_types
        assert any(
            name.endswith("ModuleNotFoundError") for name in manager.degradation_exception_types
        )
        assert manager.degradation_exceptions
        assert any(
            isinstance(exc, ModuleNotFoundError) for exc in manager.degradation_exceptions
        )
        diagnostics = manager.degradation_exception_diagnostics
        assert diagnostics
        assert any(diag.type_name.endswith("ModuleNotFoundError") for diag in diagnostics)
        assert any("ai_models" in diag.formatted for diag in diagnostics)
        since = manager.degradation_since
        assert since is not None
        assert isinstance(since, datetime)
        assert since.tzinfo is not None
        assert since.tzinfo.utcoffset(since) == timezone.utc.utcoffset(since)
        duration = manager.degradation_duration
        assert duration is not None
        assert duration.total_seconds() >= 0.0
        history = manager.degradation_history
        assert history
        event = history[-1]
        assert event.reason == manager.degradation_reason
        assert event.details == manager.degradation_details
        assert event.exception_types == manager.degradation_exception_types
        assert event.exception_diagnostics == diagnostics
        assert event.started_at == since
        assert event.ended_at is None
        assert event.duration().total_seconds() >= 0.0
        event_dict = event.as_dict()
        assert event_dict["reason"] == event.reason
        assert event_dict["ended_at"] is None
        assert event_dict["duration_seconds"] >= 0.0
        if sys.version_info >= (3, 11):
            assert len(manager.degradation_details) >= 3
            assert "ai_models" in manager.degradation_details[1]
            assert "KryptoLowca.ai_models" in manager.degradation_details[2]
        status = manager.backend_status()
        assert status.degraded is True
        assert status.reason == manager.degradation_reason
        assert tuple(status.details) == manager.degradation_details
        assert tuple(status.exception_types) == manager.degradation_exception_types
        assert tuple(status.exception_diagnostics) == diagnostics
        assert status.since == since
        status_duration = status.duration()
        assert status_duration is not None
        assert status_duration.total_seconds() >= 0.0
        status_dict = status.as_dict()
        assert status_dict["exception_types"] == list(manager.degradation_exception_types)
        assert status_dict["exception_diagnostics"]
        assert any(
            item["type"].endswith("ModuleNotFoundError") for item in status_dict["exception_diagnostics"]
        )
        assert status_dict["since"] is not None
        assert status_dict["since"].endswith("+00:00")
        assert status_dict["duration_seconds"] is not None
        assert status_dict["duration_seconds"] >= 0.0
        with pytest.raises(RuntimeError) as excinfo:
            manager.require_real_models()
        message = str(excinfo.value)
        assert "ModuleNotFoundError" in message
        assert "[since=" in message
        assert "duration=" in message
        if sys.version_info >= (3, 11):
            assert "KryptoLowca.ai_models" in message
    finally:
        importlib.reload(manager_module)


def test_ai_manager_uses_kryptolowca_backend_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kryptolowca_pkg = types.ModuleType("KryptoLowca")
    kryptolowca_ai_models = types.ModuleType("KryptoLowca.ai_models")

    class _StubAIModels:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        @staticmethod
        def load_model(path: object) -> object:
            return object()

    kryptolowca_ai_models.AIModels = _StubAIModels  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "KryptoLowca", kryptolowca_pkg)
    monkeypatch.setitem(sys.modules, "KryptoLowca.ai_models", kryptolowca_ai_models)
    monkeypatch.setitem(sys.modules, "ai_models", None)

    try:
        reloaded_module = importlib.reload(manager_module)

        manager = reloaded_module.AIManager()
        assert manager.is_degraded is False
        assert manager.degradation_details == ()
        assert manager.degradation_exceptions == ()
        assert manager.degradation_exception_types == ()
        assert manager.degradation_exception_diagnostics == ()
        assert manager.degradation_since is None
        assert manager.degradation_duration is None
        status = manager.backend_status()
        assert status.degraded is False
        assert status.details == ()
        assert status.exception_types == ()
        assert status.exception_diagnostics == ()
        assert status.since is None
        assert status.duration() is None
        status_dict = status.as_dict()
        assert status_dict["since"] is None
        assert status_dict["duration_seconds"] is None
        manager._decision_inferences["stub"] = object()  # type: ignore[attr-defined]
        manager.require_real_models()
        assert manager.degradation_history == ()
    finally:
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
    assert manager.degradation_since is None
    assert manager.degradation_duration is None
    assert manager.degradation_history == ()
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
    since = manager.degradation_since
    assert since is not None
    assert isinstance(since, datetime)
    assert since.tzinfo is not None
    duration = manager.degradation_duration
    assert duration is not None
    assert duration.total_seconds() >= 0.0
    status = manager.backend_status()
    assert status.since == since
    status_duration = status.duration()
    assert status_duration is not None
    assert status_duration.total_seconds() >= 0.0
    status_dict = status.as_dict()
    assert status_dict["since"] is not None
    assert status_dict["duration_seconds"] is not None
    assert status_dict["duration_seconds"] >= 0.0


def test_ai_manager_clears_fallback_degradation_when_backend_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(manager_module, "_AI_IMPORT_ERROR", RuntimeError("missing backend"))
    manager = manager_module.AIManager()
    assert manager.is_degraded is True
    assert manager.degradation_reason is not None
    assert manager.degradation_reason.startswith("fallback_ai_models")
    assert manager.degradation_details
    assert any("RuntimeError" in detail for detail in manager.degradation_details)
    assert any(
        name.endswith("RuntimeError") for name in manager.degradation_exception_types
    )
    assert manager.degradation_exception_diagnostics
    assert manager.degradation_since is not None
    duration = manager.degradation_duration
    assert duration is not None
    assert duration.total_seconds() >= 0.0
    history = manager.degradation_history
    assert history
    fallback_event = history[-1]
    assert fallback_event.reason.startswith("fallback_ai_models") or fallback_event.reason == "backend_validation_failed"
    assert fallback_event.ended_at is None

    inference = DecisionModelInference(ModelRepository(tmp_path))
    inference._model = object()  # type: ignore[attr-defined]
    manager._mark_backend_ready(inference)
    assert manager.is_degraded is False
    assert manager.degradation_reason is None
    assert manager.degradation_details == ()
    assert manager.degradation_exception_types == ()
    assert manager.degradation_exceptions == ()
    assert manager.degradation_exception_diagnostics == ()
    assert manager.degradation_since is None
    assert manager.degradation_duration is None
    resolved_history = manager.degradation_history
    assert resolved_history
    last_event = resolved_history[-1]
    assert last_event.ended_at is not None
    assert last_event.duration().total_seconds() >= 0.0


def test_degradation_statistics_handles_empty_history() -> None:
    manager = manager_module.AIManager()

    stats = manager.degradation_statistics()

    assert stats.total_events == 0
    assert stats.active_events == 0
    assert stats.resolved_events == 0
    assert stats.total_downtime_seconds == 0.0
    assert stats.active_downtime_seconds == 0.0
    assert stats.resolved_downtime_seconds == 0.0
    assert stats.average_downtime_seconds is None
    assert stats.longest_downtime_seconds is None
    assert stats.shortest_downtime_seconds is None
    assert stats.by_reason == ()

    stats_dict = stats.as_dict()
    assert stats_dict["total_events"] == 0
    assert stats_dict["total_downtime_seconds"] == 0.0
    assert stats_dict["average_downtime_seconds"] is None
    assert stats_dict["by_reason"] == {}


def test_degradation_statistics_accumulates_history(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = manager_module.AIManager()

    first_start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    manager._activate_degradation("first", details=("initial",), since=first_start)

    first_end = first_start + timedelta(seconds=15)
    monkeypatch.setattr(manager_module, "_utcnow", lambda: first_end)
    manager._resolve_degradation()

    second_start = first_end + timedelta(seconds=5)
    manager._activate_degradation("second", details=("followup",), since=second_start)

    current_time = second_start + timedelta(seconds=20)
    monkeypatch.setattr(manager_module, "_utcnow", lambda: current_time)

    stats = manager.degradation_statistics()

    assert stats.total_events == 2
    assert stats.active_events == 1
    assert stats.resolved_events == 1
    assert stats.resolved_downtime_seconds == pytest.approx(15.0)
    assert stats.active_downtime_seconds == pytest.approx(20.0)
    assert stats.total_downtime_seconds == pytest.approx(35.0)
    assert stats.average_downtime_seconds == pytest.approx(17.5)
    assert stats.longest_downtime_seconds == pytest.approx(20.0)
    assert stats.shortest_downtime_seconds == pytest.approx(15.0)

    breakdown = {item.reason: item for item in stats.by_reason}
    assert set(breakdown) == {"first", "second"}

    first_stats = breakdown["first"]
    assert first_stats.total_events == 1
    assert first_stats.active_events == 0
    assert first_stats.resolved_events == 1
    assert first_stats.total_downtime_seconds == pytest.approx(15.0)
    assert first_stats.active_downtime_seconds == pytest.approx(0.0)
    assert first_stats.resolved_downtime_seconds == pytest.approx(15.0)
    assert first_stats.average_downtime_seconds == pytest.approx(15.0)
    assert first_stats.longest_downtime_seconds == pytest.approx(15.0)
    assert first_stats.shortest_downtime_seconds == pytest.approx(15.0)

    second_stats = breakdown["second"]
    assert second_stats.total_events == 1
    assert second_stats.active_events == 1
    assert second_stats.resolved_events == 0
    assert second_stats.total_downtime_seconds == pytest.approx(20.0)
    assert second_stats.active_downtime_seconds == pytest.approx(20.0)
    assert second_stats.resolved_downtime_seconds == pytest.approx(0.0)
    assert second_stats.average_downtime_seconds == pytest.approx(20.0)
    assert second_stats.longest_downtime_seconds == pytest.approx(20.0)
    assert second_stats.shortest_downtime_seconds == pytest.approx(20.0)

    stats_dict = stats.as_dict()
    assert stats_dict["active_events"] == 1
    assert stats_dict["active_downtime_seconds"] == pytest.approx(20.0)
    assert stats_dict["longest_downtime_seconds"] == pytest.approx(20.0)
    by_reason = stats_dict["by_reason"]
    assert set(by_reason) == {"first", "second"}
    assert by_reason["first"]["resolved_events"] == 1
    assert by_reason["second"]["active_downtime_seconds"] == pytest.approx(20.0)
