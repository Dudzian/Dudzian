from __future__ import annotations

import types

import pytest

import bot_core.ai.manager as manager_module
from bot_core.ai.manager import AIManager


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
    manager.require_real_models()
