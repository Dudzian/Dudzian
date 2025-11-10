"""Testy domyślnych ścieżek ustawień UI."""

import importlib
from pathlib import Path


def test_default_ui_settings_path_uses_dudzian_home(monkeypatch, tmp_path):
    module_name = "core.config.ui_settings"
    module = importlib.import_module(module_name)

    try:
        monkeypatch.setenv("DUDZIAN_HOME", str(tmp_path))
        module = importlib.reload(module)
        assert module.DEFAULT_UI_SETTINGS_PATH == tmp_path / "ui_settings.json"
    finally:
        monkeypatch.delenv("DUDZIAN_HOME", raising=False)
        module = importlib.reload(module)

    assert module.DEFAULT_UI_SETTINGS_PATH == Path.home() / ".dudzian" / "ui_settings.json"
