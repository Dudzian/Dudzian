"""Testy domyślnej lokalizacji telemetrii anonimowej."""

import importlib
from pathlib import Path


def test_default_telemetry_dir_uses_dudzian_home(monkeypatch, tmp_path):
    module_name = "core.telemetry.anonymous_collector"
    module = importlib.import_module(module_name)

    try:
        monkeypatch.setenv("DUDZIAN_HOME", str(tmp_path))
        module = importlib.reload(module)
        assert module.DEFAULT_TELEMETRY_DIR == tmp_path / "telemetry"
    finally:
        monkeypatch.delenv("DUDZIAN_HOME", raising=False)
        module = importlib.reload(module)

    assert module.DEFAULT_TELEMETRY_DIR == Path.home() / ".dudzian" / "telemetry"
