"""Testy hermetyzujące rozwiązywanie ścieżek runtime."""

from pathlib import Path

from bot_core.runtime.paths import resolve_core_config_path


def test_resolve_core_config_path_accepts_injected_environ(tmp_path: Path) -> None:
    config_file = tmp_path / "core.yaml"
    environ = {"DUDZIAN_CORE_CONFIG": str(config_file)}

    resolved = resolve_core_config_path(environ=environ)

    assert resolved == config_file


def test_resolve_core_config_path_defaults_to_relative_path() -> None:
    resolved = resolve_core_config_path(environ={})

    assert resolved == Path("config/core.yaml")
