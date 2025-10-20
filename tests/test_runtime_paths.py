"""Testy pomocniczych narzędzi ścieżek runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.runtime.paths import (
    build_desktop_app_paths,
    build_desktop_app_paths_from_root,
    resolve_core_config_path,
)


def test_resolve_core_config_path_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUDZIAN_CORE_CONFIG", raising=False)
    assert resolve_core_config_path() == Path("config/core.yaml").expanduser()


def test_resolve_core_config_path_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "custom.yaml"
    monkeypatch.setenv("DUDZIAN_CORE_CONFIG", str(override))
    assert resolve_core_config_path() == override


def test_build_desktop_app_paths_creates_structure(tmp_path: Path) -> None:
    module_file = tmp_path / "pkg" / "trading_gui.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")

    logs_dir = tmp_path / "logs"
    text_log = logs_dir / "gui.log"

    paths = build_desktop_app_paths(module_file, logs_dir=logs_dir, text_log_file=text_log)

    assert paths.app_root == module_file.parent
    assert paths.logs_dir == logs_dir
    assert paths.text_log_file == text_log
    assert paths.db_file == module_file.parent / "trading_bot.db"
    assert not paths.open_positions_file.exists()
    assert paths.favorites_file.parent == module_file.parent
    assert paths.presets_dir.exists()
    assert paths.models_dir.exists()
    assert paths.keys_file == module_file.parent / "api_keys.enc"
    assert paths.salt_file == module_file.parent / "salt.bin"
    assert paths.secret_vault_file == module_file.parent / "api_keys.vault"

    # upewnij się, że katalog logów został utworzony
    assert logs_dir.exists()
    assert text_log.parent == logs_dir


def test_build_desktop_app_paths_from_root(tmp_path: Path) -> None:
    app_root = tmp_path / "desktop"
    paths = build_desktop_app_paths_from_root(app_root)

    assert paths.app_root == app_root
    assert paths.secret_vault_file == app_root / "api_keys.vault"
    assert paths.logs_dir == app_root / "logs"
    assert paths.presets_dir.exists()
