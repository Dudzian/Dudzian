"""Pomocnicze narzędzia do obsługi ścieżek używanych przez runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DesktopAppPaths",
    "build_desktop_app_paths",
    "resolve_core_config_path",
]


def resolve_core_config_path(
    *,
    env_var: str = "DUDZIAN_CORE_CONFIG",
    default: str | Path = Path("config/core.yaml"),
) -> Path:
    """Zwraca ścieżkę do pliku konfiguracyjnego rdzenia."""

    candidate = os.environ.get(env_var, str(default))
    return Path(candidate).expanduser()


@dataclass(frozen=True)
class DesktopAppPaths:
    """Zbiór ścieżek wykorzystywanych przez aplikacje desktopowe."""

    app_root: Path
    logs_dir: Path
    text_log_file: Path
    db_file: Path
    open_positions_file: Path
    favorites_file: Path
    presets_dir: Path
    models_dir: Path
    keys_file: Path
    salt_file: Path


def build_desktop_app_paths(
    app_file: str | Path,
    *,
    logs_dir: Path | None = None,
    text_log_file: Path | None = None,
) -> DesktopAppPaths:
    """Buduje strukturę ścieżek dla aplikacji desktopowej."""

    app_root = Path(app_file).resolve().parent
    resolved_logs_dir = logs_dir or app_root / "logs"
    resolved_logs_dir.mkdir(parents=True, exist_ok=True)
    resolved_text_log = text_log_file or resolved_logs_dir / "trading.log"
    resolved_text_log.parent.mkdir(parents=True, exist_ok=True)

    presets_dir = app_root / "presets"
    presets_dir.mkdir(exist_ok=True)

    models_dir = app_root / "models"
    models_dir.mkdir(exist_ok=True)

    return DesktopAppPaths(
        app_root=app_root,
        logs_dir=resolved_logs_dir,
        text_log_file=resolved_text_log,
        db_file=app_root / "trading_bot.db",
        open_positions_file=app_root / "open_positions.json",
        favorites_file=app_root / "favorites.json",
        presets_dir=presets_dir,
        models_dir=models_dir,
        keys_file=app_root / "api_keys.enc",
        salt_file=app_root / "salt.bin",
    )

