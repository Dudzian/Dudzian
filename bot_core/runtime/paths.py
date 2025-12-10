"""Pomocnicze narzędzia do obsługi ścieżek używanych przez runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = [
    "DesktopAppPaths",
    "RuntimePaths",
    "build_desktop_app_paths",
    "build_desktop_app_paths_from_root",
    "resolve_core_config_path",
]


def resolve_core_config_path(
    *,
    env_var: str = "DUDZIAN_CORE_CONFIG",
    default: str | Path = Path("config/core.yaml"),
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Zwraca ścieżkę do pliku konfiguracyjnego rdzenia.

    Parametr ``environ`` pozwala na wstrzyknięcie mapy zmiennych środowiskowych,
    co ułatwia hermetyzację testów bez modyfikowania globalnego ``os.environ``.
    """

    env = environ or os.environ
    candidate = env.get(env_var, str(default))
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
    secret_vault_file: Path


def _build_paths_for_root(
    app_root: Path,
    *,
    logs_dir: Path | None = None,
    text_log_file: Path | None = None,
) -> DesktopAppPaths:
    app_root = app_root.resolve()
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
        secret_vault_file=(app_root / "api_keys.vault"),
    )


def build_desktop_app_paths(
    app_file: str | Path,
    *,
    logs_dir: Path | None = None,
    text_log_file: Path | None = None,
) -> DesktopAppPaths:
    """Buduje strukturę ścieżek dla aplikacji desktopowej."""

    app_root = Path(app_file).resolve().parent
    return _build_paths_for_root(app_root, logs_dir=logs_dir, text_log_file=text_log_file)


def build_desktop_app_paths_from_root(
    app_root: str | Path,
    *,
    logs_dir: Path | None = None,
    text_log_file: Path | None = None,
) -> DesktopAppPaths:
    """Buduje strukturę ścieżek na podstawie katalogu aplikacji desktopowej."""

    return _build_paths_for_root(Path(app_root), logs_dir=logs_dir, text_log_file=text_log_file)

@dataclass(frozen=True)
class RuntimePaths:
    """Zestaw ścieżek używanych przez runtime do cache, presetów i dzienników."""

    data_cache_root: Path
    presets_dir: Path
    decisions_dir: Path

    @classmethod
    def from_environment(cls, environment: object) -> "RuntimePaths":
        """Buduje ścieżki runtime na podstawie konfiguracji środowiska.

        Obiekt ``environment`` jest traktowany duck-typingowo i powinien
        udostępniać pola ``data_cache_path`` oraz ``name``.
        """

        base_value = getattr(environment, "data_cache_path", None) or Path("var/data") / getattr(
            environment, "name", "default",
        )
        base_path = Path(str(base_value)).expanduser()

        def _resolve(candidate: str | Path | None, default: str) -> Path:
            if candidate is None:
                return base_path / default
            path = Path(str(candidate)).expanduser()
            return path if path.is_absolute() else base_path / path

        return cls(
            data_cache_root=base_path,
            presets_dir=_resolve(getattr(environment, "presets_dir", None), "presets"),
            decisions_dir=_resolve(getattr(environment, "decisions_dir", None), "decisions"),
        )

    def resolve_data_path(self, candidate: str | Path | None, *, default: str | Path) -> Path:
        """Zwraca ścieżkę w ``data_cache_root`` lub absolutną, jeśli podano pełną.

        Args:
            candidate: Konfigurowana ścieżka (może być względna lub absolutna).
            default: Nazwa pliku/katalogu używana, gdy ``candidate`` jest puste.
        """

        if candidate in (None, ""):
            return self.data_cache_root / default

        path = Path(str(candidate)).expanduser()
        return path if path.is_absolute() else self.data_cache_root / path
