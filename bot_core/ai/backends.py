"""Narzędzia pomocnicze do obsługi opcjonalnych backendów ML."""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Final, Mapping, Sequence

try:  # pragma: no cover - środowiska minimalne mogą nie mieć PyYAML
    import yaml
except Exception:  # pragma: no cover - fallback bez dodatkowych zależności
    yaml = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Wyjątek podnoszony, gdy wymagany backend ML nie jest dostępny."""

    def __init__(
        self,
        backend: str,
        module_name: str | None = None,
        *,
        install_hint: str | None = None,
    ) -> None:
        backend_name = backend.strip() or "(nieznany)"
        if module_name:
            base = (
                f"Backend '{backend_name}' wymaga modułu '{module_name}', który nie jest "
                "zainstalowany."
            )
        else:
            base = (
                f"Backend '{backend_name}' nie posiada zdefiniowanej zależności modułowej "
                "i jest obecnie niedostępny."
            )
        if install_hint:
            message = f"{base} Instalacja: {install_hint}."
        else:
            message = base
        super().__init__(message)
        self.backend: Final[str] = backend_name
        self.module_name: Final[str | None] = module_name
        self.install_hint: Final[str | None] = install_hint


_DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "config" / "ml" / "backends.yml"

# Domyślne mapowanie backendów na moduły importowe. Zostanie nadpisane przez
# konfigurację, jeżeli taka istnieje.
_DEFAULT_BACKEND_MODULES: Mapping[str, str] = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "pytorch": "torch",
    "torch": "torch",
    "sklearn": "sklearn",
    "scikit-learn": "sklearn",
}


@lru_cache(maxsize=None)
def _load_backend_config(config_path: Path | None = None) -> Mapping[str, object]:
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    if yaml is None:
        _LOGGER.warning(
            "Nie udało się załadować konfiguracji backendów ML – brak biblioteki PyYAML (%s)",
            path,
        )
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - logowanie błędów IO
        _LOGGER.error("Nie można odczytać konfiguracji backendów ML z %s: %s", path, exc)
        return {}
    if not isinstance(data, Mapping):
        return {}
    return data


def _resolve_backend_entry(
    backend: str,
    *,
    config_path: Path | None = None,
) -> tuple[str | None, Mapping[str, object]]:
    normalized = backend.strip().lower()
    config = _load_backend_config(config_path)
    entry: Mapping[str, object] | None = None
    backends = config.get("backends") if isinstance(config, Mapping) else None
    if isinstance(backends, Mapping):
        candidate = backends.get(normalized)
        if isinstance(candidate, Mapping):
            entry = candidate

    module_name = None
    if entry is not None:
        raw_module = entry.get("module")
        if isinstance(raw_module, str) and raw_module.strip():
            module_name = raw_module.strip()

    if module_name is None:
        module_name = _DEFAULT_BACKEND_MODULES.get(normalized)

    if entry is None:
        entry = {}
    return module_name, entry


@lru_cache(maxsize=None)
def _import_backend_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def is_backend_available(backend: str, *, config_path: Path | None = None) -> bool:
    """Sprawdza, czy wskazany backend ML ma dostępny moduł importowy."""

    module_name, entry = _resolve_backend_entry(backend, config_path=config_path)
    if module_name is None:
        # Backend typu "builtin" nie wymaga importu zewnętrznego.
        available_flag = entry.get("available", True) if isinstance(entry, Mapping) else True
        return bool(available_flag)
    try:
        _import_backend_module(module_name)
    except ModuleNotFoundError:
        return False
    return True


def require_backend(backend: str, *, config_path: Path | None = None) -> ModuleType:
    """Importuje backend ML lub zgłasza :class:`BackendUnavailableError`."""

    module_name, entry = _resolve_backend_entry(backend, config_path=config_path)
    if module_name is None:
        raise BackendUnavailableError(backend, module_name)
    install_hint = None
    if isinstance(entry, Mapping):
        raw_hint = entry.get("install_hint")
        if isinstance(raw_hint, str) and raw_hint.strip():
            install_hint = raw_hint.strip()
    try:
        return _import_backend_module(module_name)
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(backend, module_name, install_hint=install_hint) from exc


def get_backend_priority(*, config_path: Path | None = None) -> tuple[str, ...]:
    """Zwraca priorytetową kolejność backendów z konfiguracji."""

    config = _load_backend_config(config_path)
    raw_priority = config.get("priority") if isinstance(config, Mapping) else None
    if not isinstance(raw_priority, Sequence):
        return ()
    normalized: list[str] = []
    for entry in raw_priority:
        if isinstance(entry, str) and entry.strip():
            normalized.append(entry.strip().lower())
    return tuple(normalized)


def clear_backend_caches() -> None:
    """Czyści pamięć podręczną importów i konfiguracji (używane w testach)."""

    _load_backend_config.cache_clear()
    _import_backend_module.cache_clear()


__all__ = [
    "BackendUnavailableError",
    "clear_backend_caches",
    "get_backend_priority",
    "is_backend_available",
    "require_backend",
]

