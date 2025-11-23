"""Ładowanie profili konfiguracyjnych dla klienta PySide6."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_profile_payload(payload: Mapping[str, Any], profile: str | None) -> Mapping[str, Any]:
    if not profile:
        return payload
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping):
        if profile in {"", "default"}:
            return payload
        raise KeyError(f"Profil UI '{profile}' nie jest zdefiniowany w pliku konfiguracyjnym")
    profile_payload = profiles.get(profile)
    if not isinstance(profile_payload, Mapping):
        raise KeyError(f"Profil UI '{profile}' nie jest prawidłowym słownikiem")
    return _merge_dicts(payload, profile_payload)


def _coerce_variant(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {k: _coerce_variant(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_variant(item) for item in value]
    return value


@dataclass(slots=True)
class UiAppConfig:
    """Struktura konfiguracji przekazywanej do PySide/QML."""

    source_path: Path
    profile: str
    payload: Mapping[str, Any]
    qml_entrypoint: Path
    decision_limit: int
    theme_palette: str

    def as_variant(self) -> dict[str, Any]:
        base: MutableMapping[str, Any] = dict(self.payload)
        base.setdefault("profile", self.profile)
        base.setdefault("source_path", self.source_path.as_posix())
        base.setdefault("qml_entrypoint", self.qml_entrypoint.as_posix())
        base.setdefault("decision_limit", self.decision_limit)
        base.setdefault("theme", {})
        theme_payload = dict(base["theme"])
        theme_payload.setdefault("palette", self.theme_palette)
        base["theme"] = theme_payload
        return _coerce_variant(base)


def load_ui_app_config(
    config_path: str | Path,
    *,
    profile: str | None = None,
    default_qml: str | Path | None = None,
) -> UiAppConfig:
    """Czyta plik YAML i przygotowuje dane do kontekstu QML."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji UI: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Plik {path} nie zawiera słownika YAML")
    merged_payload: Mapping[str, Any]
    if profile:
        selected_profile = profile
        merged_payload = _resolve_profile_payload(data, profile)
    else:
        default_profile = str(data.get("default_profile", "")).strip() or "default"
        selected_profile = default_profile
        profiles = data.get("profiles")
        if isinstance(profiles, Mapping) and default_profile in profiles:
            merged_payload = _resolve_profile_payload(data, default_profile)
        else:
            merged_payload = data
    pyside_payload = merged_payload.get("pyside")
    qml_override: Path | None = None
    if isinstance(pyside_payload, Mapping):
        entrypoint_value = pyside_payload.get("qml_entrypoint")
        if isinstance(entrypoint_value, str) and entrypoint_value.strip():
            qml_override = Path(entrypoint_value).expanduser().resolve()
    if default_qml is None:
        default_qml = Path(__file__).resolve().parent / "qml" / "MainWindow.qml"
    qml_path = qml_override or Path(default_qml).expanduser().resolve()
    theme_entry = merged_payload.get("theme")
    theme_payload: Mapping[str, Any] = theme_entry if isinstance(theme_entry, Mapping) else {}
    palette_name = str(theme_payload.get("palette", "dark"))
    decision_limit = int(merged_payload.get("history_limit", 30))
    return UiAppConfig(
        source_path=path,
        profile=selected_profile,
        payload=merged_payload,
        qml_entrypoint=qml_path,
        decision_limit=decision_limit,
        theme_palette=palette_name,
    )
