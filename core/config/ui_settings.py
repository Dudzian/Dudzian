"""Model oraz magazyn ustawień interfejsu użytkownika."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "DEFAULT_UI_SETTINGS_PATH",
    "DashboardSettings",
    "UISettings",
    "UISettingsError",
    "UISettingsStore",
]

def _default_ui_settings_path() -> Path:
    base_override = os.environ.get("DUDZIAN_HOME")
    if base_override:
        return (Path(base_override).expanduser() / "ui_settings.json").expanduser()
    return (Path.home() / ".dudzian" / "ui_settings.json").expanduser()


DEFAULT_UI_SETTINGS_PATH = _default_ui_settings_path()


class UISettingsError(RuntimeError):
    """Błąd związany z odczytem lub zapisem ustawień UI."""


@dataclass(slots=True)
class DashboardSettings:
    """Preferencje widoku dashboardu runtime."""

    card_order: tuple[str, ...] = (
        "io_queue",
        "guardrails",
        "retraining",
        "compliance",
        "ai_decisions",
    )
    hidden_cards: tuple[str, ...] = ()
    refresh_interval_ms: int = 4000
    theme: str = "system"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "DashboardSettings":
        if not mapping:
            return cls()
        card_order = tuple(str(item) for item in mapping.get("card_order", cls.card_order))
        hidden_cards = tuple(str(item) for item in mapping.get("hidden_cards", ()))
        refresh_interval = int(mapping.get("refresh_interval_ms", cls.refresh_interval_ms))
        refresh_interval = max(500, refresh_interval)
        theme = str(mapping.get("theme", cls.theme) or cls.theme)
        return cls(
            card_order=card_order,
            hidden_cards=hidden_cards,
            refresh_interval_ms=refresh_interval,
            theme=theme,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "card_order": list(self.card_order),
            "hidden_cards": list(self.hidden_cards),
            "refresh_interval_ms": int(self.refresh_interval_ms),
            "theme": self.theme,
        }

    def normalized_order(self, available: Sequence[str]) -> tuple[str, ...]:
        known = [item for item in self.card_order if item in available]
        remainder = [item for item in available if item not in known]
        return tuple(dict.fromkeys((*known, *remainder)))

    def visible_order(self, available: Sequence[str]) -> tuple[str, ...]:
        hidden = set(self.hidden_cards)
        return tuple(item for item in self.normalized_order(available) if item not in hidden)

    def with_updated_order(self, order: Iterable[str]) -> "DashboardSettings":
        normalized = tuple(dict.fromkeys(str(item) for item in order if str(item)))
        if not normalized:
            normalized = self.card_order
        return DashboardSettings(
            card_order=normalized,
            hidden_cards=self.hidden_cards,
            refresh_interval_ms=self.refresh_interval_ms,
            theme=self.theme,
        )

    def with_hidden_cards(self, hidden: Iterable[str]) -> "DashboardSettings":
        hidden_cards = tuple(dict.fromkeys(str(item) for item in hidden if str(item)))
        return DashboardSettings(
            card_order=self.card_order,
            hidden_cards=hidden_cards,
            refresh_interval_ms=self.refresh_interval_ms,
            theme=self.theme,
        )

    def with_refresh_interval(self, interval_ms: int) -> "DashboardSettings":
        interval = max(500, int(interval_ms))
        return DashboardSettings(
            card_order=self.card_order,
            hidden_cards=self.hidden_cards,
            refresh_interval_ms=interval,
            theme=self.theme,
        )

    def with_theme(self, theme: str) -> "DashboardSettings":
        normalized = theme.strip().lower() or self.theme
        return DashboardSettings(
            card_order=self.card_order,
            hidden_cards=self.hidden_cards,
            refresh_interval_ms=self.refresh_interval_ms,
            theme=normalized,
        )


@dataclass(slots=True)
class UISettings:
    """Zbiorczy model ustawień interfejsu."""

    version: int = 1
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "UISettings":
        if not mapping:
            return cls()
        version = int(mapping.get("version", 1))
        dashboard = DashboardSettings.from_mapping(
            mapping.get("dashboard") if isinstance(mapping, Mapping) else None
        )
        return cls(version=version, dashboard=dashboard)

    def to_mapping(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "dashboard": self.dashboard.to_mapping(),
        }


class UISettingsStore:
    """Odpowiada za trwały zapis ustawień interfejsu."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or DEFAULT_UI_SETTINGS_PATH).expanduser()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UISettings:
        try:
            content = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return UISettings()
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise UISettingsError(f"Nie można odczytać ustawień UI: {exc}") from exc
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise UISettingsError(f"Uszkodzony plik ustawień UI: {exc}") from exc
        if not isinstance(payload, MutableMapping):
            raise UISettingsError("Plik ustawień UI musi zawierać obiekt JSON")
        return UISettings.from_mapping(payload)

    def save(self, settings: UISettings) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(settings.to_mapping(), indent=2, ensure_ascii=False)
            self._path.write_text(serialized + "\n", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise UISettingsError(f"Nie można zapisać ustawień UI: {exc}") from exc
