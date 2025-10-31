"""Kontroler ustawień dashboardu dostępny z poziomu QML."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.config.ui_settings import (
    DashboardSettings,
    UISettings,
    UISettingsError,
    UISettingsStore,
)

_AVAILABLE_CARDS: tuple[str, ...] = ("io_queue", "guardrails", "retraining")


class DashboardSettingsController(QObject):
    """Zapewnia dostęp do ustawień dashboardu runtime."""

    cardOrderChanged = Signal()
    visibleCardOrderChanged = Signal()
    hiddenCardsChanged = Signal()
    refreshIntervalMsChanged = Signal()
    themeChanged = Signal()

    def __init__(
        self,
        *,
        store: UISettingsStore | None = None,
        available_cards: Sequence[str] = _AVAILABLE_CARDS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store or UISettingsStore()
        self._available_cards = tuple(dict.fromkeys(available_cards)) or _AVAILABLE_CARDS
        self._settings = self._load_settings()

    # ------------------------------------------------------------------
    @Property("QStringList", notify=cardOrderChanged)
    def cardOrder(self) -> list[str]:  # type: ignore[override]
        return list(self._settings.dashboard.normalized_order(self._available_cards))

    @Property("QStringList", notify=visibleCardOrderChanged)
    def visibleCardOrder(self) -> list[str]:  # type: ignore[override]
        return list(self._settings.dashboard.visible_order(self._available_cards))

    @Property("QStringList", notify=hiddenCardsChanged)
    def hiddenCards(self) -> list[str]:  # type: ignore[override]
        return list(self._settings.dashboard.hidden_cards)

    @Property("QStringList", constant=True)
    def availableCards(self) -> list[str]:  # type: ignore[override]
        return list(self._available_cards)

    @Property(int, notify=refreshIntervalMsChanged)
    def refreshIntervalMs(self) -> int:  # type: ignore[override]
        return int(self._settings.dashboard.refresh_interval_ms)

    @Property(str, notify=themeChanged)
    def theme(self) -> str:  # type: ignore[override]
        return self._settings.dashboard.theme

    @Property(str, constant=True)
    def settingsPath(self) -> str:  # type: ignore[override]
        return str(self._store.path)

    # ------------------------------------------------------------------
    @Slot("QStringList")
    def setCardOrder(self, order: list[str]) -> None:
        self._update_dashboard(self._settings.dashboard.with_updated_order(order))

    @Slot(str, bool)
    def setCardVisibility(self, card_id: str, visible: bool) -> None:
        card = str(card_id or "").strip()
        if not card or card not in self._available_cards:
            return
        hidden = set(self._settings.dashboard.hidden_cards)
        if visible:
            if card in hidden:
                hidden.remove(card)
        else:
            hidden.add(card)
        self._update_dashboard(self._settings.dashboard.with_hidden_cards(hidden))

    @Slot(str, int)
    def moveCard(self, card_id: str, offset: int) -> None:
        card = str(card_id or "").strip()
        if not card:
            return
        current = list(self._settings.dashboard.normalized_order(self._available_cards))
        if card not in current:
            return
        index = current.index(card)
        new_index = max(0, min(len(current) - 1, index + int(offset)))
        if new_index == index:
            return
        current.pop(index)
        current.insert(new_index, card)
        self._update_dashboard(self._settings.dashboard.with_updated_order(current))

    @Slot(int)
    def setRefreshIntervalMs(self, interval: int) -> None:
        self._update_dashboard(self._settings.dashboard.with_refresh_interval(interval))

    @Slot(str)
    def setTheme(self, theme: str) -> None:
        self._update_dashboard(self._settings.dashboard.with_theme(theme))

    @Slot()
    def resetDefaults(self) -> None:
        defaults = DashboardSettings()
        updated = replace(self._settings, dashboard=defaults)
        self._apply_settings(updated)

    # ------------------------------------------------------------------
    def _load_settings(self) -> UISettings:
        try:
            settings = self._store.load()
        except UISettingsError:
            settings = UISettings()
        normalized = settings.dashboard.normalized_order(self._available_cards)
        if normalized != settings.dashboard.card_order:
            settings = replace(
                settings,
                dashboard=settings.dashboard.with_updated_order(normalized),
            )
            self._persist(settings)
        return settings

    def _update_dashboard(self, dashboard: DashboardSettings) -> None:
        if dashboard == self._settings.dashboard:
            return
        updated = replace(self._settings, dashboard=dashboard)
        self._apply_settings(updated)

    def _apply_settings(self, settings: UISettings) -> None:
        self._settings = settings
        self._persist(settings)
        self.cardOrderChanged.emit()
        self.visibleCardOrderChanged.emit()
        self.hiddenCardsChanged.emit()
        self.refreshIntervalMsChanged.emit()
        self.themeChanged.emit()

    def _persist(self, settings: UISettings) -> None:
        try:
            self._store.save(settings)
        except UISettingsError:
            # W przypadku błędu zapisu pozostawiamy ustawienia w pamięci.
            pass


__all__ = ["DashboardSettingsController"]
