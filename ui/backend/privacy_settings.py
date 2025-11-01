"""Kontroler QML odpowiedzialny za ustawienia prywatności i telemetrii."""
from __future__ import annotations

import json
from typing import Mapping

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.telemetry import AnonymousTelemetryCollector, TelemetryError


class PrivacySettingsController(QObject):
    """Umożliwia zarządzanie zgodą na anonimową telemetrię."""

    optInChanged = Signal()
    pseudonymChanged = Signal()
    queuedEventsChanged = Signal()
    previewChanged = Signal()
    lastExportChanged = Signal()

    def __init__(
        self,
        *,
        collector: AnonymousTelemetryCollector | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._collector = collector or AnonymousTelemetryCollector()
        self._preview_cache: list[Mapping[str, object]] = []

    # ------------------------------------------------------------------
    @Property(bool, notify=optInChanged)
    def optInEnabled(self) -> bool:  # type: ignore[override]
        return self._collector.enabled

    @Property(str, notify=pseudonymChanged)
    def pseudonym(self) -> str:  # type: ignore[override]
        return self._collector.pseudonym or ""

    @Property(str, constant=True)
    def installationId(self) -> str:  # type: ignore[override]
        return self._collector.installation_id

    @Property(int, notify=queuedEventsChanged)
    def queuedEvents(self) -> int:  # type: ignore[override]
        return int(self._collector.queued_events())

    @Property(str, notify=previewChanged)
    def previewJson(self) -> str:  # type: ignore[override]
        if not self._preview_cache:
            return "[]"
        return json.dumps(self._preview_cache, indent=2, ensure_ascii=False)

    @Property(str, constant=True)
    def queuePath(self) -> str:  # type: ignore[override]
        return str(self._collector.queue_path)

    @Property(str, notify=lastExportChanged)
    def lastExportAt(self) -> str:  # type: ignore[override]
        return self._collector.last_export_at or ""

    # ------------------------------------------------------------------
    @Slot(bool, str)
    def setOptIn(self, enabled: bool, fingerprint: str) -> None:
        try:
            self._collector.set_opt_in(bool(enabled), fingerprint or None)
        except TelemetryError:
            return
        self.refresh()

    @Slot(str)
    def refreshPseudonym(self, fingerprint: str) -> None:
        try:
            self._collector.refresh_pseudonym(fingerprint or None)
        except TelemetryError:
            return
        self.pseudonymChanged.emit()

    @Slot()
    def refresh(self) -> None:
        self._preview_cache = self._collector.preview_events()
        self.optInChanged.emit()
        self.pseudonymChanged.emit()
        self.queuedEventsChanged.emit()
        self.previewChanged.emit()
        self.lastExportChanged.emit()

    @Slot(result=str)
    def exportTelemetry(self) -> str:
        try:
            export_path = self._collector.export_events()
        except TelemetryError:
            return ""
        if not export_path:
            return ""
        self.refresh()
        return str(export_path)

    @Slot()
    def clearQueue(self) -> None:
        try:
            self._collector.clear_queue()
        except TelemetryError:
            return
        self.refresh()

    @Slot(str, str, str, str, result=str)
    def addPreviewEvent(
        self,
        event_type: str,
        key: str,
        value: str,
        fingerprint: str,
    ) -> str:
        payload: dict[str, object] = {}
        if key:
            payload[key] = value
        self._collector.collect_event(event_type or "preview", payload, fingerprint=fingerprint or None)
        self.refresh()
        return self.previewJson


__all__ = ["PrivacySettingsController"]
