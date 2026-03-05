"""Wspólny store stanu runtime udostępniany warstwie QML."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Property, Signal

from ui.backend.runtime_service import RuntimeService

_LOGGER = logging.getLogger(__name__)


class UiRuntimeState(QObject):
    """Udostępnia w QML spójny widok stanu runtime oraz sygnały zmian."""

    cloudStatusChanged = Signal()
    handshakeChanged = Signal()
    feedHealthChanged = Signal()

    def __init__(self, runtime_service: RuntimeService, cloud_runtime_enabled: bool) -> None:
        super().__init__()
        self._runtime_service = runtime_service
        self._cloud_runtime_enabled = bool(cloud_runtime_enabled)
        self._cloud_status: dict[str, Any] = {}
        self._handshake: dict[str, Any] = {}
        self._feed_health: dict[str, Any] = {}

        self._refresh_cloud_status()
        self._refresh_feed_health()

        runtime_service.cloudRuntimeStatusChanged.connect(self._refresh_cloud_status)
        runtime_service.feedHealthChanged.connect(self._refresh_feed_health)

    @Property("QVariantMap", notify=cloudStatusChanged)
    def cloudStatus(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._cloud_status)

    @Property("QVariantMap", notify=handshakeChanged)
    def handshake(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._handshake)

    @Property(str, notify=handshakeChanged)
    def handshakeStatus(self) -> str:  # type: ignore[override]
        return str(self._handshake.get("status", "oczekuje"))

    @Property(bool, notify=handshakeChanged)
    def handshakeOk(self) -> bool:  # type: ignore[override]
        return self.handshakeStatus == "ok"

    @Property(str, notify=handshakeChanged)
    def handshakeLicenseId(self) -> str:  # type: ignore[override]
        return str(self._handshake.get("licenseId", ""))

    @Property(str, notify=handshakeChanged)
    def handshakeFingerprint(self) -> str:  # type: ignore[override]
        return str(self._handshake.get("fingerprint", ""))

    @Property(str, notify=cloudStatusChanged)
    def cloudTarget(self) -> str:  # type: ignore[override]
        return str(self._cloud_status.get("target", "client.yaml"))

    @Property(str, notify=cloudStatusChanged)
    def cloudStatusLabel(self) -> str:  # type: ignore[override]
        if not self._cloud_runtime_enabled:
            return "Cloud runtime: wyłączony"
        return f"Cloud: {self.cloudTarget} • handshake: {self.handshakeStatus}"

    @Property("QVariantMap", notify=feedHealthChanged)
    def feedHealth(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._feed_health)

    def _refresh_cloud_status(self) -> None:
        try:
            self._cloud_status = dict(self._runtime_service.cloudRuntimeStatus or {})
            self._handshake = dict(self._cloud_status.get("handshake") or {})
        except Exception:
            _LOGGER.exception("Nie udało się odczytać statusu runtime z RuntimeService")
            self._cloud_status = {}
            self._handshake = {}
        self.cloudStatusChanged.emit()
        self.handshakeChanged.emit()

    def _refresh_feed_health(self) -> None:
        try:
            self._feed_health = dict(self._runtime_service.feedHealth or {})
        except Exception:
            _LOGGER.exception("Nie udało się odczytać feedHealth z RuntimeService")
            self._feed_health = {}
        self.feedHealthChanged.emit()
