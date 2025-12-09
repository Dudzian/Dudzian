"""Cienki adapter pomiędzy QML a warstwą gRPC backendu."""
from __future__ import annotations

from PySide6.QtCore import QObject, Property

from ui.backend.runtime_service import RuntimeService


class UiGrpcBridge(QObject):
    """Zapewnia pojedynczy punkt styku QML z serwisem gRPC RuntimeService."""

    def __init__(self, runtime_service: RuntimeService) -> None:
        super().__init__()
        self._runtime_service = runtime_service

    @Property(QObject, constant=True)
    def runtimeService(self) -> RuntimeService:
        """Udostępnia oryginalny RuntimeService do powiązań QML."""

        return self._runtime_service
