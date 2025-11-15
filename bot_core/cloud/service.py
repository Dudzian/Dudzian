"""Serwis udostępniający runtime gRPC w trybie cloud."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import grpc

from bot_core.api.server import LocalRuntimeContext, LocalRuntimeServer, build_local_runtime_context
from bot_core.cloud.config import CloudServerConfig
from bot_core.cloud.orchestrator import CloudOrchestrator
from bot_core.cloud.security import (
    CloudAuthInterceptor,
    CloudAuthServicer,
    CloudSecurityError,
    CloudSecurityManager,
)
from bot_core.generated import trading_pb2_grpc
from bot_core.security.license_service import LicenseService, LicenseServiceError


LOGGER = logging.getLogger(__name__)

GrpcRegistrar = Callable[[grpc.Server, LocalRuntimeContext], None]


class CloudRuntimeService:
    """Bootstraper serwera gRPC uruchamianego za flagą cloudową."""

    def __init__(
        self,
        config: CloudServerConfig,
        *,
        context_builder: Callable[[str | Path, str | None], LocalRuntimeContext] = build_local_runtime_context,
        license_service: LicenseService | None = None,
    ) -> None:
        self._config = config
        self._context_builder = context_builder
        self._license_service = license_service
        self._context: LocalRuntimeContext | None = None
        self._server: LocalRuntimeServer | None = None
        self._orchestrator: CloudOrchestrator | None = None
        self._address: str | None = None
        self._registrars: list[GrpcRegistrar] = []
        self._lock = threading.Lock()
        self._license_snapshot: object | None = None
        self._security_manager: CloudSecurityManager | None = None

    @property
    def address(self) -> str | None:
        return self._address

    @property
    def context(self) -> LocalRuntimeContext | None:
        return self._context

    def register_servicer(self, registrar: GrpcRegistrar) -> None:
        with self._lock:
            self._registrars.append(registrar)
            if self._server is not None:
                registrar(self._server.grpc_server, self._context)  # type: ignore[arg-type]

    @property
    def security_manager(self) -> CloudSecurityManager | None:
        return self._security_manager

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            runtime_cfg = self._config.runtime
            context = self._context_builder(
                config_path=runtime_cfg.config_path,
                entrypoint=runtime_cfg.entrypoint,
            )
            context.start()
            self._context = context
            self._maybe_attach_license()
            security_manager = self._ensure_security_manager()
            interceptors: list[grpc.ServerInterceptor] = []
            if security_manager and security_manager.requires_handshake:
                interceptors.append(CloudAuthInterceptor(security_manager))
            try:
                server = LocalRuntimeServer(
                    context,
                    host=self._config.host,
                    port=self._config.port,
                    max_workers=self._config.max_workers,
                    interceptors=interceptors or None,
                )
            except Exception:
                context.stop()
                self._context = None
                raise
            if security_manager:
                trading_pb2_grpc.add_CloudAuthServiceServicer_to_server(
                    CloudAuthServicer(security_manager),
                    server.grpc_server,
                )
            for registrar in self._registrars:
                registrar(server.grpc_server, context)
            server.start()
            marketplace_interval = (
                self._config.marketplace.refresh_interval_seconds
                if self._config.marketplace.auto_reload
                else 0
            )
            orchestrator = CloudOrchestrator(
                context,
                marketplace_refresh_interval=marketplace_interval,
            )
            orchestrator.start()
            self._server = server
            self._orchestrator = orchestrator
            self._address = server.address

    def stop(self) -> None:
        with self._lock:
            if self._server is None:
                return
            try:
                self._server.stop(0.5)
            except Exception:  # pragma: no cover - defensywne
                LOGGER.debug("Błąd podczas zatrzymywania CloudRuntimeServer", exc_info=True)
            if self._orchestrator is not None:
                self._orchestrator.stop()
            if self._context is not None:
                self._context.stop()
            self._server = None
            self._orchestrator = None
            self._context = None
            self._address = None

    def wait(self) -> None:
        server = self._server
        if server is None:
            return
        server.wait()

    # --- helpers -------------------------------------------------------------
    def _maybe_attach_license(self) -> None:
        cfg = self._config.license
        if not cfg.enabled or cfg.bundle_path is None:
            return
        if self._license_service is None:
            try:
                self._license_service = LicenseService()
            except LicenseServiceError as exc:
                LOGGER.error("Nie udało się zainicjalizować LicenseService: %s", exc)
                return
        try:
            self._license_snapshot = self._license_service.load_from_file(
                cfg.bundle_path,
                expected_hwid=cfg.expected_hwid,
            )
        except FileNotFoundError:
            LOGGER.error("Pakiet licencyjny nie istnieje: %s", cfg.bundle_path)
        except Exception:  # pragma: no cover - diagnostyka
            LOGGER.exception("Nie udało się zweryfikować licencji cloudowej")

    def _ensure_security_manager(self) -> CloudSecurityManager | None:
        security_cfg = self._config.security
        if not security_cfg.allowed_clients:
            return None
        if self._security_manager is None:
            try:
                self._security_manager = CloudSecurityManager(
                    security_cfg,
                    license_service=self._license_service,
                )
            except CloudSecurityError as exc:
                LOGGER.error("Nie udało się zainicjalizować CloudSecurityManager: %s", exc)
                raise
        else:
            self._security_manager.refresh_allowed_clients(security_cfg.allowed_clients)
        return self._security_manager


__all__ = ["CloudRuntimeService", "GrpcRegistrar"]
