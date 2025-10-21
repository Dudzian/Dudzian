"""Wspólne elementy dla wrapperów serwerów gRPC."""

from __future__ import annotations

from typing import Any, Optional


class GrpcServerLifecycleMixin:
    """Dzielona logika zatrzymywania i oczekiwania na serwer gRPC."""

    _server: Any

    def stop(self, grace: Optional[float] = None) -> None:
        self._server.stop(grace).wait()

    def wait_for_termination(self, timeout: Optional[float] = None) -> None:
        self._server.wait_for_termination(timeout)


__all__ = ["GrpcServerLifecycleMixin"]
