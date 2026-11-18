"""Wspólne elementy dla wrapperów serwerów gRPC."""

from __future__ import annotations

import ipaddress
from typing import Any, Optional


def validate_plaintext_grpc_host(host: object) -> str:
    """Keep unencrypted listeners on explicit, canonical loopback addresses."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Plaintext gRPC requires an explicit loopback address.")
    normalized = host.strip().lower()
    if normalized == "localhost":
        return "127.0.0.1"
    address_text = (
        normalized[1:-1] if normalized.startswith("[") and normalized.endswith("]") else normalized
    )
    if "%" in address_text:
        raise ValueError("Plaintext gRPC requires an explicit loopback address.")
    try:
        address = ipaddress.ip_address(address_text)
    except ValueError as exc:
        raise ValueError("Plaintext gRPC requires an explicit loopback address.") from exc
    # Keep mapped IPv4 refusal consistent across Python's ipaddress versions.
    if not address.is_loopback or (
        isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None
    ):
        raise ValueError("Plaintext gRPC requires an explicit loopback address.")
    return f"[{address.compressed}]" if address.version == 6 else str(address)


class GrpcServerLifecycleMixin:
    """Dzielona logika zatrzymywania i oczekiwania na serwer gRPC."""

    _server: Any

    def stop(self, grace: Optional[float] = None) -> None:
        self._server.stop(grace).wait()

    def wait_for_termination(self, timeout: Optional[float] = None) -> None:
        self._server.wait_for_termination(timeout)


__all__ = ["GrpcServerLifecycleMixin", "validate_plaintext_grpc_host"]
