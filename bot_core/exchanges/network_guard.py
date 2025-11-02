"""Nadzorca egzekwujący politykę sieciową adapterów REST."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import socket
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from bot_core.alerts.base import AlertMessage, AlertRouter

_ALLOWED_METHODS = ("GET", "POST", "PUT", "DELETE")


@dataclass(slots=True)
class NetworkAccessViolation(RuntimeError):
    """Sygnał naruszenia polityki sieciowej adaptera REST."""

    reason: str
    details: Mapping[str, object]

    def __str__(self) -> str:  # pragma: no cover - delegacja do RuntimeError
        formatted = ", ".join(f"{key}={value!r}" for key, value in self.details.items())
        return f"{self.reason}: {formatted}" if formatted else self.reason


class NetworkAccessGuard:
    """Pilnuje, aby zapytania HTTP przestrzegały konfiguracji ``configure_network``."""

    __slots__ = (
        "_logger",
        "_alert_router",
        "_configured",
        "_ip_allowlist",
        "_proxy_snapshot",
    )

    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger
        self._alert_router: AlertRouter | None = None
        self._configured = False
        self._ip_allowlist: tuple[str, ...] = ()
        self._proxy_snapshot: tuple[tuple[str, str], ...] = ()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach_alert_router(self, alert_router: AlertRouter | None) -> None:
        """Umożliwia późniejsze podpięcie routera alertów."""

        self._alert_router = alert_router

    def configure(self, *, ip_allowlist: Sequence[str] | None) -> None:
        """Zapamiętuje allowlistę IP i aktualną konfigurację proxy."""

        if ip_allowlist:
            normalized = []
            for entry in ip_allowlist:
                value = str(entry).strip()
                if value:
                    normalized.append(value)
            self._ip_allowlist = tuple(normalized)
        else:
            self._ip_allowlist = ()
        self._proxy_snapshot = self._capture_proxy_config()
        self._configured = True

    def ensure_configured(self, *, url: str | None = None) -> None:
        """Sprawdza, czy strażnik został poprawnie skonfigurowany."""

        if not self._configured:
            self._raise_violation(
                "network_not_configured",
                url=url,
                message="configure_network() must be called before issuing HTTP requests",
            )

        current_proxies = self._capture_proxy_config()
        if current_proxies != self._proxy_snapshot:
            self._raise_violation(
                "proxy_configuration_changed",
                url=url,
                configured_proxies=self._format_proxies(self._proxy_snapshot),
                current_proxies=self._format_proxies(current_proxies),
            )

    def ensure_allowed(self, url: str, *, check_source_ip: bool = True) -> None:
        """Weryfikuje, czy zapytanie może zostać wysłane pod wskazany URL."""

        self.ensure_configured(url=url)

        if not check_source_ip or not self._ip_allowlist:
            return

        source_ip = self._determine_source_ip(url)
        if source_ip is None:
            self._raise_violation(
                "source_ip_unavailable",
                url=url,
                allowlist=self._ip_allowlist,
            )

        if source_ip not in self._ip_allowlist:
            self._raise_violation(
                "source_ip_not_allowed",
                url=url,
                allowlist=self._ip_allowlist,
                source_ip=source_ip,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _capture_proxy_config(self) -> tuple[tuple[str, str], ...]:
        entries: list[tuple[str, str]] = []
        for key in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            value = os.environ.get(key)
            if not value:
                value = os.environ.get(key.upper(), "")
            if not value:
                continue
            normalized = value.strip()
            if not normalized:
                continue
            entries.append((key.lower(), normalized))
        entries.sort()
        return tuple(entries)

    def _determine_source_ip(self, url: str) -> str | None:
        parts = urlsplit(url)
        host = parts.hostname
        if not host:
            return None
        port = parts.port
        if port is None:
            port = 443 if parts.scheme == "https" else 80

        try:
            infos = socket.getaddrinfo(host, port, 0, socket.SOCK_DGRAM)
        except OSError:
            return None

        for family, socktype, proto, _canon, sockaddr in infos:
            try:
                sock = socket.socket(family, socktype, proto)
            except OSError:
                continue
            try:
                sock.connect(sockaddr)
                local_ip = sock.getsockname()[0]
                return local_ip
            except OSError:
                continue
            finally:
                sock.close()
        return None

    def _format_proxies(self, snapshot: tuple[tuple[str, str], ...]) -> str:
        if not snapshot:
            return "<none>"
        return "; ".join(f"{key}={value}" for key, value in snapshot)

    def _raise_violation(self, reason: str, **details: object) -> None:
        payload = {key: value for key, value in details.items() if value is not None}
        payload.setdefault("allowlist", self._ip_allowlist)
        violation = NetworkAccessViolation(reason=reason, details=payload)
        self._logger.error(
            "Naruszenie allowlisty sieciowej: %s", violation, extra={"network_violation": payload}
        )
        self._dispatch_alert(violation)
        raise violation

    def _dispatch_alert(self, violation: NetworkAccessViolation) -> None:
        router = self._alert_router
        if router is None:
            return
        context = {
            "reason": violation.reason,
            "allowlist": ",".join(self._ip_allowlist) if self._ip_allowlist else "",
        }
        for key, value in violation.details.items():
            context[str(key)] = str(value)
        message = AlertMessage(
            category="security",
            title="Wykryto naruszenie allowlisty IP",
            body="Wstrzymano zapytanie HTTP z powodu naruszenia konfiguracji sieci.",
            severity="critical",
            context=context,
        )
        try:
            router.dispatch(message)
        except Exception:  # pragma: no cover - alert router nie może zablokować egzekucji
            self._logger.exception("Nie udało się dostarczyć alertu naruszenia allowlisty")


def normalize_relative_api_path(path: str) -> str:
    """Zwraca znormalizowaną ścieżkę API i blokuje nietypowe wartości."""

    if not isinstance(path, str):
        raise TypeError("Ścieżka endpointu API musi być napisem.")

    candidate = path.strip()
    if not candidate:
        raise ValueError("Ścieżka endpointu API nie może być pusta.")

    parts = urlsplit(candidate)
    if parts.scheme or parts.netloc:
        raise ValueError("Ścieżka endpointu API musi być względna wobec hosta adaptera.")
    if parts.query or parts.fragment:
        raise ValueError("Parametry zapytania muszą być przekazywane przez 'params', a nie w ścieżce.")

    normalized = parts.path
    if not normalized.startswith("/"):
        raise ValueError("Ścieżka endpointu API musi rozpoczynać się od '/'.")

    segments = [segment for segment in normalized.split("/") if segment]
    if any(segment in {"..", "."} for segment in segments):
        raise ValueError("Ścieżka endpointu API nie może zawierać sekwencji przejścia katalogów.")

    return normalized


def normalize_http_method(method: str, *, allowed: Sequence[str] | None = None) -> str:
    """Waliduje metodę HTTP i zwraca ją w wersji wielkimi literami."""

    allowed_methods = tuple(str(value).upper() for value in (allowed or _ALLOWED_METHODS))
    if not allowed_methods:
        raise ValueError("Wymagana jest co najmniej jedna dozwolona metoda HTTP.")

    normalized = str(method or "GET").strip().upper()
    if not normalized:
        normalized = "GET"

    if normalized not in allowed_methods:
        raise ValueError(f"Metoda HTTP '{normalized}' nie jest wspierana w tym adapterze.")

    return normalized


__all__ = [
    "NetworkAccessGuard",
    "NetworkAccessViolation",
    "normalize_relative_api_path",
    "normalize_http_method",
]

