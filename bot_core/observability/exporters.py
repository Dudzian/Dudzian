"""Lokalni eksporterzy metryk wspierający uruchomienie offline."""
from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass
from typing import Tuple

from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
from bot_core.observability.server import MetricsHTTPServer, start_http_server

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PrometheusExporterConfig:
    """Konfiguracja lokalnego eksportera Prometheusa."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 0
    metrics_path: str = "/metrics"


class LocalPrometheusExporter:
    """Odpowiada za uruchomienie prostego serwera HTTP z metrykami."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        metrics_path: str = "/metrics",
        registry: MetricsRegistry | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._metrics_path = metrics_path
        self._registry = registry or get_global_metrics_registry()
        self._server: MetricsHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> Tuple[str, int] | None:
        if self._server is None:
            return None
        return self._server.server_address  # type: ignore[return-value]

    @property
    def metrics_url(self) -> str | None:
        if self._server is None:
            return None
        host, port = self._server.server_address
        path = self._metrics_path or "/metrics"
        if not path.startswith("/"):
            path = "/" + path
        return f"http://{host}:{port}{path}"

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            server, thread = start_http_server(
                self._port,
                self._host,
                registry=self._registry,
                metrics_path=self._metrics_path,
            )
        except OSError as exc:  # pragma: no cover - brak zasobów
            _LOGGER.error("Nie udało się uruchomić eksportera Prometheus: %s", exc)
            raise
        self._server = server
        self._thread = thread
        # Upewnij się, że raportujemy faktyczny port w przypadku wartości 0.
        if self._port == 0 and isinstance(server.server_address, tuple):
            self._port = int(server.server_address[1])
        _LOGGER.info(
            "Eksporter Prometheus dostępny pod adresem %s:%s%s",
            self._host,
            self._port,
            self._metrics_path,
        )

    def stop(self, timeout: float = 1.0) -> None:
        server = self._server
        if server is None:
            return
        thread = self._thread
        try:
            server.shutdown()
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Błąd podczas zamykania eksportera Prometheus", exc_info=True)
        finally:
            try:
                server.server_close()
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się zamknąć gniazda serwera metryk", exc_info=True)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._server = None
        self._thread = None

    def __enter__(self) -> "LocalPrometheusExporter":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.stop()


def test_port_available(host: str, port: int) -> bool:
    """Sprawdza czy port może zostać otwarty dla eksportera."""

    if port == 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


__all__ = [
    "LocalPrometheusExporter",
    "PrometheusExporterConfig",
    "test_port_available",
]

