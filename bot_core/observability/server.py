"""Lekki serwer HTTP eksponujący metryki w formacie Prometheusa."""

from __future__ import annotations

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Callable, Tuple, cast

from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

_LOGGER = logging.getLogger(__name__)

_DEFAULT_METRICS_PATH = "/metrics"


class _MetricsHandler(BaseHTTPRequestHandler):
    """Obsługuje zapytania HTTP zwracając metryki Prometheusa."""

    registry: MetricsRegistry
    path_matcher: Callable[[str], bool]

    # W środowisku CLI nie chcemy logów każdego requestu.
    def log_message(self, format: str, *args) -> None:  # noqa: A003 - BaseHTTPRequestHandler API
        _LOGGER.debug("METRICS %s - %s", self.client_address[0], format % args)

    def do_GET(self) -> None:  # noqa: D401,N802 - nazwa wymagana przez BaseHTTPRequestHandler
        """Obsługa zapytania GET."""

        server = cast(MetricsHTTPServer, self.server)

        if not server.matches_path(self.path):
            self.send_error(HTTPStatus.NOT_FOUND, "Nie znaleziono zasobu")
            return

        payload = server.render_metrics()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self) -> None:  # noqa: D401,N802 - API HTTPServer
        """Obsługa zapytania HEAD."""

        server = cast(MetricsHTTPServer, self.server)

        if not server.matches_path(self.path):
            self.send_error(HTTPStatus.NOT_FOUND, "Nie znaleziono zasobu")
            return

        payload_length = len(server.render_metrics())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(payload_length))
        self.end_headers()


class MetricsHTTPServer(ThreadingMixIn, HTTPServer):
    """Serwer HTTP obsługujący równoległe zapytania o metryki."""

    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        *,
        registry: MetricsRegistry | None = None,
        metrics_path: str = _DEFAULT_METRICS_PATH,
    ) -> None:
        self._registry = registry or get_global_metrics_registry()
        self._path_matcher = _build_path_matcher(metrics_path)

        super().__init__(server_address, _MetricsHandler)

    def matches_path(self, path: str) -> bool:
        return self._path_matcher(path)

    def render_metrics(self) -> bytes:
        return self._registry.render_prometheus().encode("utf-8")


def _build_path_matcher(metrics_path: str) -> Callable[[str], bool]:
    normalized = metrics_path.rstrip("/") or _DEFAULT_METRICS_PATH

    def matcher(path: str) -> bool:
        request_path = path.split("?", 1)[0]
        request_path = request_path.rstrip("/") or "/"
        return request_path == normalized

    return matcher


def start_http_server(
    port: int,
    host: str = "127.0.0.1",
    *,
    registry: MetricsRegistry | None = None,
    metrics_path: str = _DEFAULT_METRICS_PATH,
) -> Tuple[MetricsHTTPServer, threading.Thread]:
    """Uruchamia serwer metryk w tle i zwraca parę (server, thread)."""

    server = MetricsHTTPServer((host, port), registry=registry, metrics_path=metrics_path)
    thread = threading.Thread(target=server.serve_forever, name="metrics-http-server", daemon=True)
    thread.start()
    _LOGGER.info("Serwer metryk uruchomiony na %s:%s%s", host, server.server_address[1], metrics_path)
    return server, thread


__all__ = ["MetricsHTTPServer", "start_http_server"]

