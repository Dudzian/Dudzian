# -*- coding: utf-8 -*-
"""Konfiguracja logowania dla całej aplikacji KryptoLowca.

Nowa wersja obsługuje:
- logi tekstowe lub JSON (przez ``python-json-logger``),
- zapisywanie do plików rotowanych oraz STDOUT (niezbędne w kontenerach),
- opcjonalne przesyłanie zdarzeń do kolektora Vector/ELK poprzez HTTP.

Dzięki kolejce ``QueueHandler`` logowanie nie blokuje głównej pętli bota,
co jest kluczowe przy dużej liczbie sygnałów oraz podczas pracy 24/7.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sys
import threading
import urllib.request
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

try:  # pragma: no cover - opcjonalne w środowisku CI
    from pythonjsonlogger import jsonlogger
except Exception:  # pragma: no cover
    jsonlogger = None  # type: ignore

_APP_ROOT = Path(__file__).resolve().parent
LOGS_DIR = Path(os.getenv("KRYPT_LOWCA_LOG_DIR", _APP_ROOT / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "trading.log"

_QUEUE: Optional[queue.Queue[logging.LogRecord]] = None
_LISTENER: Optional[QueueListener] = None
_QUEUE_LOCK = threading.Lock()


class VectorHttpHandler(logging.Handler):
    """Minimalny handler wysyłający logi JSON do kolektora Vector/ELK."""

    def __init__(self, endpoint: str, timeout: float = 2.0) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - zależne od sieci
        try:
            payload = self.format(record)
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception:
            self.handleError(record)


def _build_formatter(format_type: str, service_name: str) -> logging.Formatter:
    format_type = format_type.lower()
    base_fields = {
        "service": service_name,
        "schema_version": "1.0",
    }
    if format_type == "json" and jsonlogger is not None:
        class _JsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):  # type: ignore[override]
                super().add_fields(log_record, record, message_dict)
                log_record.setdefault("service", service_name)
                log_record.setdefault("schema_version", "1.0")
                log_record.setdefault("level", record.levelname)
                log_record.setdefault("logger", record.name)
                if record.exc_info:
                    log_record.setdefault("exc_info", self.formatException(record.exc_info))
        return _JsonFormatter("%(message)s")

    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)
    formatter.defaults = getattr(formatter, "defaults", {})  # type: ignore[attr-defined]
    formatter.defaults.update(base_fields)  # type: ignore[attr-defined]
    return formatter


def _ensure_queue_listener(handlers: Iterable[logging.Handler]) -> QueueHandler:
    global _QUEUE, _LISTENER
    with _QUEUE_LOCK:
        if _QUEUE is None:
            _QUEUE = queue.Queue()
            _LISTENER = QueueListener(_QUEUE, *handlers, respect_handler_level=True)
            _LISTENER.start()
atexit.register(lambda: (_LISTENER is not None) and (_LISTENER.stop() is None))
assert _QUEUE is not None
        return QueueHandler(_QUEUE)


def setup_app_logging(
    *,
    log_file: Path | str = DEFAULT_LOG_FILE,
    level: int | str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 10,
    service_name: str = "kryptolowca",
) -> logging.Logger:
    """Skonfiguruj główny logger aplikacji.

    Parametry mogą być nadpisane przez zmienne środowiskowe:
    - ``KRYPT_LOWCA_LOG_FORMAT``: ``"json"`` lub ``"text"`` (domyślnie ``json`` w kontenerze),
    - ``KRYPT_LOWCA_LOG_LEVEL``: poziom logowania,
    - ``KRYPT_LOWCA_LOG_SHIP_VECTOR``: adres HTTP kolektora Vector.
    """

    root = logging.getLogger("KryptoLowca")
    if getattr(root, "_krypto_logging_configured", False):
        return root

    env_level = os.getenv("KRYPT_LOWCA_LOG_LEVEL")
    resolved_level = level or env_level or "INFO"
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

    format_type = os.getenv("KRYPT_LOWCA_LOG_FORMAT", "json")
    formatter = _build_formatter(format_type, service_name)

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [file_handler, stream_handler]

    vector_endpoint = os.getenv("KRYPT_LOWCA_LOG_SHIP_VECTOR")
    if vector_endpoint:
        vector_handler = VectorHttpHandler(vector_endpoint)
        vector_handler.setFormatter(formatter)
        handlers.append(vector_handler)

    queue_handler = _ensure_queue_listener(handlers)
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(resolved_level)
    root.propagate = False
    setattr(root, "_krypto_logging_configured", True)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    setup_app_logging()
    return logging.getLogger(name if name else "KryptoLowca")


__all__ = [
    "LOGS_DIR",
    "DEFAULT_LOG_FILE",
    "setup_app_logging",
    "get_logger",
]



