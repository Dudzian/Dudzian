"""Application-level logging setup for bot_core runtime and tools."""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sys
import threading
import urllib.request
from contextlib import suppress
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency in minimal installations
    from pythonjsonlogger import jsonlogger
except Exception:  # pragma: no cover - json logging is optional in tests/CI
    jsonlogger = None  # type: ignore


_LEGACY_ENV_HINT = (
    "Zmienna środowiskowa {legacy} została wycofana. Ustaw {current} i usuń pozostałości "
    "po pakiecie KryptoLowca (patrz docs/migrations/kryptolowca_namespace_mapping.md)."
)


def _env(name: str, *, legacy: str | None = None) -> str | None:
    """Read environment variable ensuring legacy prefixes are not used."""

    if legacy and legacy in os.environ:
        raise RuntimeError(_LEGACY_ENV_HINT.format(legacy=legacy, current=name))
    return os.getenv(name)


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = Path(
    _env("BOT_CORE_LOG_DIR", legacy="KRYPT_LOWCA_LOG_DIR") or (_PACKAGE_ROOT / "logs")
)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = Path(
    _env("BOT_CORE_LOG_FILE", legacy="KRYPT_LOWCA_LOG_FILE") or (LOGS_DIR / "trading.log")
)

_QUEUE: Optional[queue.Queue[logging.LogRecord]] = None
_LISTENER: Optional[QueueListener] = None
_QUEUE_LOCK = threading.Lock()
_ATEEXIT_REGISTERED = False


def _stop_queue_listener() -> None:
    """Safely stop the background QueueListener (used in tests/atexit)."""

    global _QUEUE, _LISTENER, _ATEEXIT_REGISTERED

    if _ATEEXIT_REGISTERED:
        with suppress(Exception):
            atexit.unregister(_stop_queue_listener)
        _ATEEXIT_REGISTERED = False

    listener = _LISTENER
    if listener is None:
        return

    thread = getattr(listener, "_thread", None)
    if thread is None:
        _QUEUE = None
        _LISTENER = None
        return

    with suppress(Exception):
        listener.stop()
    _QUEUE = None
    _LISTENER = None


class VectorHttpHandler(logging.Handler):
    """Minimal HTTP handler that ships JSON logs to observability pipelines."""

    def __init__(self, endpoint: str, timeout: float = 2.0) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - network side effects
        try:
            payload = self.format(record)
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            request = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=self.timeout).read()
        except Exception:
            self.handleError(record)


def _build_formatter(format_type: str, service_name: str) -> logging.Formatter:
    format_type = (format_type or "json").lower()
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

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    formatter.defaults = getattr(formatter, "defaults", {})  # type: ignore[attr-defined]
    formatter.defaults.update(base_fields)  # type: ignore[attr-defined]
    return formatter


def _ensure_queue_listener(handlers: Iterable[logging.Handler]) -> QueueHandler:
    global _QUEUE, _LISTENER, _ATEEXIT_REGISTERED
    with _QUEUE_LOCK:
        if _QUEUE is None:
            _QUEUE = queue.Queue()
            _LISTENER = QueueListener(_QUEUE, *handlers, respect_handler_level=True)
            _LISTENER.start()
            if not _ATEEXIT_REGISTERED:
                atexit.register(_stop_queue_listener)
                _ATEEXIT_REGISTERED = True
        assert _QUEUE is not None
        return QueueHandler(_QUEUE)


def setup_app_logging(
    *,
    log_file: Path | str = DEFAULT_LOG_FILE,
    level: int | str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 10,
    service_name: str = "bot_core",
) -> logging.Logger:
    """Configure the primary ``bot_core`` logger with queue-based handlers."""

    logger_name = _env("BOT_CORE_LOGGER_NAME", legacy="KRYPT_LOWCA_LOGGER_NAME") or "bot_core"
    root = logging.getLogger(logger_name)
    if getattr(root, "_bot_core_logging_configured", False):
        return root

    env_level = _env("BOT_CORE_LOG_LEVEL", legacy="KRYPT_LOWCA_LOG_LEVEL")
    resolved_level = level or env_level or "INFO"
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

    format_type = _env("BOT_CORE_LOG_FORMAT", legacy="KRYPT_LOWCA_LOG_FORMAT") or "json"
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

    vector_endpoint = _env(
        "BOT_CORE_LOG_SHIP_VECTOR", legacy="KRYPT_LOWCA_LOG_SHIP_VECTOR"
    )
    if vector_endpoint:
        vector_handler = VectorHttpHandler(vector_endpoint)
        vector_handler.setFormatter(formatter)
        handlers.append(vector_handler)

    queue_handler = _ensure_queue_listener(handlers)
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(resolved_level)
    root.propagate = False
    setattr(root, "_bot_core_logging_configured", True)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger, installing the app logging configuration if needed."""

    setup_app_logging()
    target = (
        name
        or _env("BOT_CORE_LOGGER_NAME", legacy="KRYPT_LOWCA_LOGGER_NAME")
        or "bot_core"
    )
    return logging.getLogger(target if name is None else name)


__all__ = ["LOGS_DIR", "DEFAULT_LOG_FILE", "setup_app_logging", "get_logger", "VectorHttpHandler"]
