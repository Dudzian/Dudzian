"""Konfiguracja logowania integrująca metryki i alerty offline."""
from __future__ import annotations

import logging
import threading
from typing import Iterable

from bot_core.observability.metrics import (
    MetricsRegistry,
    get_global_metrics_registry,
    observe_exchange_log_record,
    observe_security_log_record,
    observe_strategy_log_record,
)

_LOGGER = logging.getLogger(__name__)
_HANDLER_LOCK = threading.Lock()
_HANDLER_TOKEN: logging.Handler | None = None


class MetricsLoggingHandler(logging.Handler):
    """Przechwytuje rekordy logowania i aktualizuje metryki domenowe."""

    def __init__(self, *, registry: MetricsRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry or get_global_metrics_registry()

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        """Aktualizuje metryki w oparciu o rekord logowania."""

        try:
            observe_exchange_log_record(record, registry=self._registry)
            observe_strategy_log_record(record, registry=self._registry)
            observe_security_log_record(record, registry=self._registry)
        except Exception:  # pragma: no cover - metryki nie powinny przerywać logowania
            _LOGGER.debug("Nie udało się zarejestrować metryk z rekordu logowania", exc_info=True)


def install_metrics_logging_handler(
    *,
    registry: MetricsRegistry | None = None,
    logger_names: Iterable[str] | None = None,
) -> logging.Handler:
    """Instaluje globalny handler metryczny na wskazanych loggerach."""

    global _HANDLER_TOKEN
    with _HANDLER_LOCK:
        if _HANDLER_TOKEN is not None:
            return _HANDLER_TOKEN

        handler = MetricsLoggingHandler(registry=registry)
        handler.setLevel(logging.DEBUG)

        targets = list(logger_names or ()) or ["bot_core", "bot"]
        # Zapewniamy obsługę również dla root loggera, jeżeli nie określono celów.
        root_logger = logging.getLogger()
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)
        for name in targets:
            if not name:
                continue
            logger = logging.getLogger(name)
            if handler not in logger.handlers:
                logger.addHandler(handler)

        _HANDLER_TOKEN = handler
        _LOGGER.debug("Zainstalowano handler metryk logowania na %s", targets or ["<root>"])
        return handler


__all__ = ["MetricsLoggingHandler", "install_metrics_logging_handler"]

