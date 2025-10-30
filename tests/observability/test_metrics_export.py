from __future__ import annotations

import logging
from urllib.request import urlopen

from bot_core.alerts.dispatcher import get_alert_dispatcher
from bot_core.logging.config import MetricsLoggingHandler
from bot_core.observability.metrics import MetricsRegistry
from bot_core.observability.exporters import LocalPrometheusExporter


def test_metrics_logging_handler_updates_exchange_metrics() -> None:
    registry = MetricsRegistry()
    handler = MetricsLoggingHandler(registry=registry)

    logger = logging.getLogger("bot_core.exchanges.binance.spot.test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    dispatcher = get_alert_dispatcher()
    captured: list[str] = []
    token = dispatcher.register(lambda event: captured.append(event.source), name="test-metrics-exchange")

    try:
        for _ in range(5):
            logger.error("API rate limit", extra={"latency_ms": 120, "rate_limited": True})
        payload = registry.render_prometheus()
        assert 'bot_exchange_requests_total' in payload
        assert 'exchange="binance"' in payload
        assert 'bot_exchange_rate_limited_total{exchange="binance"}' in payload
        assert any(source.startswith("exchange:binance") for source in captured)
    finally:
        logger.removeHandler(handler)
        dispatcher.unregister(token)


def test_metrics_logging_handler_strategy_and_security() -> None:
    registry = MetricsRegistry()
    handler = MetricsLoggingHandler(registry=registry)

    strategy_logger = logging.getLogger("bot_core.trading.strategies.grid")
    security_logger = logging.getLogger("bot_core.security.license_service")

    strategy_logger.addHandler(handler)
    security_logger.addHandler(handler)

    try:
        strategy_logger.warning("Decision rejected", extra={"decision_latency_seconds": 0.2})
        security_logger.error("License validation failed", extra={"security_event": "validation"})
        payload = registry.render_prometheus()
        assert "bot_strategy_decisions_total" in payload
        assert 'strategy="grid"' in payload
        assert 'outcome="rejected"' in payload
        assert "bot_security_events_total" in payload
        assert 'source="license_service"' in payload
        assert 'event="validation"' in payload
    finally:
        strategy_logger.removeHandler(handler)
        security_logger.removeHandler(handler)


def test_local_prometheus_exporter_serves_metrics() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("test_counter", "licznik testowy")
    counter.inc(labels={"name": "demo"})

    exporter = LocalPrometheusExporter(host="127.0.0.1", port=0, registry=registry)
    try:
        exporter.start()
        url = exporter.metrics_url
        assert url is not None
        response = urlopen(url, timeout=5)
        payload = response.read().decode("utf-8")
        assert "test_counter" in payload
    finally:
        exporter.stop()
