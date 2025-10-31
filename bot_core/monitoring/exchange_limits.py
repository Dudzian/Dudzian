"""Monitorowanie limitów i retry adapterów giełdowych."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, MutableMapping

from bot_core.alerts.dispatcher import AlertSeverity, emit_alert
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RateLimitEvent:
    """Metadane pojedynczego zdarzenia oczekiwania na limiter."""

    waited: float
    labels: Mapping[str, str]
    weight: float
    rule: tuple[float, float, float] | None


@dataclass(slots=True)
class RetryEvent:
    """Metadane powtórzeń watchdog-a."""

    operation: str
    attempt: int
    delay: float
    exception: BaseException
    max_attempts: int


class ExchangeLimitMonitor:
    """Rejestruje zdarzenia ograniczeń API i generuje alerty."""

    def __init__(
        self,
        *,
        metrics_registry: MetricsRegistry | None = None,
        wait_alert_threshold: float = 1.0,
        wait_alert_streak: int = 3,
        retry_alert_threshold: int = 3,
    ) -> None:
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._wait_counter = self._metrics.counter(
            "exchange_rate_limit_monitor_events_total",
            "Liczba zdarzeń oczekiwania raportowanych przez monitor limitów.",
        )
        self._retry_counter = self._metrics.counter(
            "exchange_retry_monitor_events_total",
            "Liczba zdarzeń retry raportowanych przez monitor limitów.",
        )
        self._wait_alert_counter = self._metrics.counter(
            "exchange_rate_limit_alerts_total",
            "Liczba alertów o przekroczeniu limitów API.",
        )
        self._retry_alert_counter = self._metrics.counter(
            "exchange_retry_alerts_total",
            "Liczba alertów o nadmiernych retry watchdog-a.",
        )
        self._wait_alert_threshold = float(wait_alert_threshold)
        self._wait_alert_streak = max(1, int(wait_alert_streak))
        self._retry_alert_threshold = max(1, int(retry_alert_threshold))
        self._wait_streaks: MutableMapping[tuple[str, str], int] = {}
        self._retry_streaks: MutableMapping[str, int] = {}

    @staticmethod
    def _extract_label(labels: Mapping[str, str], key: str, default: str) -> str:
        value = labels.get(key)
        return str(value) if value is not None else default

    def record_rate_limit_wait(self, event: RateLimitEvent) -> None:
        labels = {str(key): str(value) for key, value in event.labels.items()}
        exchange = self._extract_label(labels, "exchange", "unknown")
        environment = self._extract_label(labels, "environment", "unknown")
        rule = event.rule
        rule_label = f"{rule[0]}/{rule[1]}" if rule else "unknown"
        counter_labels = {**labels, "rule": rule_label}
        self._wait_counter.inc(labels=counter_labels)

        key = (exchange, environment)
        streak = self._wait_streaks.get(key, 0)
        if event.waited >= self._wait_alert_threshold:
            streak += 1
        else:
            streak = 0
        self._wait_streaks[key] = streak

        _LOGGER.info(
            "Rate limit wait %.3fs for %s/%s (weight=%.2f, rule=%s, streak=%s)",
            event.waited,
            exchange,
            environment,
            event.weight,
            rule_label,
            streak,
        )

        if streak >= self._wait_alert_streak:
            context = {
                "exchange": exchange,
                "environment": environment,
                "rule": rule_label,
                "waited": round(event.waited, 3),
                "streak": streak,
            }
            emit_alert(
                "Wielokrotne oczekiwanie na limiter żądań giełdy.",
                severity=AlertSeverity.WARNING,
                source="exchange.limit-monitor",
                context=context,
            )
            self._wait_alert_counter.inc(labels=labels)
            self._wait_streaks[key] = 0

    def record_retry_event(self, event: RetryEvent) -> None:
        operation = event.operation
        exchange = operation.split("_", 1)[0] if operation else "unknown"
        labels = {"exchange": exchange, "operation": operation}
        self._retry_counter.inc(labels=labels)

        _LOGGER.warning(
            "Retry %s attempt=%s/%s delay=%.2fs error=%s",  # noqa: G004
            operation,
            event.attempt,
            event.max_attempts,
            event.delay,
            type(event.exception).__name__,
        )

        streak_key = operation or exchange
        streak = self._retry_streaks.get(streak_key, 0)
        if event.attempt <= 1:
            streak = 1
        else:
            streak += 1
        self._retry_streaks[streak_key] = streak

        threshold_reached = event.attempt >= self._retry_alert_threshold
        nearing_exhaustion = event.attempt >= max(1, event.max_attempts - 1)
        if threshold_reached or nearing_exhaustion:
            context = {
                "operation": operation,
                "attempt": event.attempt,
                "max_attempts": event.max_attempts,
                "delay": round(event.delay, 3),
                "exception": type(event.exception).__name__,
            }
            emit_alert(
                "Nadmierna liczba ponowień w watchdogu adaptera.",
                severity=AlertSeverity.ERROR,
                source="exchange.limit-monitor",
                context=context,
                exception=event.exception,
            )
            self._retry_alert_counter.inc(labels=labels)
            self._retry_streaks[streak_key] = 0


_MONITOR: ExchangeLimitMonitor | None = None


def get_exchange_limit_monitor() -> ExchangeLimitMonitor:
    """Zwraca singleton monitorujący limity giełdowe."""

    global _MONITOR
    if _MONITOR is None:
        _MONITOR = ExchangeLimitMonitor()
    return _MONITOR


def configure_exchange_limit_monitor(*, monitor: ExchangeLimitMonitor | None = None) -> None:
    """Pozwala nadpisać globalny monitor (używane w testach)."""

    global _MONITOR
    _MONITOR = monitor


__all__ = [
    "ExchangeLimitMonitor",
    "RateLimitEvent",
    "RetryEvent",
    "configure_exchange_limit_monitor",
    "get_exchange_limit_monitor",
]
