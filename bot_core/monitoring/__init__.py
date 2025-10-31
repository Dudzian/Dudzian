"""Wspólne narzędzia monitoringu runtime."""
from __future__ import annotations

from typing import Mapping

from bot_core.monitoring.exchange_limits import (
    ExchangeLimitMonitor,
    RateLimitEvent,
    RetryEvent,
    configure_exchange_limit_monitor,
    get_exchange_limit_monitor,
)

__all__ = [
    "ExchangeLimitMonitor",
    "RateLimitEvent",
    "RetryEvent",
    "configure_exchange_limit_monitor",
    "get_exchange_limit_monitor",
    "record_rate_limit_wait",
    "record_retry_event",
]


def record_rate_limit_wait(
    *,
    waited: float,
    labels: Mapping[str, str],
    weight: float,
    rule: tuple[float, float, float] | None,
) -> None:
    """Rejestruje oczekiwanie na limiter w globalnym monitorze."""

    monitor = get_exchange_limit_monitor()
    monitor.record_rate_limit_wait(
        RateLimitEvent(waited=waited, labels=labels, weight=weight, rule=rule)
    )


def record_retry_event(
    *,
    operation: str,
    attempt: int,
    delay: float,
    exception: BaseException,
    max_attempts: int,
) -> None:
    """Rejestruje zdarzenie retry watchdog-a."""

    monitor = get_exchange_limit_monitor()
    monitor.record_retry_event(
        RetryEvent(
            operation=operation,
            attempt=attempt,
            delay=delay,
            exception=exception,
            max_attempts=max_attempts,
        )
    )
