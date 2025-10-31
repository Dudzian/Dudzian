from __future__ import annotations

import types

import pytest

import bot_core.exchanges.rate_limiter as rate_limiter_module
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.health import RetryPolicy, Watchdog
from bot_core.monitoring import ExchangeLimitMonitor, configure_exchange_limit_monitor
from bot_core.observability.metrics import MetricsRegistry


class _StubMonitor(ExchangeLimitMonitor):
    def __init__(self) -> None:
        super().__init__(
            metrics_registry=MetricsRegistry(),
            wait_alert_threshold=999.0,
            retry_alert_threshold=99,
        )
        self.wait_events = []
        self.retry_events = []

    def record_rate_limit_wait(self, event):  # type: ignore[override]
        self.wait_events.append(event)

    def record_retry_event(self, event):  # type: ignore[override]
        self.retry_events.append(event)


@pytest.fixture(autouse=True)
def _reset_monitor():
    configure_exchange_limit_monitor(monitor=None)
    yield
    configure_exchange_limit_monitor(monitor=None)


def test_rate_limiter_notifies_monitor(monkeypatch):
    stub = _StubMonitor()
    configure_exchange_limit_monitor(monitor=stub)

    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        return clock["value"]

    def fake_sleep(duration: float) -> None:
        clock["value"] += duration

    monkeypatch.setattr(
        rate_limiter_module,
        "time",
        types.SimpleNamespace(monotonic=fake_monotonic, sleep=fake_sleep),
    )

    limiter = rate_limiter_module.RateLimiter(
        [rate_limiter_module.RateLimitRule(rate=1, per=1)],
        metrics_registry=MetricsRegistry(),
        metric_labels={"exchange": "test", "environment": "paper"},
    )

    limiter.acquire()
    limiter.acquire()

    assert stub.wait_events, "Monitor powinien otrzymać informację o oczekiwaniu"
    event = stub.wait_events[0]
    assert pytest.approx(event.waited, abs=1e-6) == 1.0
    assert event.labels["exchange"] == "test"


def test_watchdog_retry_notifies_monitor():
    stub = _StubMonitor()
    configure_exchange_limit_monitor(monitor=stub)

    watchdog = Watchdog(
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=(0.0, 0.0)),
        sleep=lambda _: None,
    )

    def _fail() -> None:
        raise ExchangeNetworkError("fail", reason=None)

    with pytest.raises(ExchangeNetworkError):
        watchdog.execute("binance_spot_fetch_account", _fail)

    assert stub.retry_events, "Monitor powinien otrzymać informację o retry"
    retry_event = stub.retry_events[0]
    assert retry_event.operation == "binance_spot_fetch_account"
    assert retry_event.attempt == 1
