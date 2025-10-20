import itertools

import pytest

from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.health import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    RetryPolicy,
    Watchdog,
)


def test_watchdog_retries_transient_failures() -> None:
    attempts: list[int] = []
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0))
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0, clock=lambda: 0.0)
    watchdog = Watchdog(retry_policy=policy, circuit_breaker=breaker, sleep=lambda _: None)

    def flaky() -> str:
        attempts.append(len(attempts))
        if len(attempts) < 2:
            raise ExchangeNetworkError("temporary outage")
        return "ok"

    result = watchdog.execute("flaky_operation", flaky)

    assert result == "ok"
    assert len(attempts) == 2
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_watchdog_opens_and_recovers_circuit() -> None:
    clock = itertools.count()
    policy = RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0))
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=2.0,
        half_open_success_threshold=1,
        clock=lambda: next(clock),
    )
    watchdog = Watchdog(retry_policy=policy, circuit_breaker=breaker, sleep=lambda _: None)

    with pytest.raises(ExchangeNetworkError):
        watchdog.execute("unstable", lambda: (_ for _ in ()).throw(ExchangeNetworkError("boom")))
    with pytest.raises(ExchangeNetworkError):
        watchdog.execute("unstable", lambda: (_ for _ in ()).throw(ExchangeNetworkError("boom")))

    with pytest.raises(CircuitOpenError):
        watchdog.execute("unstable", lambda: "should not run")

    # advance time beyond recovery timeout
    next(clock)
    next(clock)

    result = watchdog.execute("unstable", lambda: "recovered")
    assert result == "recovered"
    assert breaker.state is CircuitState.CLOSED


def test_health_monitor_combines_statuses() -> None:
    policy = RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0))
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0, clock=lambda: 0.0)
    watchdog = Watchdog(retry_policy=policy, circuit_breaker=breaker, sleep=lambda _: None)

    def failing_check() -> None:
        raise ExchangeNetworkError("api down")

    monitor = HealthMonitor(
        [
            HealthCheck(name="public_api", check=lambda: None),
            HealthCheck(name="private_api", check=failing_check, critical=True),
        ],
        watchdog=watchdog,
    )

    results = monitor.run()
    statuses = {result.name: result.status for result in results}
    assert statuses["public_api"] is HealthStatus.HEALTHY
    assert statuses["private_api"] is HealthStatus.UNAVAILABLE
    assert HealthMonitor.overall_status(results) is HealthStatus.UNAVAILABLE
