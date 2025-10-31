"""Narzędzia do monitorowania kondycji adapterów giełdowych."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import random
import time
from typing import Callable, Iterable, Mapping, Sequence, TypeVar

from bot_core.exchanges.errors import ExchangeError, ExchangeNetworkError, ExchangeThrottlingError
from bot_core.monitoring import record_retry_event


_LOGGER = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Możliwe stany komponentu giełdowego."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class HealthCheckResult:
    """Wynik pojedynczego testu zdrowia."""

    name: str
    status: HealthStatus
    latency: float
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class HealthCheck:
    """Deklaracja testu zdrowia."""

    name: str
    check: Callable[[], object]
    critical: bool = True


@dataclass(slots=True)
class RetryPolicy:
    """Parametry ponawiania operacji w ramach watchdog-a."""

    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 2.0
    jitter: tuple[float, float] = (0.0, 0.2)

    def compute_delay(self, attempt: int) -> float:
        """Zwraca czas uśpienia po nieudanej próbie."""

        capped = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        jitter = 0.0
        if self.jitter[1] > 0:
            jitter = random.uniform(self.jitter[0], self.jitter[1])
        return capped + jitter


class CircuitState(str, Enum):
    """Stan wewnętrzny wyłącznika przeciążeniowego."""

    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


class CircuitOpenError(RuntimeError):
    """Wyjątek sygnalizujący otwarty wyłącznik."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"Circuit breaker for operation {operation!r} is open")
        self.operation = operation


@dataclass(slots=True)
class CircuitBreaker:
    """Prosta implementacja wyłącznika przeciążeniowego."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_success_threshold: int = 2
    clock: Callable[[], float] = time.monotonic

    _state: CircuitState = field(init=False, default=CircuitState.CLOSED)
    _failure_count: int = field(init=False, default=0)
    _success_count: int = field(init=False, default=0)
    _last_failure_ts: float = field(init=False, default=0.0)

    def before_call(self, operation: str) -> None:
        """Waliduje stan przed wywołaniem operacji."""

        if self._state is CircuitState.OPEN:
            elapsed = self.clock() - self._last_failure_ts
            if elapsed >= self.recovery_timeout:
                _LOGGER.debug("Circuit breaker half-open for operation %s", operation)
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
            else:
                raise CircuitOpenError(operation)

    def record_failure(self, operation: str) -> None:
        self._failure_count += 1
        self._last_failure_ts = self.clock()
        if self._state is CircuitState.HALF_OPEN:
            # Po porażce w stanie półotwartym natychmiast otwieramy obwód.
            self._state = CircuitState.OPEN
            _LOGGER.warning("Circuit breaker re-opened for %s after half-open failure", operation)
            return
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            _LOGGER.error(
                "Circuit breaker opened for %s after %s consecutive failures",
                operation,
                self._failure_count,
            )

    def record_success(self, operation: str) -> None:
        self._failure_count = 0
        if self._state is CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_success_threshold:
                self._state = CircuitState.CLOSED
                _LOGGER.info("Circuit breaker closed for %s after recovery", operation)
        else:
            self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count


T = TypeVar("T")


@dataclass(slots=True)
class Watchdog:
    """Łączy retry policy z wyłącznikiem przeciążeniowym."""

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    retry_exceptions: Sequence[type[Exception]] = (
        ExchangeNetworkError,
        ExchangeThrottlingError,
    )
    sleep: Callable[[float], None] = time.sleep

    def execute(self, operation: str, func: Callable[[], T]) -> T:
        """Uruchamia funkcję z obsługą retry/circuit breaker."""

        self.circuit_breaker.before_call(operation)
        attempt = 1
        while True:
            try:
                result = func()
            except Exception as exc:  # noqa: BLE001 - kontrolowana obsługa wyjątków
                should_retry = isinstance(exc, tuple(self.retry_exceptions))
                self.circuit_breaker.record_failure(operation)
                if not should_retry or attempt >= self.retry_policy.max_attempts:
                    raise
                delay = self.retry_policy.compute_delay(attempt)
                _LOGGER.warning(
                    "Watchdog retry %s (attempt=%s/%s, delay=%.2fs, error=%s)",
                    operation,
                    attempt,
                    self.retry_policy.max_attempts,
                    delay,
                    type(exc).__name__,
                )
                try:
                    record_retry_event(
                        operation=operation,
                        attempt=attempt,
                        delay=delay,
                        exception=exc,
                        max_attempts=self.retry_policy.max_attempts,
                    )
                except Exception:  # pragma: no cover - monitorowanie nie powinno zatrzymywać retry
                    _LOGGER.debug("record_retry_event failed", exc_info=True)
                attempt += 1
                self.sleep(delay)
                continue
            else:
                self.circuit_breaker.record_success(operation)
                return result


class HealthMonitor:
    """Uruchamia zdefiniowane testy zdrowia z użyciem watchdog-a."""

    def __init__(
        self,
        checks: Sequence[HealthCheck],
        *,
        watchdog: Watchdog | None = None,
    ) -> None:
        self._checks = tuple(checks)
        self._watchdog = watchdog or Watchdog()

    def run(self) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for check in self._checks:
            start = time.monotonic()
            try:
                self._watchdog.execute(check.name, check.check)
            except CircuitOpenError as exc:
                latency = time.monotonic() - start
                results.append(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNAVAILABLE,
                        latency=latency,
                        details={"error": str(exc)},
                    )
                )
            except ExchangeError as exc:
                latency = time.monotonic() - start
                status = HealthStatus.DEGRADED if not check.critical else HealthStatus.UNAVAILABLE
                results.append(
                    HealthCheckResult(
                        name=check.name,
                        status=status,
                        latency=latency,
                        details={"error": str(exc)},
                    )
                )
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                latency = time.monotonic() - start
                _LOGGER.exception("Unexpected error during health check %s", check.name)
                results.append(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNAVAILABLE,
                        latency=latency,
                        details={"error": str(exc)},
                    )
                )
            else:
                latency = time.monotonic() - start
                results.append(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.HEALTHY,
                        latency=latency,
                        details={},
                    )
                )
        return results

    @staticmethod
    def overall_status(results: Iterable[HealthCheckResult]) -> HealthStatus:
        worst = HealthStatus.HEALTHY
        for result in results:
            if result.status is HealthStatus.UNAVAILABLE:
                return HealthStatus.UNAVAILABLE
            if result.status is HealthStatus.DEGRADED:
                worst = HealthStatus.DEGRADED
        return worst


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "HealthCheck",
    "HealthCheckResult",
    "HealthMonitor",
    "HealthStatus",
    "RetryPolicy",
    "Watchdog",
]
