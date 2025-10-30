"""Wspólny limiter żądań HTTP dla adapterów giełdowych."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """Opis pojedynczego limitu tokenów."""

    rate: float
    per: float
    weight: float = 1.0

    def normalized(self) -> tuple[float, float, float]:
        if self.rate <= 0 or self.per <= 0 or self.weight <= 0:
            raise ValueError("RateLimitRule wymaga dodatnich wartości rate/per/weight")
        return float(self.rate), float(self.per), float(self.weight)


class RateLimiter:
    """Realizuje algorytm token bucket z obsługą wielu reguł."""

    __slots__ = (
        "_rules",
        "_clock",
        "_lock",
        "_state",
        "_metrics",
        "_metric_labels",
        "_metrics_gauge",
        "_metrics_wait",
    )

    def __init__(
        self,
        rules: Sequence[RateLimitRule],
        *,
        clock: Callable[[], float] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        metric_labels: Mapping[str, str] | None = None,
    ) -> None:
        if not rules:
            raise ValueError("RateLimiter wymaga przynajmniej jednej reguły")
        self._rules: tuple[tuple[float, float, float], ...] = tuple(
            rule.normalized() for rule in rules
        )
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        now = self._clock()
        self._state = [
            {
                "tokens": rate,
                "updated": now,
                "rule": (rate, per, weight),
            }
            for rate, per, weight in self._rules
        ]
        self._metrics = metrics_registry or get_global_metrics_registry()
        labels = dict(metric_labels or {})
        labels.setdefault("component", "exchange_rate_limiter")
        self._metric_labels = labels
        self._metrics_gauge = self._metrics.gauge(
            "exchange_rate_limiter_tokens",
            "Aktualna liczba tokenów dostępna w limiterze.",
        )
        self._metrics_wait = self._metrics.histogram(
            "exchange_rate_limiter_wait_seconds",
            "Czas oczekiwania na zwolnienie limitu.",
            buckets=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

    def acquire(self, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError("Waga żądania musi być dodatnia")

        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                ready = True
                next_wait = 0.0
                for bucket in self._state:
                    rate, per, rule_weight = bucket["rule"]
                    tokens = bucket["tokens"]
                    elapsed = max(0.0, now - bucket["updated"])
                    refill = elapsed * (rate / per)
                    if refill > 0:
                        tokens = min(rate, tokens + refill)
                    required = weight * rule_weight
                    if tokens >= required:
                        tokens -= required
                    else:
                        ready = False
                        missing = required - tokens
                        wait_for = missing / (rate / per)
                        next_wait = max(next_wait, wait_for)
                    bucket["tokens"] = tokens
                    bucket["updated"] = now
                    self._metrics_gauge.set(tokens, labels={**self._metric_labels, "rule": f"{rate}/{per}"})

                if ready:
                    if waited:
                        self._metrics_wait.observe(waited, labels=self._metric_labels)
                    return

            sleep_for = max(0.0, next_wait)
            if sleep_for == 0:
                sleep_for = 0.01
            time.sleep(sleep_for)
            waited += sleep_for


class RateLimiterRegistry:
    """Rejestr współdzielący limitery między instancjami adapterów."""

    def __init__(self) -> None:
        self._limiters: MutableMapping[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def configure(
        self,
        key: str,
        rules: Sequence[RateLimitRule],
        *,
        clock: Callable[[], float] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        metric_labels: Mapping[str, str] | None = None,
    ) -> RateLimiter:
        if not key:
            raise ValueError("Identyfikator limitera nie może być pusty")
        normalized_rules = tuple(rule.normalized() for rule in rules)
        with self._lock:
            limiter = self._limiters.get(key)
            if limiter is not None:
                return limiter
            limiter = RateLimiter(
                [RateLimitRule(rate, per, weight) for rate, per, weight in normalized_rules],
                clock=clock,
                metrics_registry=metrics_registry,
                metric_labels=metric_labels,
            )
            self._limiters[key] = limiter
            return limiter


_GLOBAL_REGISTRY = RateLimiterRegistry()


def get_global_rate_limiter_registry() -> RateLimiterRegistry:
    return _GLOBAL_REGISTRY


def set_global_rate_limiter_registry(registry: RateLimiterRegistry) -> None:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry


def normalize_rate_limit_rules(
    rules: Sequence[RateLimitRule | Mapping[str, Any]] | None,
    default: Sequence[RateLimitRule],
) -> tuple[RateLimitRule, ...]:
    """Konwertuje konfigurację reguł na krotkę :class:`RateLimitRule`."""

    if rules is None:
        return tuple(default)
    normalized: list[RateLimitRule] = []
    for rule in rules:
        if isinstance(rule, RateLimitRule):
            normalized.append(rule)
        elif isinstance(rule, Mapping):
            normalized.append(
                RateLimitRule(
                    rate=float(rule.get("rate", 0.0)),
                    per=float(rule.get("per", 1.0)),
                    weight=float(rule.get("weight", 1.0)),
                )
            )
        else:
            raise TypeError(f"Nieobsługiwany typ reguły limitera: {type(rule)!r}")
    if not normalized:
        return tuple(default)
    return tuple(normalized)


__all__ = [
    "RateLimitRule",
    "RateLimiter",
    "RateLimiterRegistry",
    "get_global_rate_limiter_registry",
    "set_global_rate_limiter_registry",
    "normalize_rate_limit_rules",
]
