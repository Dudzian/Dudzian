"""Eksport metryk Prometheus dla bota KryptoLowca."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

try:  # pragma: no cover - biblioteka instalowana w środowisku runtime
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
except Exception:  # pragma: no cover
    CollectorRegistry = None  # type: ignore
    Counter = Gauge = Histogram = None  # type: ignore
    start_http_server = None  # type: ignore


@dataclass(slots=True)
class RiskSnapshot:
    symbol: str
    state: str
    fraction: float
    mode: str


class PrometheusMetrics:
    """Wspólny rejestr metryk; bezpieczny dla wielu wątków."""

    def __init__(self) -> None:
        self._registry = CollectorRegistry() if CollectorRegistry else None
        self._lock = threading.RLock()
        self._port: Optional[int] = None
        self._started = False
        self._last_risk: Dict[str, RiskSnapshot] = {}

        if self._registry is not None:
            self.decisions_total = Counter(  # type: ignore[call-arg]
                "kryptolowca_risk_decisions_total",
                "Ile decyzji ryzyka zostało wyemitowanych",
                labelnames=("state", "mode"),
                registry=self._registry,
            )
            self.risk_fraction = Gauge(  # type: ignore[call-arg]
                "kryptolowca_risk_fraction",
                "Frakcja kapitału rekomendowana przez moduł ryzyka",
                labelnames=("symbol",),
                registry=self._registry,
            )
            self.risk_state = Gauge(  # type: ignore[call-arg]
                "kryptolowca_risk_state",
                "Aktualny stan guardrails (0=ok,1=warn,2=lock)",
                labelnames=("symbol",),
                registry=self._registry,
            )
            self.orders_total = Counter(  # type: ignore[call-arg]
                "kryptolowca_orders_total",
                "Liczba zleceń wysłanych przez AutoTradera",
                labelnames=("symbol", "side"),
                registry=self._registry,
            )
            self.last_order_fraction = Gauge(  # type: ignore[call-arg]
                "kryptolowca_last_order_fraction",
                "Frakcja użyta przy ostatnim zleceniu",
                labelnames=("symbol",),
                registry=self._registry,
            )
            self.closed_trades_total = Counter(  # type: ignore[call-arg]
                "kryptolowca_closed_trades_total",
                "Liczba zamkniętych transakcji",
                labelnames=("symbol",),
                registry=self._registry,
            )
            self.realised_pnl = Histogram(  # type: ignore[call-arg]
                "kryptolowca_realised_pnl",
                "Dystrybucja PnL zamkniętych transakcji",
                labelnames=("symbol",),
                buckets=(-5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0),
                registry=self._registry,
            )
            self.open_positions = Gauge(  # type: ignore[call-arg]
                "kryptolowca_open_positions",
                "Liczba otwartych pozycji (wg portfela risk managera)",
                labelnames=("mode",),
                registry=self._registry,
            )

        port = int(os.getenv("KRYPT_LOWCA_PROMETHEUS_PORT", "0") or 0)
        if port and start_http_server and self._registry is not None:
            start_http_server(port, registry=self._registry)
            self._port = port
            self._started = True

    @property
    def enabled(self) -> bool:
        return self._registry is not None

    def observe_risk(self, symbol: str, state: str, fraction: float, mode: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.decisions_total.labels(state=state, mode=mode).inc()
            self.risk_fraction.labels(symbol=symbol).set(float(fraction))
            self.risk_state.labels(symbol=symbol).set(self._state_to_number(state))
            self._last_risk[symbol] = RiskSnapshot(symbol, state, float(fraction), mode)

    def record_order(self, symbol: str, side: str, fraction: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.orders_total.labels(symbol=symbol, side=side.lower()).inc()
            self.last_order_fraction.labels(symbol=symbol).set(float(fraction))

    def record_trade_close(self, symbol: str, pnl: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.closed_trades_total.labels(symbol=symbol).inc()
            self.realised_pnl.labels(symbol=symbol).observe(float(pnl))

    def set_open_positions(self, *, count: int, mode: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.open_positions.labels(mode=mode).set(float(count))

    @staticmethod
    def _state_to_number(state: str) -> int:
        mapping = {"ok": 0, "warn": 1, "lock": 2}
        return mapping.get(state.lower(), 1)


metrics = PrometheusMetrics()

__all__ = ["metrics", "PrometheusMetrics", "RiskSnapshot"]
