"""Agregacja jakości sygnałów giełdowych oraz korelacja z watchdogami."""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Mapping, MutableMapping, Sequence

from bot_core.exchanges.health import HealthCheckResult, HealthStatus
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry


@dataclass(slots=True)
class SignalExecutionRecord:
    """Opis pojedynczej realizacji sygnału handlowego."""

    timestamp: datetime
    backend: str
    symbol: str
    side: str
    order_type: str
    requested_quantity: float
    filled_quantity: float
    requested_price: float | None
    executed_price: float | None
    slippage_bps: float | None
    latency: float | None
    status: str
    error: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "backend": self.backend,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "requested_price": self.requested_price,
            "executed_price": self.executed_price,
            "slippage_bps": self.slippage_bps,
            "latency": self.latency,
            "status": self.status,
        }
        if self.error:
            payload["error"] = self.error
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(slots=True)
class WatchdogEvent:
    """Reprezentuje wynik watchdog-a powiązany z raportem jakości sygnałów."""

    timestamp: datetime
    check: str
    status: HealthStatus
    latency: float
    backend: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "check": self.check,
            "status": self.status.value,
            "latency": float(self.latency),
        }
        if self.backend:
            payload["backend"] = self.backend
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class SignalQualityReporter:
    """Rejestruje wyniki egzekucji sygnałów i generuje raporty jakości."""

    def __init__(
        self,
        *,
        exchange_id: str,
        history_limit: int = 200,
        report_dir: str | os.PathLike[str] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        watchdog_history_limit: int | None = None,
    ) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit musi być dodatni")
        self._exchange_id = exchange_id
        self._records: Deque[SignalExecutionRecord] = deque(maxlen=int(history_limit))
        self._report_dir = Path(report_dir) if report_dir else Path("reports/exchanges/signal_quality")
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = metrics_registry or get_global_metrics_registry()
        labels = {"exchange": exchange_id, "component": "signal_quality"}
        self._metric_labels = labels
        watchdog_limit = watchdog_history_limit if watchdog_history_limit is not None else max(10, int(history_limit))
        self._watchdog_events: Deque[WatchdogEvent] = deque(maxlen=watchdog_limit)
        self._fill_ratio_hist = self._metrics.histogram(
            "exchange_signal_fill_ratio",
            "Rozkład fill ratio dla sygnałów giełdowych.",
            buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
        )
        self._slippage_hist = self._metrics.histogram(
            "exchange_signal_slippage_bps",
            "Rozkład poślizgu (bps) pomiędzy zleceniem a realizacją.",
            buckets=(0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0),
        )
        self._latency_hist = self._metrics.histogram(
            "exchange_signal_latency_seconds",
            "Latencja realizacji sygnału w sekundach.",
            buckets=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self._status_counter = self._metrics.counter(
            "exchange_signal_status_total",
            "Liczba sygnałów według statusu realizacji.",
        )
        self._watchdog_status_counter = self._metrics.counter(
            "exchange_watchdog_events_total",
            "Liczba wyników watchdog-ów giełdowych według statusu.",
        )
        self._watchdog_status_gauge = self._metrics.gauge(
            "exchange_watchdog_status",
            "Ostatni status watchdog-a (0=healthy, 1=degraded, 2=unavailable).",
        )
        self._watchdog_alert_counter = self._metrics.counter(
            "exchange_watchdog_degradation_total",
            "Liczba zdarzeń degradacji lub niedostępności backendu giełdowego.",
        )

    @property
    def records(self) -> tuple[SignalExecutionRecord, ...]:
        return tuple(self._records)

    def record_success(
        self,
        *,
        backend: str,
        symbol: str,
        side: str,
        order_type: str,
        requested_quantity: float,
        requested_price: float | None,
        filled_quantity: float,
        executed_price: float | None,
        latency: float | None,
        extra: Mapping[str, object] | None = None,
    ) -> SignalExecutionRecord:
        fill_ratio = 0.0
        if requested_quantity > 0:
            fill_ratio = max(0.0, min(1.0, filled_quantity / requested_quantity))
        slippage_bps: float | None = None
        if requested_price and executed_price:
            try:
                if requested_price > 0:
                    slippage_bps = abs(executed_price - requested_price) / requested_price * 10_000.0
            except ZeroDivisionError:
                slippage_bps = None
        record = SignalExecutionRecord(
            timestamp=datetime.now(timezone.utc),
            backend=backend,
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_quantity=requested_quantity,
            filled_quantity=filled_quantity,
            requested_price=requested_price,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            latency=latency,
            status="filled" if fill_ratio >= 0.999 else "partial",
            error=None,
            extra=dict(extra or {}),
        )
        self._append_record(record)
        self._fill_ratio_hist.observe(fill_ratio, labels=self._metric_labels)
        if slippage_bps is not None:
            self._slippage_hist.observe(slippage_bps, labels=self._metric_labels)
        if latency is not None:
            self._latency_hist.observe(latency, labels=self._metric_labels)
        self._status_counter.inc(labels={**self._metric_labels, "status": record.status})
        self._persist()
        return record

    def record_failure(
        self,
        *,
        backend: str,
        symbol: str,
        side: str,
        order_type: str,
        requested_quantity: float,
        requested_price: float | None,
        error: Exception,
    ) -> SignalExecutionRecord:
        record = SignalExecutionRecord(
            timestamp=datetime.now(timezone.utc),
            backend=backend,
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_quantity=requested_quantity,
            filled_quantity=0.0,
            requested_price=requested_price,
            executed_price=None,
            slippage_bps=None,
            latency=None,
            status="failed",
            error=str(error),
            extra={},
        )
        self._append_record(record)
        self._status_counter.inc(labels={**self._metric_labels, "status": record.status})
        self._persist()
        return record

    def record_watchdog_results(
        self,
        results: Sequence[HealthCheckResult],
        *,
        backend: str | None = None,
    ) -> tuple[WatchdogEvent, ...]:
        """Zapisuje wyniki watchdog-ów i aktualizuje metryki degradacji."""

        if not results:
            return tuple()

        timestamp = datetime.now(timezone.utc)
        events: list[WatchdogEvent] = []
        for result in results:
            event = WatchdogEvent(
                timestamp=timestamp,
                check=result.name,
                status=result.status,
                latency=result.latency,
                backend=backend,
                details=dict(result.details),
            )
            self._append_watchdog_event(event)
            events.append(event)

            status_value = 0.0
            if result.status is HealthStatus.DEGRADED:
                status_value = 1.0
            elif result.status is HealthStatus.UNAVAILABLE:
                status_value = 2.0

            labels = {**self._metric_labels, "check": result.name}
            self._watchdog_status_counter.inc(labels={**labels, "status": result.status.value})
            self._watchdog_status_gauge.set(status_value, labels=labels)
            if status_value > 0.0:
                self._watchdog_alert_counter.inc(labels=labels)

        self._persist()
        return tuple(events)

    def summarize(self) -> Mapping[str, object]:
        total = len(self._records)
        if total == 0:
            payload: dict[str, object] = {
                "exchange": self._exchange_id,
                "total": 0,
                "failures": 0,
                "fill_ratio": 0.0,
                "slippage_bps": 0.0,
                "records": [],
            }
        else:
            filled = [record for record in self._records if record.status != "failed"]
            failures = total - len(filled)
            sum_ratio = 0.0
            sum_slippage = 0.0
            counted_slippage = 0
            for record in filled:
                if record.requested_quantity > 0:
                    sum_ratio += min(1.0, record.filled_quantity / record.requested_quantity)
                if record.slippage_bps is not None:
                    sum_slippage += record.slippage_bps
                    counted_slippage += 1
            avg_ratio = sum_ratio / len(filled) if filled else 0.0
            avg_slippage = sum_slippage / counted_slippage if counted_slippage else 0.0
            payload = {
                "exchange": self._exchange_id,
                "total": total,
                "failures": failures,
                "fill_ratio": avg_ratio,
                "slippage_bps": avg_slippage,
                "records": [record.as_dict() for record in self._records],
            }

        payload["watchdog"] = self._summarize_watchdog_events()
        return payload

    def _append_record(self, record: SignalExecutionRecord) -> None:
        self._records.append(record)

    def _append_watchdog_event(self, event: WatchdogEvent) -> None:
        self._watchdog_events.append(event)

    def _summarize_watchdog_events(self) -> Mapping[str, object]:
        if not self._watchdog_events:
            return {"recent": [], "alerts": [], "last_status": {}}

        recent = [event.as_dict() for event in list(self._watchdog_events)[-10:]]
        last_status: dict[str, MutableMapping[str, object]] = {}
        alerts: list[MutableMapping[str, object]] = []

        for event in self._watchdog_events:
            payload: MutableMapping[str, object] = {
                "status": event.status.value,
                "observed_at": event.timestamp.astimezone(timezone.utc).isoformat(),
                "latency": float(event.latency),
            }
            if event.backend:
                payload["backend"] = event.backend
            if event.details:
                payload["details"] = dict(event.details)
            last_status[event.check] = payload

        seen_alerts: set[str] = set()
        for event in reversed(self._watchdog_events):
            if event.status is HealthStatus.HEALTHY:
                continue
            if event.check in seen_alerts:
                continue
            alert_payload: MutableMapping[str, object] = {
                "check": event.check,
                "status": event.status.value,
                "observed_at": event.timestamp.astimezone(timezone.utc).isoformat(),
                "latency": float(event.latency),
            }
            if event.backend:
                alert_payload["backend"] = event.backend
            if event.details:
                alert_payload["details"] = dict(event.details)
            alerts.append(alert_payload)
            seen_alerts.add(event.check)

        alerts.reverse()
        return {"recent": recent, "alerts": alerts, "last_status": last_status}

    def _persist(self) -> None:
        summary = self.summarize()
        path = self._report_dir / f"{self._exchange_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        payload = json.dumps(summary, indent=2, sort_keys=True)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)


__all__ = ["SignalExecutionRecord", "SignalQualityReporter", "WatchdogEvent"]
