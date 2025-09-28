"""Domyślna implementacja routera alertów."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, MutableSequence

from bot_core.alerts.base import AlertChannel, AlertMessage, AlertRouter, AlertAuditLog, AlertDeliveryError
from bot_core.alerts.throttle import AlertThrottle


try:  # pragma: no cover - fallback dla gałęzi bez modułu observability
    from bot_core.observability.metrics import (  # type: ignore
        MetricsRegistry,
        get_global_metrics_registry,
    )
except Exception:  # pragma: no cover - minimalny no-op gdy moduł nie istnieje
    class _NoopMetric:
        def inc(self, *_args: object, **_kwargs: object) -> None:
            return None

    class MetricsRegistry:  # type: ignore[override]
        def counter(self, *_args: object, **_kwargs: object) -> _NoopMetric:
            return _NoopMetric()

    def get_global_metrics_registry() -> MetricsRegistry:  # type: ignore[override]
        return MetricsRegistry()


_SUPPRESSED_CHANNEL = "__suppressed__"


@dataclass(slots=True)
class DefaultAlertRouter(AlertRouter):
    """Zarządza kanałami powiadomień i rejestruje zdarzenia w audycie."""

    audit_log: AlertAuditLog
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts"))
    stop_on_error: bool = False
    channels: MutableSequence[AlertChannel] = field(default_factory=list)
    throttle: AlertThrottle | None = None
    metrics_registry: MetricsRegistry | None = None
    metric_labels: Mapping[str, str] | None = None

    _metrics: MetricsRegistry = field(init=False, repr=False)
    _metric_labels: Mapping[str, str] = field(init=False, repr=False)
    _metric_alerts_total: Any = field(init=False, repr=False)
    _metric_alert_failures_total: Any = field(init=False, repr=False)
    _metric_alert_suppressed_total: Any = field(init=False, repr=False)
    _metric_health_errors_total: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._metrics = self.metrics_registry or get_global_metrics_registry()
        base_labels: MutableMapping[str, str] = {}
        if self.metric_labels:
            base_labels.update({str(key): str(value) for key, value in self.metric_labels.items()})
        self._metric_labels = base_labels
        self._metric_alerts_total = self._metrics.counter(
            "alerts_sent_total",
            "Łączna liczba dostarczonych alertów do kanałów powiadomień.",
        )
        self._metric_alert_failures_total = self._metrics.counter(
            "alerts_failed_total",
            "Liczba prób wysyłki alertów zakończonych błędem.",
        )
        self._metric_alert_suppressed_total = self._metrics.counter(
            "alerts_suppressed_total",
            "Liczba alertów wstrzymanych przez mechanizm throttlingu.",
        )
        self._metric_health_errors_total = self._metrics.counter(
            "alert_healthcheck_errors_total",
            "Liczba błędów podczas health-check kanałów alertowych.",
        )

    def register(self, channel: AlertChannel) -> None:
        if any(existing.name == channel.name for existing in self.channels):
            raise ValueError(f"Kanał o nazwie '{channel.name}' został już zarejestrowany")
        self.channels.append(channel)

    def dispatch(self, message: AlertMessage) -> None:
        if self.throttle and not self.throttle.allow(message):
            remaining = self.throttle.remaining_seconds(message)
            self.logger.info(
                "Tłumię powtarzający się alert %s/%s – kolejne wyślemy za %.1fs.",
                message.category,
                message.title,
                remaining,
            )
            self.audit_log.append(message, channel=_SUPPRESSED_CHANNEL)
            self._metric_alert_suppressed_total.inc(
                labels=self._metric_labels_with(channel=_SUPPRESSED_CHANNEL, severity=message.severity)
            )
            return

        failures: Dict[str, str] = {}
        delivered = False
        for channel in list(self.channels):
            try:
                channel.send(message)
            except AlertDeliveryError as exc:  # pragma: no cover - defensive guard
                self.logger.error("Nie udało się wysłać alertu", extra={"channel": channel.name, "error": str(exc)})
                failures[channel.name] = str(exc)
                self._metric_alert_failures_total.inc(
                    labels=self._metric_labels_with(channel=channel.name, severity=message.severity)
                )
                if self.stop_on_error:
                    raise
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Nieznany błąd kanału {channel.name}: {exc}"
                self.logger.exception(error_msg)
                failures[channel.name] = str(exc)
                self._metric_alert_failures_total.inc(
                    labels=self._metric_labels_with(channel=channel.name, severity=message.severity)
                )
                if self.stop_on_error:
                    raise AlertDeliveryError(error_msg) from exc
            else:
                self.audit_log.append(message, channel=channel.name)
                delivered = True
                self._metric_alerts_total.inc(
                    labels=self._metric_labels_with(channel=channel.name, severity=message.severity)
                )

        if failures and not self.stop_on_error:
            summary = ", ".join(f"{name}: {reason}" for name, reason in failures.items())
            self.logger.warning("Część kanałów zgłosiła błędy: %s", summary)

        if delivered and self.throttle:
            self.throttle.record(message)

    def health_snapshot(self) -> Dict[str, Dict[str, str]]:
        snapshot: Dict[str, Dict[str, str]] = {}
        now = datetime.now(timezone.utc).isoformat()
        for channel in self.channels:
            data = {"checked_at": now}
            try:
                data.update(channel.health_check())
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Błąd podczas health-check kanału %s", channel.name)
                data.update({"status": "error", "detail": str(exc)})
                self._metric_health_errors_total.inc(labels=self._metric_labels_with(channel=channel.name))
            snapshot[channel.name] = data
        return snapshot

    def _metric_labels_with(self, **labels: str) -> Mapping[str, str]:
        if not self._metric_labels:
            return {key: str(value) for key, value in labels.items()}
        merged: Dict[str, str] = dict(self._metric_labels)
        for key, value in labels.items():
            merged[key] = str(value)
        return merged


__all__ = ["DefaultAlertRouter"]

