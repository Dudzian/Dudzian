"""Komponent utrzymujący stan health decision feedu."""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone


class FeedHealthTracker:
    def __init__(
        self,
        *,
        feed_channels: tuple[str, ...],
        latency_buffer_size: int,
        percentile_fn: Callable[[Iterable[float], float], float],
    ) -> None:
        self._feed_channels = tuple(feed_channels)
        self._latency_buffer_size = max(1, int(latency_buffer_size))
        self._percentile = percentile_fn
        self._feed_reconnects = 0
        self._feed_downtime_started: float | None = None
        self._feed_downtime_total = 0.0
        self._feed_last_error = ""
        self._transport_latency_samples: dict[str, deque[float]] = {
            "grpc": deque(maxlen=self._latency_buffer_size),
            "fallback": deque(maxlen=self._latency_buffer_size),
        }
        self._feed_transport_stats: dict[str, dict[str, object]] = {
            "grpc": {
                "status": "initializing",
                "p50_ms": None,
                "p95_ms": None,
                "reconnects": 0,
                "downtimeMs": 0.0,
                "lastError": "",
                "updatedAt": 0.0,
            },
            "fallback": {
                "status": "offline",
                "p50_ms": None,
                "p95_ms": None,
                "reconnects": 0,
                "downtimeMs": 0.0,
                "lastError": "",
                "updatedAt": 0.0,
            },
        }
        self._feed_health: dict[str, object] = {
            "status": "initializing",
            "reconnects": 0,
            "downtimeMs": 0.0,
            "lastError": "",
            "channels": list(self._feed_channels),
            "channelStates": {},
            "transports": {key: dict(value) for key, value in self._feed_transport_stats.items()},
        }
        self._feed_sla_report: dict[str, object] = {}
        self._consecutive_degraded_periods: int = 0
        self._consecutive_healthy_periods: int = 0

    @property
    def reconnects(self) -> int:
        return self._feed_reconnects

    @reconnects.setter
    def reconnects(self, value: int) -> None:
        self._feed_reconnects = max(0, int(value))

    @property
    def downtime_started(self) -> float | None:
        return self._feed_downtime_started

    @downtime_started.setter
    def downtime_started(self, value: float | None) -> None:
        self._feed_downtime_started = value

    @property
    def downtime_total(self) -> float:
        return self._feed_downtime_total

    @downtime_total.setter
    def downtime_total(self, value: float) -> None:
        self._feed_downtime_total = max(0.0, float(value))

    @property
    def last_error(self) -> str:
        return self._feed_last_error

    @last_error.setter
    def last_error(self, value: str) -> None:
        self._feed_last_error = str(value)

    @property
    def feed_health(self) -> dict[str, object]:
        return self._feed_health

    @property
    def feed_sla_report(self) -> dict[str, object]:
        return self._feed_sla_report

    def latency_samples_for(self, key: str | None) -> deque[float]:
        transport = key or "grpc"
        samples = self._transport_latency_samples.get(transport)
        if samples is None:
            samples = deque(maxlen=self._latency_buffer_size)
            self._transport_latency_samples[transport] = samples
        return samples

    def mark_feed_disconnected(self) -> None:
        if self._feed_downtime_started is None:
            self._feed_downtime_started = time.monotonic()

    def mark_feed_connected(self) -> None:
        if self._feed_downtime_started is not None:
            self._feed_downtime_total += max(0.0, time.monotonic() - self._feed_downtime_started)
            self._feed_downtime_started = None

    def update_feed_health(
        self,
        *,
        status: str | None,
        reconnects: int | None,
        last_error: str | None,
        next_retry: float | None,
        latest_latency: float | None,
        transport_key: str,
        channel_status: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, object], float | None, float | None]:
        payload = self._feed_health
        if status is not None:
            payload["status"] = status
        if reconnects is not None:
            self.reconnects = reconnects
        payload["reconnects"] = self._feed_reconnects
        if last_error is not None:
            self.last_error = last_error
        payload["lastError"] = self._feed_last_error
        if latest_latency is not None:
            payload["lastLatencyMs"] = max(0.0, float(latest_latency))
        downtime_ms = self._feed_downtime_total * 1000.0
        if self._feed_downtime_started is not None:
            downtime_ms += max(0.0, (time.monotonic() - self._feed_downtime_started) * 1000.0)
        payload["downtimeMs"] = max(0.0, float(downtime_ms))
        payload["channels"] = list(self._feed_channels)
        if next_retry is not None:
            payload["nextRetrySeconds"] = max(0.0, float(next_retry))
        else:
            payload.pop("nextRetrySeconds", None)
        latencies = list(self.latency_samples_for(transport_key))
        latency_p95: float | None = None
        latency_p50: float | None = None
        if latencies:
            latency_p95 = float(self._percentile(latencies, 95.0))
            payload["p95LatencyMs"] = latency_p95
            latency_p50 = float(statistics.median(latencies))
            payload["p50LatencyMs"] = latency_p50
        else:
            payload.pop("p95LatencyMs", None)
            payload.pop("p50LatencyMs", None)
        payload["channelStates"] = {channel: dict(state) for channel, state in channel_status.items()}
        self._update_transport_breakdown(
            transport_key,
            payload,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
        )
        payload["transports"] = self.serialize_transport_stats()
        return payload, latency_p95, latency_p50

    def _update_transport_breakdown(
        self,
        key: str,
        payload: Mapping[str, object],
        *,
        latency_p50: float | None,
        latency_p95: float | None,
    ) -> None:
        stats = self._feed_transport_stats.setdefault(
            key,
            {
                "status": "unknown",
                "p50_ms": None,
                "p95_ms": None,
                "reconnects": 0,
                "downtimeMs": 0.0,
                "lastError": "",
                "updatedAt": 0.0,
            },
        )
        stats.update(
            {
                "status": str(payload.get("status", stats.get("status", "unknown"))),
                "p50_ms": latency_p50,
                "p95_ms": latency_p95,
                "reconnects": int(payload.get("reconnects", stats.get("reconnects", 0)) or 0),
                "downtimeMs": float(payload.get("downtimeMs", stats.get("downtimeMs", 0.0)) or 0.0),
                "lastError": str(payload.get("lastError", stats.get("lastError", "")) or ""),
                "updatedAt": time.time(),
            }
        )

    def serialize_transport_stats(self) -> dict[str, dict[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for key, stats in self._feed_transport_stats.items():
            snapshot[key] = {
                "status": stats.get("status"),
                "p50_ms": stats.get("p50_ms"),
                "p95_ms": stats.get("p95_ms"),
                "reconnects": stats.get("reconnects", 0),
                "downtimeMs": stats.get("downtimeMs", 0.0),
                "lastError": stats.get("lastError", ""),
                "updatedAt": stats.get("updatedAt", 0.0),
            }
        return snapshot

    def build_sla_report(
        self,
        *,
        transport_source: str,
        thresholds: Mapping[str, float | None],
    ) -> dict[str, object]:
        latencies = list(self.latency_samples_for("grpc"))
        if not latencies:
            latencies = list(self.latency_samples_for(transport_source))
        health = self._feed_health
        downtime_value = float(health.get("downtimeMs", 0.0))
        next_retry_value = health.get("nextRetrySeconds")
        status_value = health.get("status", "unknown")
        last_error_value = health.get("lastError", "")
        stats_payload: dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(latencies),
            "reconnects": self._feed_reconnects,
            "downtime_ms": downtime_value,
            "downtimeMs": downtime_value,
            "status": status_value,
            "last_error": last_error_value,
            "lastError": last_error_value,
            "next_retry_seconds": float(next_retry_value) if next_retry_value is not None else None,
            "nextRetrySeconds": float(next_retry_value) if next_retry_value is not None else None,
            "channels": list(self._feed_channels),
        }
        if latencies:
            stats_payload.update(
                {
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "avg_ms": sum(latencies) / len(latencies),
                    "p50_ms": statistics.median(latencies),
                    "p95_ms": self._percentile(latencies, 95.0),
                }
            )
        else:
            stats_payload.update({"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0})
        if "lastLatencyMs" in health:
            stats_payload["last_latency_ms"] = float(health["lastLatencyMs"])
            stats_payload["lastLatencyMs"] = float(health["lastLatencyMs"])
        latency_p95 = float(stats_payload.get("p95_ms", 0.0))
        stats_payload["latency_p95_ms"] = latency_p95
        stats_payload["p95_seconds"] = latency_p95 / 1000.0
        stats_payload["p95"] = latency_p95 / 1000.0
        reconnects_state = self.classify_threshold(
            float(self._feed_reconnects),
            warning=thresholds.get("reconnects_warning"),
            critical=thresholds.get("reconnects_critical"),
        )
        latency_state = self.classify_threshold(
            latency_p95,
            warning=thresholds.get("latency_warning_ms"),
            critical=thresholds.get("latency_critical_ms"),
        )
        downtime_seconds = max(0.0, downtime_value / 1000.0)
        downtime_state = self.classify_threshold(
            downtime_seconds,
            warning=thresholds.get("downtime_warning_seconds"),
            critical=thresholds.get("downtime_critical_seconds"),
        )
        sla_state = self.aggregate_sla_state((latency_state, reconnects_state, downtime_state))
        if sla_state in {"warning", "critical"}:
            self._consecutive_degraded_periods += 1
            self._consecutive_healthy_periods = 0
        else:
            self._consecutive_healthy_periods += 1
            self._consecutive_degraded_periods = 0
        stats_payload.update(
            {
                "latency_state": latency_state,
                "reconnects_state": reconnects_state,
                "downtime_state": downtime_state,
                "latency_warning_ms": thresholds.get("latency_warning_ms"),
                "latency_critical_ms": thresholds.get("latency_critical_ms"),
                "reconnects_warning": thresholds.get("reconnects_warning"),
                "reconnects_critical": thresholds.get("reconnects_critical"),
                "downtime_warning_seconds": thresholds.get("downtime_warning_seconds"),
                "downtime_critical_seconds": thresholds.get("downtime_critical_seconds"),
                "downtime_seconds": downtime_seconds,
                "sla_state": sla_state,
                "consecutive_degraded_periods": self._consecutive_degraded_periods,
                "consecutive_healthy_periods": self._consecutive_healthy_periods,
                "transport_source": transport_source,
                "transports": self.serialize_transport_stats(),
            }
        )
        self._feed_sla_report.clear()
        self._feed_sla_report.update(stats_payload)
        return self._feed_sla_report

    @staticmethod
    def classify_threshold(value: float | None, *, warning: float | None, critical: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "ok"
        if critical is not None and value >= critical:
            return "critical"
        if warning is not None and value >= warning:
            return "warning"
        return "ok"

    @staticmethod
    def aggregate_sla_state(states: Iterable[str]) -> str:
        bucket = {state for state in states if state}
        if "critical" in bucket:
            return "critical"
        if "warning" in bucket:
            return "warning"
        return "ok"
