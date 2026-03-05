"""Wspólny sink metryk i alertów ryzyka używany przez silnik i warstwę UI."""

from __future__ import annotations

from copy import deepcopy
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, MutableMapping

from bot_core.alerts import AlertSeverity
from bot_core.risk.events import RiskAlertLog


@dataclass(slots=True)
class RiskAlertSink:
    """Bufor alertów ryzyka możliwy do ponownego wykorzystania przez UI/telemetrię."""

    _log: RiskAlertLog = field(default_factory=RiskAlertLog)

    def record(
        self,
        *,
        profile: str,
        limit: str,
        value: float,
        threshold: float,
        severity: str = "warning",
        context: Mapping[str, object] | None = None,
    ) -> None:
        self._log.record(
            profile=profile,
            limit=limit,
            value=value,
            threshold=threshold,
            severity=severity,
            context=context,
        )

    def tail(self, *, profile: str | None = None, limit: int = 20):
        return self._log.tail(profile=profile, limit=limit)

    @property
    def log(self) -> RiskAlertLog:
        return self._log


@dataclass(slots=True)
class RiskMetricsSink:
    """Trzyma ostatnio opublikowane snapshoty metryk ryzyka per profil."""

    _latest: MutableMapping[str, Mapping[str, object]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def publish_snapshot(
        self,
        profile: str,
        snapshot: Mapping[str, object],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {"snapshot": dict(snapshot)}
        if metadata:
            payload["metadata"] = dict(metadata)
        payload.setdefault("published_at", datetime.utcnow().isoformat())
        with self._lock:
            self._latest[profile] = payload

    def latest(self, profile: str) -> Mapping[str, object] | None:
        with self._lock:
            payload = self._latest.get(profile)
        if payload is None:
            return None
        return deepcopy(payload)


@dataclass(slots=True)
class RiskObservabilitySink:
    """Łączy sink alertów i metryk, aby ujednolicić publikację do obserwowalności/UI."""

    metrics: RiskMetricsSink = field(default_factory=RiskMetricsSink)
    alerts: RiskAlertSink = field(default_factory=RiskAlertSink)

    def record_alert(
        self,
        *,
        profile: str,
        limit: str,
        value: float,
        threshold: float,
        severity: str | AlertSeverity = "warning",
        context: Mapping[str, object] | None = None,
    ) -> None:
        normalized_severity = severity.value if isinstance(severity, AlertSeverity) else severity
        self.alerts.record(
            profile=profile,
            limit=limit,
            value=value,
            threshold=threshold,
            severity=str(normalized_severity or "warning"),
            context=context,
        )

    def publish_snapshot(
        self,
        profile: str,
        snapshot: Mapping[str, object],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.metrics.publish_snapshot(profile, snapshot, metadata=metadata or {})

    @property
    def alert_log(self) -> RiskAlertLog:
        return self.alerts.log
