"""Guardrails monitorujące kolejkę I/O, retraining i powiązane alerty."""
from __future__ import annotations

"""Guardrails monitorujące kolejkę I/O i retraining."""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

from .events import (
    DataDriftDetected,
    MissingDataDetected,
    MonitoringEvent,
    RetrainingCycleCompleted,
    RetrainingDelayInjected,
)
from .metrics import AsyncIOMetricSet, RetrainingMetricSet


@dataclass(slots=True)
class RateLimitWaitEvent:
    """Metadane zdarzenia oczekiwania na limiter kolejki."""

    key: str
    waited_seconds: float
    burst_limit: int
    pending_after: int


@dataclass(slots=True)
class TimeoutEvent:
    """Metadane timeoutu zgłoszonego przez kolejkę I/O."""

    key: str
    duration_seconds: float
    exception: BaseException


GuardrailUiNotifier = Callable[[str, Mapping[str, object]], None]


class AsyncIOGuardrails:
    """Subskrybuje kolejkę I/O i rejestruje zdarzenia w metrykach oraz logach."""

    def __init__(
        self,
        *,
        environment: str | None = None,
        metrics: AsyncIOMetricSet | None = None,
        log_directory: str | Path = "logs/guardrails",
        rate_limit_warning_threshold: float = 0.75,
        timeout_warning_threshold: float = 10.0,
        ui_alerts_path: Path | None = None,
        ui_notifier: GuardrailUiNotifier | None = None,
        retraining_metrics: RetrainingMetricSet | None = None,
        retraining_log_directory: str | Path | None = None,
        retraining_duration_warning_threshold: float = 300.0,
        drift_warning_threshold: float | None = None,
    ) -> None:
        self._environment = environment or "unknown"
        self._metrics = metrics or AsyncIOMetricSet()
        self._rate_limit_threshold = max(0.0, float(rate_limit_warning_threshold))
        self._timeout_threshold = max(0.0, float(timeout_warning_threshold))
        self._log_path = Path(log_directory)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"core.monitoring.guardrails[{self._log_path}]")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.FileHandler(self._log_path / "events.log", encoding="utf-8")
            handler.setLevel(logging.WARNING)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z")
            )
            self._logger.addHandler(handler)
        self._ui_alerts_path = ui_alerts_path
        self._ui_lock = threading.Lock()
        self._ui_notifier = ui_notifier
        self._rate_limit_streaks: MutableMapping[str, int] = {}
        self._timeout_streaks: MutableMapping[str, int] = {}
        self._retraining_metrics = retraining_metrics or RetrainingMetricSet()
        self._retraining_duration_threshold = max(0.0, float(retraining_duration_warning_threshold))
        self._drift_warning_threshold = None if drift_warning_threshold is None else max(
            0.0, float(drift_warning_threshold)
        )
        self._retraining_log_path = Path(
            retraining_log_directory or (self._log_path / "retraining")
        )
        self._retraining_log_path.mkdir(parents=True, exist_ok=True)
        self._retraining_logger = logging.getLogger(
            f"core.monitoring.guardrails.retraining[{self._retraining_log_path}]"
        )
        self._retraining_logger.setLevel(logging.INFO)
        self._retraining_logger.propagate = False
        if not self._retraining_logger.handlers:
            retraining_handler = logging.FileHandler(
                self._retraining_log_path / "events.log", encoding="utf-8"
            )
            retraining_handler.setLevel(logging.INFO)
            retraining_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z"
                )
            )
            self._retraining_logger.addHandler(retraining_handler)

    def on_rate_limit_wait(self, *, key: str, waited: float, burst: int, pending: int) -> None:
        """Obsługuje zdarzenie oczekiwania na limiter kolejki."""

        event = RateLimitWaitEvent(
            key=str(key),
            waited_seconds=float(waited),
            burst_limit=int(burst),
            pending_after=int(pending),
        )
        labels = {"queue": event.key, "environment": self._environment}
        self._metrics.rate_limit_wait_total.inc(labels=labels)
        self._metrics.rate_limit_wait_seconds.observe(event.waited_seconds, labels=labels)

        streak = self._rate_limit_streaks.get(event.key, 0)
        if event.waited_seconds >= self._rate_limit_threshold:
            streak += 1
        else:
            streak = 0
        self._rate_limit_streaks[event.key] = streak

        payload = {
            "queue": event.key,
            "environment": self._environment,
            "waited_seconds": round(event.waited_seconds, 6),
            "burst_limit": event.burst_limit,
            "pending_after": event.pending_after,
            "streak": streak,
        }

        if event.waited_seconds >= self._rate_limit_threshold:
            self._logger.warning(
                "RATE_LIMIT queue=%s waited=%.6fs streak=%s",
                event.key,
                event.waited_seconds,
                streak,
            )
            self._emit_ui_event("io_rate_limit_wait", "warning", payload)
        else:
            self._logger.info("rate_limit queue=%s waited=%.6fs", event.key, event.waited_seconds)

    def on_timeout(self, *, key: str, duration: float, exception: BaseException) -> None:
        """Obsługuje zdarzenie timeoutu zgłoszonego przez kolejkę."""

        event = TimeoutEvent(
            key=str(key),
            duration_seconds=float(duration),
            exception=exception,
        )
        labels = {"queue": event.key, "environment": self._environment}
        self._metrics.timeout_total.inc(labels=labels)
        self._metrics.timeout_duration.observe(event.duration_seconds, labels=labels)

        streak = self._timeout_streaks.get(event.key, 0) + 1
        self._timeout_streaks[event.key] = streak

        payload = {
            "queue": event.key,
            "environment": self._environment,
            "duration_seconds": round(event.duration_seconds, 6),
            "exception": type(event.exception).__name__,
            "streak": streak,
        }

        severity = "error" if event.duration_seconds >= self._timeout_threshold else "warning"
        log_method = self._logger.error if severity == "error" else self._logger.warning
        log_method(
            "TIMEOUT queue=%s duration=%.6fs exception=%s streak=%s",
            event.key,
            event.duration_seconds,
            type(event.exception).__name__,
            streak,
        )
        self._emit_ui_event("io_timeout", severity, payload)

    def __call__(self, event: MonitoringEvent) -> None:
        """Umożliwia traktowanie guardrail'i jako subskrybenta eventów retrainingu."""

        self.handle_monitoring_event(event)

    # pylint: disable=too-many-branches
    def handle_monitoring_event(self, event: MonitoringEvent) -> None:
        """Przetwarza zdarzenia monitorujące publikowane przez retraining scheduler."""

        if isinstance(event, RetrainingCycleCompleted):
            duration = max(0.0, float(event.duration_seconds))
            labels = {"environment": self._environment, "status": event.status}
            self._retraining_metrics.duration_seconds.observe(duration, labels=labels)

            severity = "info"
            if self._retraining_duration_threshold > 0 and duration >= self._retraining_duration_threshold:
                severity = "warning"

            payload = {
                "environment": self._environment,
                "status": event.status,
                "duration_seconds": round(duration, 6),
                "source": event.source,
            }
            if event.drift_score is not None:
                payload["drift_score"] = float(event.drift_score)
            if event.metadata:
                payload["metadata"] = dict(event.metadata)

            self._retraining_logger.log(
                logging.WARNING if severity == "warning" else logging.INFO,
                "RETRAINING duration=%.6fs status=%s source=%s",
                duration,
                event.status,
                event.source,
            )
            self._emit_ui_event("retraining_cycle_completed", severity, payload)

            if event.drift_score is not None:
                drift_labels = {"environment": self._environment, "status": event.status}
                self._retraining_metrics.drift_score.observe(event.drift_score, labels=drift_labels)
            return

        if isinstance(event, DataDriftDetected):
            drift_value = float(event.drift_score)
            labels = {"environment": self._environment, "source": event.source}
            self._retraining_metrics.drift_score.observe(drift_value, labels=labels)
            threshold = (
                self._drift_warning_threshold
                if self._drift_warning_threshold is not None
                else event.drift_threshold
            )
            severity = "warning" if threshold and drift_value >= threshold else "info"
            payload = {
                "environment": self._environment,
                "source": event.source,
                "drift_score": drift_value,
                "drift_threshold": event.drift_threshold,
            }
            self._retraining_logger.log(
                logging.WARNING if severity == "warning" else logging.INFO,
                "RETRAINING DRIFT source=%s score=%.6f threshold=%.6f",
                event.source,
                drift_value,
                event.drift_threshold,
            )
            self._emit_ui_event("retraining_drift_detected", severity, payload)
            return

        if isinstance(event, MissingDataDetected):
            payload = {
                "environment": self._environment,
                "source": event.source,
                "missing_batches": int(event.missing_batches),
            }
            self._retraining_logger.error(
                "RETRAINING MISSING_DATA source=%s missing_batches=%s",
                event.source,
                event.missing_batches,
            )
            self._emit_ui_event("retraining_missing_data", "error", payload)
            return

        if isinstance(event, RetrainingDelayInjected):
            payload = {
                "environment": self._environment,
                "reason": event.reason,
                "delay_seconds": round(float(event.delay_seconds), 6),
            }
            self._retraining_logger.info(
                "RETRAINING DELAY reason=%s delay_seconds=%.6f",
                event.reason,
                event.delay_seconds,
            )
            self._emit_ui_event("retraining_delay_injected", "info", payload)

    def _emit_ui_event(self, event: str, severity: str, payload: Mapping[str, object]) -> None:
        if self._ui_alerts_path is not None:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "severity": severity,
                "source": "core.monitoring.guardrails",
                "environment": self._environment,
                "payload": dict(payload),
            }
            with self._ui_lock:
                self._ui_alerts_path.parent.mkdir(parents=True, exist_ok=True)
                with self._ui_alerts_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self._ui_notifier is not None:
            try:
                self._ui_notifier(event, payload)
            except Exception:  # pragma: no cover - kanał UI jest opcjonalny
                logging.getLogger(__name__).debug("Nie udało się wysłać zdarzenia UI", exc_info=True)


__all__ = [
    "AsyncIOGuardrails",
    "GuardrailUiNotifier",
    "RateLimitWaitEvent",
    "TimeoutEvent",
]
