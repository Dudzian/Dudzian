"""Guardrails monitorujące kolejkę I/O i limity zapytań adapterów."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

from .metrics import AsyncIOMetricSet


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
