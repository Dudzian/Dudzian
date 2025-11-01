"""Harmonogram uruchamiania audytu zgodności."""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import yaml

from core.compliance import ComplianceAuditResult
from core.monitoring.events import ComplianceAuditCompleted, EventPublisher

_DEFAULT_SCHEDULE_PATH = Path("config/compliance/schedule.yml")


@dataclass(slots=True)
class ComplianceScheduleSettings:
    """Konfiguracja harmonogramu audytu zgodności."""

    enabled: bool = True
    interval: timedelta = timedelta(hours=24)
    window_start: time | None = None
    window_end: time | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ComplianceScheduleSettings":
        if not data:
            return cls()
        enabled = bool(data.get("enabled", True))
        interval = _parse_interval(data)
        window_cfg = data.get("window")
        start: time | None = None
        end: time | None = None
        if isinstance(window_cfg, Mapping):
            start = _parse_time(window_cfg.get("start"))
            end = _parse_time(window_cfg.get("end"))
        return cls(enabled=enabled, interval=interval, window_start=start, window_end=end)


@dataclass(slots=True)
class ComplianceAuditRunOutcome:
    """Wynik pojedynczego wywołania audytu przez harmonogram."""

    status: str
    result: ComplianceAuditResult | None
    reason: str | None
    events: tuple[object, ...]


class ComplianceScheduler:
    """Zarządza okresowym uruchamianiem audytu zgodności."""

    def __init__(
        self,
        *,
        settings: ComplianceScheduleSettings | None = None,
        clock: Callable[[], datetime] | None = None,
        event_publisher: EventPublisher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings or ComplianceScheduleSettings()
        if self._settings.interval <= timedelta(0):  # pragma: no cover - walidacja konfiguracji
            raise ValueError("Interwał audytu musi być dodatni")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._publish_event = event_publisher or (lambda event: None)
        self._logger = logger or logging.getLogger(__name__)
        self._last_run: datetime | None = None
        self._next_run: datetime | None = None

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def next_run(self) -> datetime | None:
        return self._next_run

    def should_run(self, *, now: datetime | None = None) -> bool:
        moment = now or self._clock()
        if not self._settings.enabled:
            return False
        if not self._within_window(moment):
            return False
        if self._next_run is None:
            return True
        return moment >= self._next_run

    async def run_once(
        self,
        callback: Callable[[], Awaitable[ComplianceAuditResult]],
    ) -> ComplianceAuditRunOutcome:
        if not self._settings.enabled:
            return ComplianceAuditRunOutcome(
                status="disabled",
                result=None,
                reason="disabled",
                events=(),
            )
        now = self._clock()
        if not self._within_window(now):
            self._next_run = self._next_window_start_after(now)
            self._logger.debug(
                "Audyt zgodności pominięty – poza oknem czasowym. Kolejne okno: %s", self._next_run
            )
            return ComplianceAuditRunOutcome(
                status="skipped",
                result=None,
                reason="outside_window",
                events=(),
            )
        if self._next_run is not None and now < self._next_run:
            return ComplianceAuditRunOutcome(
                status="skipped",
                result=None,
                reason="not_due",
                events=(),
            )

        result = await callback()
        self._last_run = now
        self._next_run = self._compute_next_run(now)

        severity_counts: dict[str, int] = {}
        for finding in result.findings:
            key = finding.severity.lower().strip() or "unknown"
            severity_counts[key] = severity_counts.get(key, 0) + 1
        completion_event = ComplianceAuditCompleted(
            passed=result.passed,
            findings_total=len(result.findings),
            severity_breakdown=severity_counts,
            config_path=str(result.config_path) if result.config_path else None,
        )
        self._publish_event(completion_event)
        return ComplianceAuditRunOutcome(
            status="completed",
            result=result,
            reason=None,
            events=(completion_event,),
        )

    def _compute_next_run(self, reference: datetime) -> datetime:
        candidate = reference + self._settings.interval
        if self._within_window(candidate):
            return candidate
        return self._next_window_start_after(candidate)

    def _within_window(self, timestamp: datetime) -> bool:
        start = self._settings.window_start
        end = self._settings.window_end
        if start is None and end is None:
            return True
        current = timestamp.timetz().replace(tzinfo=None, fold=0)
        if start is not None and end is not None:
            if start <= end:
                return start <= current <= end
            return current >= start or current <= end
        if start is not None:
            return current >= start
        if end is not None:
            return current <= end
        return True

    def _next_window_start_after(self, timestamp: datetime) -> datetime:
        tzinfo = timestamp.tzinfo or timezone.utc
        start = self._settings.window_start or time(0, 0)
        candidate = timestamp
        if self._settings.window_start is None and self._settings.window_end is not None:
            # okno od północy do end, więc następne otwarcie to kolejny dzień 00:00
            candidate = datetime.combine(timestamp.date(), time(0, 0), tzinfo=tzinfo)
            if candidate <= timestamp:
                candidate = datetime.combine(timestamp.date() + timedelta(days=1), time(0, 0), tzinfo=tzinfo)
            return candidate
        candidate = datetime.combine(timestamp.date(), start, tzinfo=tzinfo)
        if candidate <= timestamp:
            candidate = datetime.combine(timestamp.date() + timedelta(days=1), start, tzinfo=tzinfo)
        return candidate


def _parse_interval(data: Mapping[str, object]) -> timedelta:
    hours = float(data.get("interval_hours", 24.0))
    minutes = float(data.get("interval_minutes", 0.0))
    seconds = float(data.get("interval_seconds", 0.0))
    total = timedelta(hours=hours) + timedelta(minutes=minutes) + timedelta(seconds=seconds)
    if total <= timedelta(0):  # pragma: no cover - walidacja
        raise ValueError("Interwał audytu musi być dodatni")
    return total


def _parse_time(value: object | None) -> time | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Niepoprawny format czasu okna audytu: {value!r}")


def load_schedule_settings(path: Path | None = None) -> ComplianceScheduleSettings:
    """Wczytuje ustawienia harmonogramu z pliku YAML."""

    target = Path(path or _DEFAULT_SCHEDULE_PATH)
    if not target.exists():
        return ComplianceScheduleSettings()
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if payload is None:
        return ComplianceScheduleSettings()
    if not isinstance(payload, Mapping):  # pragma: no cover - ochrona przed błędnym formatem
        raise ValueError("Konfiguracja harmonogramu musi być mapą klucz→wartość")
    return ComplianceScheduleSettings.from_mapping(payload)


__all__ = [
    "ComplianceAuditRunOutcome",
    "ComplianceScheduleSettings",
    "ComplianceScheduler",
    "load_schedule_settings",
]
