"""Trading schedule utilities for AutoTrader runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping, Sequence

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None  # type: ignore


_DAY_NAME_MAP = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default if value is None else bool(value)


@dataclass(frozen=True)
class ScheduleWindow:
    """Single trading window definition.

    A window can optionally span midnight. When ``start == end`` the window
    covers the whole day.
    """

    start: time
    end: time
    mode: str = "live"
    allow_trading: bool = True
    days: frozenset[int] = field(default_factory=lambda: frozenset(range(7)))
    label: str | None = None

    def __post_init__(self) -> None:
        normalized_days = frozenset(int(day) % 7 for day in self.days)
        object.__setattr__(self, "days", normalized_days)

    @property
    def _start_seconds(self) -> int:
        return (
            self.start.hour * 3600
            + self.start.minute * 60
            + self.start.second
            + self.start.microsecond // 1_000_000
        )

    @property
    def _end_seconds(self) -> int:
        return (
            self.end.hour * 3600
            + self.end.minute * 60
            + self.end.second
            + self.end.microsecond // 1_000_000
        )

    @property
    def crosses_midnight(self) -> bool:
        if self.start == self.end:
            return False
        return self._end_seconds < self._start_seconds

    @property
    def duration(self) -> timedelta:
        if self.start == self.end:
            return timedelta(days=1)
        start_s = self._start_seconds
        end_s = self._end_seconds
        if self.crosses_midnight:
            seconds = 24 * 3600 - start_s + end_s
        else:
            seconds = max(0, end_s - start_s)
        return timedelta(seconds=seconds)

    def contains(self, moment: datetime) -> bool:
        """Return ``True`` if ``moment`` falls within this window."""

        weekday = moment.weekday()
        seconds = (
            moment.hour * 3600
            + moment.minute * 60
            + moment.second
            + moment.microsecond / 1_000_000
        )

        if self.start == self.end:
            return weekday in self.days

        if weekday in self.days and seconds >= self._start_seconds:
            if self.crosses_midnight:
                return True
            return seconds < self._end_seconds

        if not self.crosses_midnight:
            return False

        previous_day = (weekday - 1) % 7
        if previous_day not in self.days:
            return False
        return seconds < self._end_seconds

    def next_end(self, moment: datetime) -> datetime:
        """Return the moment when the current window ends."""

        if self.start == self.end:
            base_date = moment.date()
            return datetime.combine(base_date + timedelta(days=1), self.end, moment.tzinfo)

        start_seconds = self._start_seconds
        end_seconds = self._end_seconds
        current_seconds = (
            moment.hour * 3600
            + moment.minute * 60
            + moment.second
            + moment.microsecond / 1_000_000
        )
        base_date = moment.date()
        if self.crosses_midnight:
            if current_seconds >= start_seconds:
                base_date = base_date + timedelta(days=1)
            return datetime.combine(base_date, self.end, moment.tzinfo)
        return datetime.combine(base_date, self.end, moment.tzinfo)

    def start_datetime(self, base_date: datetime) -> datetime:
        return datetime.combine(base_date.date(), self.start, base_date.tzinfo)

    @staticmethod
    def _normalise_time(value: time | str) -> time:
        if isinstance(value, time):
            return value
        return time.fromisoformat(str(value))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ScheduleWindow":
        """Create a window definition from a mapping payload."""

        start_raw = payload.get("start", "00:00:00")
        end_raw = payload.get("end", "00:00:00")
        start = cls._normalise_time(start_raw)  # type: ignore[arg-type]
        end = cls._normalise_time(end_raw)  # type: ignore[arg-type]
        mode = str(payload.get("mode", "live"))
        allow_trading = _coerce_bool(payload.get("allow_trading", True), True)
        days_raw = payload.get("days")
        if days_raw is None:
            days_iter: Iterable[int] = range(7)
        else:
            if isinstance(days_raw, (str, bytes)):
                iterable = [days_raw]
            elif isinstance(days_raw, Iterable):
                iterable = list(days_raw)
            else:
                iterable = [days_raw]
            normalized_days: list[int] = []
            for item in iterable:
                if isinstance(item, int):
                    value = item
                else:
                    key = str(item).strip().lower()
                    value = _DAY_NAME_MAP.get(key)
                    if value is None:
                        try:
                            value = int(key)
                        except ValueError as exc:
                            raise ValueError(f"Invalid day value: {item!r}") from exc
                normalized_days.append(int(value) % 7)
            days_iter = normalized_days
        label = payload.get("label")
        if label is not None:
            label = str(label)
        return cls(start=start, end=end, mode=mode, allow_trading=allow_trading, days=frozenset(days_iter), label=label)

    def to_mapping(self) -> dict[str, object]:
        """Serialize the window into a JSON-friendly mapping."""

        payload: dict[str, object] = {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "mode": self.mode,
            "allow_trading": bool(self.allow_trading),
            "days": sorted(int(day) for day in self.days),
        }
        if self.label is not None:
            payload["label"] = self.label
        return payload


@dataclass(frozen=True)
class ScheduleOverride:
    """Ad-hoc override window that temporarily changes trading mode."""

    start: datetime
    end: datetime
    mode: str = "maintenance"
    allow_trading: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        start = self._ensure_datetime(self.start)
        end = self._ensure_datetime(self.end, reference=start)
        if end <= start:
            raise ValueError("Override end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @staticmethod
    def _ensure_datetime(moment: datetime, *, reference: datetime | None = None) -> datetime:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        if reference is not None:
            moment = moment.astimezone(reference.tzinfo)
        return moment

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def contains(self, moment: datetime) -> bool:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        moment = moment.astimezone(self.start.tzinfo)
        return self.start <= moment < self.end

    def to_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "mode": self.mode,
            "allow_trading": bool(self.allow_trading),
        }
        if self.label is not None:
            payload["label"] = self.label
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ScheduleOverride":
        start_raw = payload.get("start")
        end_raw = payload.get("end")
        if start_raw is None or end_raw is None:
            raise ValueError("Override payload requires start and end timestamps")
        start = datetime.fromisoformat(str(start_raw))
        end = datetime.fromisoformat(str(end_raw))
        mode = str(payload.get("mode", "maintenance"))
        allow_trading = _coerce_bool(payload.get("allow_trading", False), False)
        label_raw = payload.get("label")
        label = str(label_raw) if label_raw is not None else None
        return cls(start=start, end=end, mode=mode, allow_trading=allow_trading, label=label)


@dataclass(frozen=True)
class ScheduleState:
    """Snapshot describing the state of the trading schedule."""

    mode: str
    is_open: bool
    window: ScheduleWindow | None
    next_transition: datetime | None
    override: ScheduleOverride | None = None
    next_override: ScheduleOverride | None = None
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def time_until_transition(self) -> float | None:
        if self.next_transition is None:
            return None
        base = self.as_of.astimezone(self.next_transition.tzinfo)
        remaining = (self.next_transition - base).total_seconds()
        return max(0.0, remaining)

    @property
    def time_until_next_override(self) -> float | None:
        if self.next_override is None:
            return None
        base = self.as_of.astimezone(self.next_override.start.tzinfo)
        remaining = (self.next_override.start - base).total_seconds()
        return max(0.0, remaining)

    @property
    def override_active(self) -> bool:
        if self.override is None:
            return False
        return self.override.contains(self.as_of)


class TradingSchedule:
    """Determine the active trading mode based on time windows."""

    def __init__(
        self,
        windows: Sequence[ScheduleWindow],
        *,
        timezone_name: str | None = None,
        tz: timezone | None = None,
        default_mode: str = "demo",
        overrides: Sequence[ScheduleOverride] | None = None,
    ) -> None:
        if tz is None:
            if timezone_name is None:
                tz = timezone.utc
            else:
                tzinfo: timezone
                if ZoneInfo is not None:
                    tzinfo = ZoneInfo(timezone_name)
                else:  # pragma: no cover - fallback for legacy builds
                    tzinfo = timezone.utc
                tz = tzinfo
        self._tz = tz
        self._timezone_name = timezone_name
        self._windows = tuple(windows)
        self._default_mode = default_mode
        self._overrides = tuple(sorted(overrides or (), key=lambda item: item.start))

    @property
    def windows(self) -> tuple[ScheduleWindow, ...]:
        return self._windows

    @property
    def overrides(self) -> tuple[ScheduleOverride, ...]:
        return self._overrides

    @property
    def timezone(self) -> timezone:
        return self._tz

    @property
    def timezone_name(self) -> str | None:
        return self._timezone_name

    @property
    def default_mode(self) -> str:
        return self._default_mode

    def describe(self, now: datetime | None = None) -> ScheduleState:
        reference = self._normalize_datetime(now)
        base_mode, base_open, base_window, base_transition = self._describe_base(reference)
        override, next_override = self._resolve_overrides(reference)

        mode = base_mode
        is_open = base_open
        next_transition = base_transition

        if override is not None:
            mode = override.mode
            is_open = override.allow_trading
            next_transition = override.end.astimezone(self._tz)
        elif next_override is not None:
            start = next_override.start.astimezone(self._tz)
            if next_transition is None or start < next_transition:
                next_transition = start

        return ScheduleState(
            mode=mode,
            is_open=is_open,
            window=base_window,
            next_transition=next_transition,
            override=override,
            next_override=next_override,
            as_of=reference,
        )

    def _normalize_datetime(self, now: datetime | None) -> datetime:
        if now is None:
            now = datetime.now(self._tz)
        else:
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            now = now.astimezone(self._tz)
        return now

    def _describe_base(self, reference: datetime) -> tuple[str, bool, ScheduleWindow | None, datetime | None]:
        intervals = self._build_intervals(reference.date())

        for start, end, window in intervals:
            if start <= reference < end:
                return window.mode, window.allow_trading, window, end

        for start, _end, window in intervals:
            if start > reference:
                return self._default_mode, False, window, start

        return self._default_mode, False, None, None

    def _resolve_overrides(
        self, reference: datetime
    ) -> tuple[ScheduleOverride | None, ScheduleOverride | None]:
        active: ScheduleOverride | None = None
        upcoming: ScheduleOverride | None = None
        for override in self._overrides:
            if override.end <= reference:
                continue
            if override.contains(reference):
                active = override
                continue
            if override.start > reference and upcoming is None:
                upcoming = override
                break
        if active is not None and upcoming is None:
            for override in self._overrides:
                if override.start > reference:
                    upcoming = override
                    break
        return active, upcoming

    def _build_intervals(self, anchor_date: date) -> list[tuple[datetime, datetime, ScheduleWindow]]:
        intervals: list[tuple[datetime, datetime, ScheduleWindow]] = []
        start_date = anchor_date - timedelta(days=1)
        for offset in range(0, 10):
            day = start_date + timedelta(days=offset)
            weekday = day.weekday()
            day_dt = datetime.combine(day, time(0, 0), self._tz)
            for window in self._windows:
                if weekday not in window.days:
                    continue
                start_dt = window.start_datetime(day_dt)
                end_dt = start_dt + window.duration
                intervals.append((start_dt, end_dt, window))
        intervals.sort(key=lambda item: item[0])
        return intervals

    def replace_overrides(self, overrides: Sequence[ScheduleOverride]) -> "TradingSchedule":
        return TradingSchedule(
            self._windows,
            timezone_name=self._timezone_name,
            tz=self._tz,
            default_mode=self._default_mode,
            overrides=overrides,
        )

    def with_override(
        self,
        override: ScheduleOverride,
        *,
        replace_conflicts: bool = True,
    ) -> "TradingSchedule":
        overrides = list(self._overrides)
        if replace_conflicts:
            overrides = [
                existing
                for existing in overrides
                if not self._overrides_conflict(existing, override)
            ]
        overrides.append(override)
        return self.replace_overrides(overrides)

    def without_overrides(
        self,
        *,
        labels: Iterable[str | None] | None = None,
    ) -> "TradingSchedule":
        if labels is None:
            overrides: Sequence[ScheduleOverride] = ()
        else:
            normalized: set[str | None] = set()
            for label in labels:
                if label is None:
                    normalized.add(None)
                else:
                    normalized.add(str(label))
            overrides = [
                override
                for override in self._overrides
                if override.label not in normalized
            ]
        return self.replace_overrides(overrides)

    def list_overrides(
        self,
        *,
        labels: Iterable[str | None] | None = None,
        include_past: bool = False,
        now: datetime | None = None,
    ) -> tuple[ScheduleOverride, ...]:
        """Return overrides filtered by label and temporal status.

        Args:
            labels: Optional iterable of labels to include. ``None`` entries match
                unlabeled overrides. When omitted all overrides are considered.
            include_past: When ``False`` (default) only overrides whose ``end`` is
                in the future are returned.
            now: Optional reference time used for filtering past overrides.
                Falls back to ``datetime.now`` in the schedule timezone.
        """

        normalized_labels: set[str | None] | None
        if labels is None:
            normalized_labels = None
        else:
            normalized_labels = set()
            for label in labels:
                if label is None:
                    normalized_labels.add(None)
                else:
                    normalized_labels.add(str(label))

        reference: datetime | None
        if include_past:
            reference = None
        else:
            reference = self._normalize_datetime(now)

        filtered: list[ScheduleOverride] = []
        for override in self._overrides:
            if normalized_labels is not None and override.label not in normalized_labels:
                continue
            if reference is not None and override.end <= reference:
                continue
            filtered.append(override)
        return tuple(filtered)

    @staticmethod
    def _overrides_conflict(
        left: ScheduleOverride,
        right: ScheduleOverride,
    ) -> bool:
        if left.label is not None and right.label is not None and left.label == right.label:
            return True
        left_start = TradingSchedule._to_utc(left.start)
        left_end = TradingSchedule._to_utc(left.end)
        right_start = TradingSchedule._to_utc(right.start)
        right_end = TradingSchedule._to_utc(right.end)
        latest_start = max(left_start, right_start)
        earliest_end = min(left_end, right_end)
        return latest_start < earliest_end

    @staticmethod
    def _to_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    @classmethod
    def always_on(
        cls,
        *,
        mode: str = "live",
        timezone_name: str | None = None,
        tz: timezone | None = None,
    ) -> "TradingSchedule":
        window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode=mode, allow_trading=True)
        return cls((window,), timezone_name=timezone_name, tz=tz, default_mode=mode)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "default_mode": self._default_mode,
            "windows": [window.to_mapping() for window in self._windows],
            "overrides": [override.to_mapping() for override in self._overrides],
        }
        if self._timezone_name:
            payload["timezone"] = self._timezone_name
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TradingSchedule":
        timezone_raw = payload.get("timezone")
        timezone_name = str(timezone_raw) if isinstance(timezone_raw, str) else None
        default_mode = str(payload.get("default_mode", "demo"))
        windows_payload = payload.get("windows") or []
        if not isinstance(windows_payload, Iterable):
            raise TypeError("windows payload must be iterable")
        windows = [ScheduleWindow.from_mapping(item) for item in windows_payload]  # type: ignore[arg-type]
        overrides_payload = payload.get("overrides") or []
        if not isinstance(overrides_payload, Iterable):
            raise TypeError("overrides payload must be iterable")
        overrides = [
            ScheduleOverride.from_mapping(item) for item in overrides_payload  # type: ignore[arg-type]
        ]
        return cls(windows, timezone_name=timezone_name if timezone_name is not None else None, default_mode=default_mode, overrides=overrides)


__all__ = ["ScheduleOverride", "ScheduleWindow", "ScheduleState", "TradingSchedule"]
