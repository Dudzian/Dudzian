"""Trading schedule utilities for AutoTrader runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Iterable, Mapping, MutableMapping, Sequence

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None  # type: ignore


_DAY_ALIASES: Mapping[str, int] = {
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


def _parse_time(value: object) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        try:
            parsed = time.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - validation guard
            raise ValueError(f"Invalid time value '{value}'") from exc
        return parsed
    if isinstance(value, Mapping):  # pragma: no cover - backwards compatibility
        return _parse_time(value.get("value"))
    raise TypeError(f"Unsupported time representation: {type(value)!r}")


def _parse_day_tokens(days: Iterable[object]) -> frozenset[int]:
    normalized: set[int] = set()
    for candidate in days:
        if isinstance(candidate, int):
            normalized.add(candidate % 7)
            continue
        if isinstance(candidate, str):
            token = candidate.strip().lower()
            if not token:
                continue
            if token.isdigit():
                normalized.add(int(token) % 7)
                continue
            mapped = _DAY_ALIASES.get(token)
            if mapped is not None:
                normalized.add(mapped)
                continue
        raise ValueError(f"Unsupported day token: {candidate!r}")
    return frozenset(normalized)


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

    def to_mapping(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "start": self.start.isoformat(timespec="seconds"),
            "end": self.end.isoformat(timespec="seconds"),
            "mode": self.mode,
            "allow_trading": bool(self.allow_trading),
            "days": sorted(int(day) for day in self.days),
        }
        if self.label is not None:
            payload["label"] = self.label
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ScheduleWindow":
        if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
            raise TypeError("ScheduleWindow.from_mapping requires a mapping payload")
        start = _parse_time(payload.get("start", "00:00"))
        end = _parse_time(payload.get("end", "00:00"))
        mode = str(payload.get("mode", "live"))
        allow_trading = bool(payload.get("allow_trading", True))
        days_payload = payload.get("days")
        if days_payload is None:
            days = frozenset(range(7))
        elif isinstance(days_payload, (list, tuple, set, frozenset)):
            days = _parse_day_tokens(days_payload)
        else:
            days = _parse_day_tokens((days_payload,))
        label_obj = payload.get("label")
        label = str(label_obj) if isinstance(label_obj, str) else None
        return cls(start=start, end=end, mode=mode, allow_trading=allow_trading, days=days, label=label)


@dataclass(frozen=True)
class ScheduleOverride:
    """Temporal override forcing a specific trading mode for a time range."""

    start: datetime
    end: datetime
    mode: str
    allow_trading: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        start = self.start
        end = self.end
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("ScheduleOverride requires timezone-aware datetimes")
        if end <= start:
            raise ValueError("ScheduleOverride end must be after start")

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def contains(self, moment: datetime) -> bool:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return self.start <= moment.astimezone(self.start.tzinfo) < self.end

    def to_mapping(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "start": self.start.astimezone(timezone.utc).isoformat(),
            "end": self.end.astimezone(timezone.utc).isoformat(),
            "mode": self.mode,
            "allow_trading": bool(self.allow_trading),
        }
        if self.label is not None:
            payload["label"] = self.label
        return payload

    @staticmethod
    def _parse_datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            candidate = value
        elif isinstance(value, (int, float)):
            candidate = datetime.fromtimestamp(float(value), tz=timezone.utc)
        elif isinstance(value, str):
            try:
                candidate = datetime.fromisoformat(value)
            except ValueError as exc:  # pragma: no cover - validation guard
                raise ValueError(f"Invalid datetime value '{value}'") from exc
        else:
            raise TypeError(f"Unsupported datetime representation: {type(value)!r}")
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)
        return candidate

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ScheduleOverride":
        if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
            raise TypeError("ScheduleOverride.from_mapping requires a mapping payload")
        start = cls._parse_datetime(payload.get("start"))
        end = cls._parse_datetime(payload.get("end"))
        mode = str(payload.get("mode", "maintenance"))
        allow_trading = bool(payload.get("allow_trading", False))
        label_obj = payload.get("label")
        label = str(label_obj) if isinstance(label_obj, str) else None
        return cls(start=start, end=end, mode=mode, allow_trading=allow_trading, label=label)


@dataclass(frozen=True)
class ScheduleState:
    """Snapshot describing the state of the trading schedule."""

    mode: str
    is_open: bool
    window: ScheduleWindow | None
    next_transition: datetime | None
    reference_time: datetime
    override: ScheduleOverride | None = None
    next_override: ScheduleOverride | None = None
    override_active: bool = False

    @property
    def time_until_transition(self) -> float | None:
        if self.next_transition is None:
            return None
        remaining = (self.next_transition - self.reference_time).total_seconds()
        return max(0.0, remaining)

    @property
    def time_until_next_override(self) -> float | None:
        if self.next_override is None:
            return None
        remaining = (self.next_override.start - self.reference_time).total_seconds()
        return max(0.0, remaining)


class TradingSchedule:
    """Determine the active trading mode based on time windows."""

    def __init__(
        self,
        windows: Sequence[ScheduleWindow],
        *,
        timezone_name: str | None = None,
        tz: tzinfo | None = None,
        default_mode: str = "demo",
        overrides: Sequence[ScheduleOverride] | None = None,
    ) -> None:
        if tz is None:
            if timezone_name is None:
                tz = timezone.utc
            else:
                tzinfo_obj: tzinfo
                if ZoneInfo is not None:
                    tzinfo_obj = ZoneInfo(timezone_name)
                else:  # pragma: no cover - fallback for legacy builds
                    tzinfo_obj = timezone.utc
                tz = tzinfo_obj
        self._tz = tz
        if timezone_name:
            self._timezone_name = timezone_name
        else:
            tz_key = getattr(self._tz, "key", None)
            tz_zone = getattr(self._tz, "zone", None)
            if isinstance(tz_key, str) and tz_key:
                self._timezone_name = tz_key
            elif isinstance(tz_zone, str) and tz_zone:
                self._timezone_name = tz_zone
            else:
                snapshot = datetime.now(timezone.utc).astimezone(self._tz)
                self._timezone_name = snapshot.tzname() or "UTC"
        self._windows = tuple(windows)
        self._default_mode = default_mode
        overrides_iter = overrides or ()
        normalized_overrides = []
        for override in overrides_iter:
            normalized_overrides.append(
                ScheduleOverride(
                    start=override.start.astimezone(self._tz),
                    end=override.end.astimezone(self._tz),
                    mode=override.mode,
                    allow_trading=override.allow_trading,
                    label=override.label,
                )
            )
        normalized_overrides.sort(key=lambda item: item.start)
        self._overrides: tuple[ScheduleOverride, ...] = tuple(normalized_overrides)

    def describe(self, now: datetime | None = None) -> ScheduleState:
        reference = self._normalize_datetime(now)
        intervals = self._build_intervals(reference.date())

        active_window: ScheduleWindow | None = None
        next_window_transition: datetime | None = None
        next_window: ScheduleWindow | None = None

        for start, end, window in intervals:
            if start <= reference < end:
                active_window = window
                next_window_transition = end
                break
            if start > reference and next_window is None:
                next_window = window
                next_window_transition = start
                break

        base_mode = active_window.mode if active_window is not None else self._default_mode
        base_open = active_window.allow_trading if active_window is not None else False
        base_window = active_window if active_window is not None else next_window

        active_override: ScheduleOverride | None = None
        next_override: ScheduleOverride | None = None
        for override in self._overrides:
            if override.end <= reference:
                continue
            if override.start <= reference < override.end:
                active_override = override
                continue
            if override.start > reference:
                next_override = override
                break

        if active_override is not None:
            mode = active_override.mode
            is_open = active_override.allow_trading
            next_transition = active_override.end
            override_active = True
        else:
            mode = base_mode
            is_open = base_open
            next_transition = next_window_transition
            override_active = False
            if next_override is not None:
                if next_transition is None or next_override.start < next_transition:
                    next_transition = next_override.start

        return ScheduleState(
            mode=mode,
            is_open=is_open,
            window=base_window,
            next_transition=next_transition,
            reference_time=reference,
            override=active_override,
            next_override=next_override,
            override_active=override_active,
        )

    def _normalize_datetime(self, now: datetime | None) -> datetime:
        if now is None:
            now = datetime.now(self._tz)
        else:
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            now = now.astimezone(self._tz)
        return now

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

    @classmethod
    def always_on(
        cls,
        *,
        mode: str = "live",
        timezone_name: str | None = None,
        tz: tzinfo | None = None,
    ) -> "TradingSchedule":
        window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode=mode, allow_trading=True)
        return cls((window,), timezone_name=timezone_name, tz=tz, default_mode=mode)

    def to_payload(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "default_mode": self._default_mode,
            "windows": [window.to_mapping() for window in self._windows],
            "overrides": [override.to_mapping() for override in self._overrides],
        }
        if self._timezone_name:
            payload["timezone"] = self._timezone_name
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TradingSchedule":
        if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
            raise TypeError("TradingSchedule.from_payload requires a mapping payload")
        windows_payload = payload.get("windows", ())
        if not isinstance(windows_payload, Iterable):  # pragma: no cover - validation guard
            raise TypeError("TradingSchedule windows payload must be iterable")
        windows = [ScheduleWindow.from_mapping(entry) for entry in windows_payload]
        overrides_payload = payload.get("overrides", ())
        if not isinstance(overrides_payload, Iterable):  # pragma: no cover - validation guard
            raise TypeError("TradingSchedule overrides payload must be iterable")
        overrides = [ScheduleOverride.from_mapping(entry) for entry in overrides_payload]
        timezone_name = payload.get("timezone")
        tz_obj: tzinfo | None = None
        if timezone_name is not None and not isinstance(timezone_name, str):  # pragma: no cover
            timezone_name = str(timezone_name)
        if timezone_name is None and overrides:
            tz_obj = overrides[0].start.tzinfo
        default_mode = str(payload.get("default_mode", "demo"))
        return cls(
            windows,
            timezone_name=timezone_name if isinstance(timezone_name, str) else None,
            tz=tz_obj,
            default_mode=default_mode,
            overrides=overrides,
        )


__all__ = ["ScheduleWindow", "ScheduleState", "TradingSchedule", "ScheduleOverride"]
