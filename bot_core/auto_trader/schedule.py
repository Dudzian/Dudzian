"""Trading schedule utilities for AutoTrader runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping, Sequence

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None  # type: ignore


_WEEKDAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _parse_time(value: time | str) -> time:
    if isinstance(value, time):
        return value
    candidate = str(value).strip()
    if not candidate:
        raise ValueError("ScheduleWindow requires non-empty time values")
    try:
        return time.fromisoformat(candidate)
    except ValueError:
        pass
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(candidate, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Unsupported time format: {value!r}")


def _normalize_day(value: int | str) -> int:
    if isinstance(value, int):
        return int(value) % 7
    key = str(value).strip().lower()
    if not key:
        raise ValueError("ScheduleWindow day identifiers cannot be empty")
    if key.isdigit():
        return int(key) % 7
    try:
        return _WEEKDAY_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported weekday identifier: {value!r}") from exc


def _normalize_days(days: Iterable[int | str] | None) -> frozenset[int]:
    if days is None:
        return frozenset(range(7))
    return frozenset(_normalize_day(day) for day in days)


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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ScheduleWindow":
        start_raw = payload.get("start")
        if start_raw is None:
            raise ValueError("ScheduleWindow payload requires 'start' time")
        end_raw = payload.get("end", start_raw)
        mode_raw = payload.get("mode", "live")
        allow_trading_raw = payload.get("allow_trading", True)
        days_raw = payload.get("days")
        label_raw = payload.get("label")

        if isinstance(start_raw, time):
            start_time = start_raw
        else:
            start_time = _parse_time(str(start_raw))

        if end_raw is None:
            end_time = start_time
        elif isinstance(end_raw, time):
            end_time = end_raw
        else:
            end_time = _parse_time(str(end_raw))
        days: frozenset[int]
        if isinstance(days_raw, Iterable) and not isinstance(days_raw, (str, bytes, bytearray)):
            days = _normalize_days(days_raw)
        elif days_raw is None:
            days = _normalize_days(None)
        else:
            days = _normalize_days((days_raw,))

        label = None if label_raw is None else str(label_raw)
        return cls(
            start=start_time,
            end=end_time,
            mode=str(mode_raw),
            allow_trading=bool(allow_trading_raw),
            days=days,
            label=label,
        )

    def to_mapping(self) -> dict[str, object]:
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
    """Ad-hoc override changing the active trading mode."""

    start: datetime
    end: datetime
    mode: str = "live"
    allow_trading: bool = True
    label: str | None = None

    def __post_init__(self) -> None:
        start = self.start
        end = self.end
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start:
            raise ValueError("ScheduleOverride end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def is_active(self, moment: datetime) -> bool:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(self.start.tzinfo)
        return self.start <= moment < self.end

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        default_timezone: timezone | None = None,
    ) -> "ScheduleOverride":
        tz = default_timezone or timezone.utc
        start_raw = payload.get("start")
        end_raw = payload.get("end")
        if start_raw is None or end_raw is None:
            raise ValueError("ScheduleOverride requires start and end timestamps")
        start = _parse_datetime(start_raw, tz)
        end = _parse_datetime(end_raw, tz)
        mode_raw = payload.get("mode", "live")
        allow_raw = payload.get("allow_trading", True)
        label_raw = payload.get("label")
        label = None if label_raw is None else str(label_raw)
        return cls(start=start, end=end, mode=str(mode_raw), allow_trading=bool(allow_raw), label=label)

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


def _parse_datetime(value: object, default_timezone: timezone) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        candidate = str(value).strip()
        if not candidate:
            raise ValueError("Datetime value cannot be empty")
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
                else:
                    break
            else:
                raise
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_timezone)
    else:
        dt = dt.astimezone(default_timezone)
    return dt


@dataclass(frozen=True)
class ScheduleState:
    """Snapshot describing the state of the trading schedule."""

    mode: str
    is_open: bool
    window: ScheduleWindow | None
    next_transition: datetime | None
    override: ScheduleOverride | None = None
    next_override: ScheduleOverride | None = None
    override_active: bool = False

    @property
    def time_until_transition(self) -> float | None:
        if self.next_transition is None:
            return None
        now = datetime.now(self.next_transition.tzinfo)
        remaining = (self.next_transition - now).total_seconds()
        return max(0.0, remaining)

    @property
    def time_until_next_override(self) -> float | None:
        if self.next_override is None:
            return None
        now = datetime.now(self.next_override.start.tzinfo)
        remaining = (self.next_override.start - now).total_seconds()
        return max(0.0, remaining)


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
        self._timezone_name = timezone_name if isinstance(timezone_name, str) else None
        self._windows = tuple(windows)
        self._default_mode = default_mode
        normalized_overrides: list[ScheduleOverride] = []
        if overrides:
            for override in sorted(overrides, key=lambda item: item.start):
                normalized_overrides.append(
                    ScheduleOverride(
                        start=override.start.astimezone(self._tz),
                        end=override.end.astimezone(self._tz),
                        mode=override.mode,
                        allow_trading=override.allow_trading,
                        label=override.label,
                    )
                )
        self._overrides = tuple(normalized_overrides)

    @property
    def timezone(self) -> timezone:
        """Return timezone associated with the schedule."""

        return self._tz

    @property
    def timezone_name(self) -> str | None:
        return self._timezone_name

    @property
    def default_mode(self) -> str:
        return self._default_mode

    @property
    def windows(self) -> tuple[ScheduleWindow, ...]:
        return self._windows

    @property
    def overrides(self) -> tuple[ScheduleOverride, ...]:
        return self._overrides

    def with_overrides(
        self,
        overrides: Sequence[ScheduleOverride] | Sequence[Mapping[str, object]],
    ) -> "TradingSchedule":
        """Return a copy of the schedule with provided overrides."""

        normalized: list[ScheduleOverride] = []
        for override in overrides:
            if isinstance(override, Mapping):
                normalized.append(
                    ScheduleOverride.from_mapping(override, default_timezone=self._tz)
                )
            elif isinstance(override, ScheduleOverride):
                normalized.append(override)
            else:  # pragma: no cover - defensive branch
                raise TypeError(
                    "Overrides must be ScheduleOverride or mapping payloads"
                )
        return TradingSchedule(
            self._windows,
            timezone_name=self._timezone_name,
            tz=self._tz,
            default_mode=self._default_mode,
            overrides=normalized,
        )

    def describe(self, now: datetime | None = None) -> ScheduleState:
        reference = self._normalize_datetime(now)
        intervals = self._build_intervals(reference.date())

        active_window: tuple[datetime, datetime, ScheduleWindow] | None = None
        upcoming_window: tuple[datetime, datetime, ScheduleWindow] | None = None
        for start, end, window in intervals:
            if start <= reference < end:
                active_window = (start, end, window)
                break
            if start > reference and upcoming_window is None:
                upcoming_window = (start, end, window)

        active_override, next_override = self._locate_overrides(reference)

        if active_override is not None:
            next_transition = active_override.end
            window = active_window[2] if active_window else None
            state = ScheduleState(
                mode=active_override.mode,
                is_open=bool(active_override.allow_trading),
                window=window,
                next_transition=next_transition,
                override=active_override,
                next_override=next_override,
                override_active=True,
            )
        elif active_window is not None:
            start, end, window = active_window
            next_transition = end
            if next_override is not None and next_override.start < next_transition:
                next_transition = next_override.start
            state = ScheduleState(
                mode=window.mode,
                is_open=bool(window.allow_trading),
                window=window,
                next_transition=next_transition,
                override=None,
                next_override=next_override,
                override_active=False,
            )
        else:
            next_transition = None
            window = upcoming_window[2] if upcoming_window else None
            if upcoming_window is not None:
                next_transition = upcoming_window[0]
            if next_override is not None and (
                next_transition is None or next_override.start < next_transition
            ):
                next_transition = next_override.start
            state = ScheduleState(
                mode=self._default_mode,
                is_open=False,
                window=window,
                next_transition=next_transition,
                override=None,
                next_override=next_override,
                override_active=False,
            )
        return state

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

    def _locate_overrides(
        self, reference: datetime
    ) -> tuple[ScheduleOverride | None, ScheduleOverride | None]:
        active: ScheduleOverride | None = None
        upcoming: ScheduleOverride | None = None
        for idx, override in enumerate(self._overrides):
            if override.start <= reference < override.end:
                active = override
                for future in self._overrides[idx + 1 :]:
                    if future.start > reference:
                        upcoming = future
                        break
                break
            if override.start > reference:
                upcoming = override
                break
        if active is None and upcoming is None:
            # reference may be after all overrides; nothing to report
            return None, None
        return active, upcoming

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

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TradingSchedule":
        windows_payload = payload.get("windows")
        if not isinstance(windows_payload, Iterable):
            raise TypeError("TradingSchedule payload requires iterable 'windows'")
        window_objs: list[ScheduleWindow] = []
        for entry in windows_payload:
            if not isinstance(entry, Mapping):
                raise TypeError("TradingSchedule windows must be mappings")
            window_objs.append(ScheduleWindow.from_mapping(entry))
        if not window_objs:
            raise ValueError("TradingSchedule requires at least one window")

        timezone_name = payload.get("timezone")
        tz: timezone | None = None
        if isinstance(timezone_name, str) and timezone_name.strip():
            if ZoneInfo is not None:
                tz = ZoneInfo(timezone_name)
            else:  # pragma: no cover - legacy builds without zoneinfo
                tz = timezone.utc
        default_mode = str(payload.get("default_mode", "demo"))

        overrides_payload = payload.get("overrides")
        overrides: list[ScheduleOverride] = []
        if isinstance(overrides_payload, Iterable):
            for entry in overrides_payload:
                if not isinstance(entry, Mapping):
                    raise TypeError("Schedule overrides must be mappings")
                overrides.append(
                    ScheduleOverride.from_mapping(
                        entry,
                        default_timezone=tz or timezone.utc,
                    )
                )

        return cls(
            window_objs,
            timezone_name=timezone_name if isinstance(timezone_name, str) else None,
            tz=tz,
            default_mode=default_mode,
            overrides=overrides,
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "default_mode": self._default_mode,
            "windows": [window.to_mapping() for window in self._windows],
        }
        if self._timezone_name:
            payload["timezone"] = self._timezone_name
        if self._overrides:
            payload["overrides"] = [override.to_mapping() for override in self._overrides]
        return payload


__all__ = [
    "ScheduleOverride",
    "ScheduleWindow",
    "ScheduleState",
    "TradingSchedule",
]
