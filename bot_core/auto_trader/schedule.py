"""Trading schedule utilities for AutoTrader runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None  # type: ignore


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


@dataclass(frozen=True)
class ScheduleState:
    """Snapshot describing the state of the trading schedule."""

    mode: str
    is_open: bool
    window: ScheduleWindow | None
    next_transition: datetime | None

    @property
    def time_until_transition(self) -> float | None:
        if self.next_transition is None:
            return None
        now = datetime.now(self.next_transition.tzinfo)
        remaining = (self.next_transition - now).total_seconds()
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
        self._windows = tuple(windows)
        self._default_mode = default_mode

    def describe(self, now: datetime | None = None) -> ScheduleState:
        reference = self._normalize_datetime(now)
        intervals = self._build_intervals(reference.date())

        for start, end, window in intervals:
            if start <= reference < end:
                return ScheduleState(
                    mode=window.mode,
                    is_open=window.allow_trading,
                    window=window,
                    next_transition=end,
                )

        for start, _end, window in intervals:
            if start > reference:
                return ScheduleState(
                    mode=self._default_mode,
                    is_open=False,
                    window=window,
                    next_transition=start,
                )

        return ScheduleState(
            mode=self._default_mode,
            is_open=False,
            window=None,
            next_transition=None,
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
        tz: timezone | None = None,
    ) -> "TradingSchedule":
        window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode=mode, allow_trading=True)
        return cls((window,), timezone_name=timezone_name, tz=tz, default_mode=mode)


__all__ = ["ScheduleWindow", "ScheduleState", "TradingSchedule"]
