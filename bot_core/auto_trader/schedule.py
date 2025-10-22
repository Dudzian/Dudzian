"""Trading schedule utilities for AutoTrader runtime."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

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


def _parse_time(value: Any, *, field_name: str) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        try:
            return time.fromisoformat(text)
        except ValueError:
            for fmt in ("%H:%M", "%H:%M:%S"):
                try:
                    return datetime.strptime(text, fmt).time()
                except ValueError:
                    continue
        raise ValueError(f"Invalid time format for {field_name!r}: {value!r}")
    raise TypeError(f"{field_name} must be a datetime.time or ISO formatted string")


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError as exc:  # pragma: no cover - error details bubbled up
            raise ValueError(f"Invalid datetime format for {field_name!r}: {value!r}") from exc
    raise TypeError(f"{field_name} must be a datetime.datetime or ISO formatted string")


def _parse_days(days: Iterable[Any] | None) -> frozenset[int]:
    if days is None:
        return frozenset(range(7))
    normalized: set[int] = set()
    for item in days:
        if isinstance(item, int):
            normalized.add(int(item) % 7)
            continue
        if isinstance(item, str):
            token = item.strip().lower()
            if not token:
                continue
            if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
                normalized.add(int(token) % 7)
                continue
            alias = _DAY_ALIASES.get(token)
            if alias is not None:
                normalized.add(alias)
                continue
        raise TypeError("days entries must be integers or day names")
    if not normalized:
        raise ValueError("days must not be empty")
    return frozenset(normalized)


def _coerce_bool(value: Any, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean value")


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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScheduleWindow":
        """Build a schedule window from a mapping configuration."""

        if "start" not in payload or "end" not in payload:
            raise KeyError("schedule window payload must include 'start' and 'end'")
        start = _parse_time(payload["start"], field_name="start")
        end = _parse_time(payload["end"], field_name="end")
        mode = str(payload.get("mode", "live") or "live")
        allow_trading = _coerce_bool(
            payload.get("allow_trading"),
            default=True,
            field_name="allow_trading",
        )
        label_raw = payload.get("label")
        label = str(label_raw) if label_raw is not None else None
        days = _parse_days(payload.get("days"))
        return cls(
            start=start,
            end=end,
            mode=mode,
            allow_trading=allow_trading,
            days=days,
            label=label,
        )

    def to_mapping(self, *, include_duration: bool = False) -> MutableMapping[str, Any]:
        """Serialise window definition to a mapping."""

        payload: MutableMapping[str, Any] = {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "mode": self.mode,
            "allow_trading": bool(self.allow_trading),
            "days": sorted(int(day) for day in self.days),
        }
        if include_duration:
            payload["duration_s"] = int(self.duration.total_seconds())
        if self.label is not None:
            payload["label"] = self.label
        return payload

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
class ScheduleOverride:
    """One-off override that temporarily replaces the base schedule."""

    start: datetime
    end: datetime
    mode: str = "live"
    allow_trading: bool = True
    label: str | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("override end must be after start")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScheduleOverride":
        if "start" not in payload or "end" not in payload:
            raise KeyError("schedule override payload must include 'start' and 'end'")
        start = _parse_datetime(payload["start"], field_name="start")
        end = _parse_datetime(payload["end"], field_name="end")
        mode = str(payload.get("mode", "live") or "live")
        allow_trading = _coerce_bool(
            payload.get("allow_trading"),
            default=True,
            field_name="allow_trading",
        )
        label_raw = payload.get("label")
        label = str(label_raw) if label_raw is not None else None
        return cls(start=start, end=end, mode=mode, allow_trading=allow_trading, label=label)

    def to_mapping(
        self,
        *,
        include_duration: bool = False,
        timezone_hint: timezone | None = None,
    ) -> MutableMapping[str, Any]:
        start = self.start.astimezone(timezone_hint) if timezone_hint else self.start
        end = self.end.astimezone(timezone_hint) if timezone_hint else self.end
        payload: MutableMapping[str, Any] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "mode": self.mode,
            "allow_trading": bool(self.allow_trading),
        }
        if self.label is not None:
            payload["label"] = self.label
        if include_duration:
            payload["duration_s"] = int(self.duration.total_seconds())
        return payload

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass(frozen=True)
class ScheduleState:
    """Snapshot describing the state of the trading schedule."""

    mode: str
    is_open: bool
    window: ScheduleWindow | None
    next_transition: datetime | None
    override: ScheduleOverride | None = None
    next_override: ScheduleOverride | None = None

    @property
    def time_until_transition(self) -> float | None:
        if self.next_transition is None:
            return None
        now = datetime.now(self.next_transition.tzinfo)
        remaining = (self.next_transition - now).total_seconds()
        return max(0.0, remaining)

    @property
    def override_active(self) -> bool:
        return self.override is not None

    @property
    def time_until_next_override(self) -> float | None:
        if self.next_override is None:
            return None
        now = datetime.now(self.next_override.start.tzinfo)
        remaining = (self.next_override.start - now).total_seconds()
        return max(0.0, remaining)

    def to_mapping(
        self,
        *,
        timezone_hint: timezone | None = timezone.utc,
        include_remaining: bool = True,
    ) -> MutableMapping[str, Any]:
        """Serialise the state to a mapping suitable for JSON payloads."""

        payload: MutableMapping[str, Any] = {
            "mode": self.mode,
            "is_open": bool(self.is_open),
            "override_active": self.override_active,
        }
        if self.window is not None:
            payload["window"] = self.window.to_mapping(include_duration=True)
        if self.next_transition is not None:
            reference = (
                self.next_transition.astimezone(timezone_hint)
                if timezone_hint is not None
                else self.next_transition
            )
            payload["next_transition"] = reference.isoformat()
        if self.override is not None:
            payload["override"] = self.override.to_mapping(
                include_duration=True,
                timezone_hint=timezone_hint,
            )
        if self.next_override is not None:
            payload["next_override"] = self.next_override.to_mapping(
                include_duration=True,
                timezone_hint=timezone_hint,
            )
        if include_remaining:
            remaining = self.time_until_transition
            if remaining is not None:
                payload["time_until_transition_s"] = remaining
            next_override_delay = self.time_until_next_override
            if next_override_delay is not None:
                payload["time_until_next_override_s"] = next_override_delay
        return payload


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
        self._windows = tuple(windows)
        self._default_mode = default_mode
        self._overrides = self._normalise_overrides(overrides or ())

    @property
    def windows(self) -> tuple[ScheduleWindow, ...]:
        """Return immutable access to configured schedule windows."""

        return self._windows

    @property
    def overrides(self) -> tuple[ScheduleOverride, ...]:
        """Return immutable access to registered overrides."""

        return self._overrides

    def describe(self, now: datetime | None = None) -> ScheduleState:
        reference = self._normalize_datetime(now)
        intervals = self._build_intervals(reference.date())

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

        for start, end, window in intervals:
            if start <= reference < end:
                if active_override is not None:
                    return ScheduleState(
                        mode=active_override.mode,
                        is_open=active_override.allow_trading,
                        window=window,
                        next_transition=active_override.end,
                        override=active_override,
                        next_override=next_override,
                    )
                next_transition = end
                if next_override is not None and next_override.start < end:
                    next_transition = next_override.start
                return ScheduleState(
                    mode=window.mode,
                    is_open=window.allow_trading,
                    window=window,
                    next_transition=next_transition,
                    override=None,
                    next_override=next_override,
                )

        for start, _end, window in intervals:
            if start > reference:
                if active_override is not None:
                    return ScheduleState(
                        mode=active_override.mode,
                        is_open=active_override.allow_trading,
                        window=window,
                        next_transition=active_override.end,
                        override=active_override,
                        next_override=next_override,
                    )
                next_transition = start
                if next_override is not None and next_override.start < next_transition:
                    next_transition = next_override.start
                return ScheduleState(
                    mode=self._default_mode,
                    is_open=False,
                    window=window,
                    next_transition=next_transition,
                    override=None,
                    next_override=next_override,
                )

        if active_override is not None:
            next_transition = active_override.end
            return ScheduleState(
                mode=active_override.mode,
                is_open=active_override.allow_trading,
                window=None,
                next_transition=next_transition,
                override=active_override,
                next_override=next_override,
            )

        next_transition = None
        if next_override is not None:
            next_transition = next_override.start
        return ScheduleState(
            mode=self._default_mode,
            is_open=False,
            window=None,
            next_transition=next_transition,
            override=None,
            next_override=next_override,
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | Sequence[Any],
        *,
        timezone_name: str | None = None,
        tz: timezone | None = None,
        default_mode: str | None = None,
        overrides: Sequence[ScheduleOverride] | None = None,
    ) -> "TradingSchedule":
        """Create a schedule from a serialised representation."""

        if isinstance(payload, TradingSchedule):
            return payload

        windows_payload: Sequence[Any]
        schedule_timezone = timezone_name
        schedule_tz = tz
        schedule_default_mode = default_mode
        overrides_payload: Sequence[Any] | None = None

        if isinstance(payload, Mapping):
            windows_payload = payload.get("windows")  # type: ignore[assignment]
            if windows_payload is None:
                raise ValueError("schedule payload must define 'windows'")
            if "timezone" in payload:
                schedule_timezone = payload.get("timezone") or schedule_timezone
            if "timezone_name" in payload:
                schedule_timezone = payload.get("timezone_name") or schedule_timezone
            if "tz" in payload and isinstance(payload.get("tz"), timezone):
                schedule_tz = payload.get("tz")  # type: ignore[assignment]
            if "default_mode" in payload:
                schedule_default_mode = payload.get("default_mode") or schedule_default_mode
            overrides_payload = payload.get("overrides")  # type: ignore[assignment]
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            windows_payload = payload
        else:
            raise TypeError("payload must be a mapping or a sequence of windows")

        windows: list[ScheduleWindow] = []
        for item in windows_payload:
            if isinstance(item, ScheduleWindow):
                windows.append(item)
            elif isinstance(item, Mapping):
                windows.append(ScheduleWindow.from_mapping(item))
            else:
                raise TypeError("windows must be mappings or ScheduleWindow instances")

        schedule_overrides: list[ScheduleOverride] = []
        combined_overrides: Sequence[Any] | None = overrides or overrides_payload
        if combined_overrides is not None:
            for item in combined_overrides:
                if isinstance(item, ScheduleOverride):
                    schedule_overrides.append(item)
                elif isinstance(item, Mapping):
                    schedule_overrides.append(ScheduleOverride.from_mapping(item))
                else:
                    raise TypeError("overrides must be mappings or ScheduleOverride instances")

        return cls(
            windows,
            timezone_name=schedule_timezone,
            tz=schedule_tz,
            default_mode=str(schedule_default_mode or "demo"),
            overrides=tuple(schedule_overrides),
        )

    def to_payload(self) -> MutableMapping[str, Any]:
        """Serialise schedule definition to a mapping."""

        payload: MutableMapping[str, Any] = {
            "default_mode": self._default_mode,
            "windows": [window.to_mapping() for window in self._windows],
        }
        tz_name = getattr(self._tz, "key", None) or getattr(self._tz, "zone", None)
        if not tz_name and hasattr(self._tz, "tzname"):
            tz_name = self._tz.tzname(None)
        if tz_name:
            payload["timezone"] = tz_name
        if self._overrides:
            payload["overrides"] = [override.to_mapping() for override in self._overrides]
        return payload

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

    def _normalise_overrides(
        self, overrides: Sequence[ScheduleOverride]
    ) -> tuple[ScheduleOverride, ...]:
        normalised: list[ScheduleOverride] = []
        for override in overrides:
            if not isinstance(override, ScheduleOverride):
                raise TypeError("overrides must be ScheduleOverride instances")
            start = self._normalize_datetime(override.start)
            end = self._normalize_datetime(override.end)
            if end <= start:
                raise ValueError("override end must be after start")
            normalised.append(replace(override, start=start, end=end))
        normalised.sort(key=lambda item: item.start)
        return tuple(normalised)

    def with_overrides(
        self, overrides: Sequence[ScheduleOverride] | None
    ) -> "TradingSchedule":
        """Return a copy of the schedule with a new set of overrides."""

        return TradingSchedule(
            self._windows,
            tz=self._tz,
            default_mode=self._default_mode,
            overrides=tuple(overrides or ()),
        )

    @classmethod
    def always_on(
        cls,
        *,
        mode: str = "live",
        timezone_name: str | None = None,
        tz: timezone | None = None,
        overrides: Sequence[ScheduleOverride] | None = None,
    ) -> "TradingSchedule":
        window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode=mode, allow_trading=True)
        return cls(
            (window,),
            timezone_name=timezone_name,
            tz=tz,
            default_mode=mode,
            overrides=overrides,
        )


__all__ = ["ScheduleWindow", "ScheduleState", "TradingSchedule", "ScheduleOverride"]
