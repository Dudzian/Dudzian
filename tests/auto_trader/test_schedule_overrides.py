from __future__ import annotations

from datetime import datetime, time, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


_schedule_path = Path(__file__).resolve().parents[2] / "bot_core" / "auto_trader" / "schedule.py"
_spec = spec_from_file_location("schedule_module_for_tests", _schedule_path)
assert _spec is not None and _spec.loader is not None
_schedule = module_from_spec(_spec)
sys.modules[_spec.name] = _schedule
_spec.loader.exec_module(_schedule)

ScheduleOverride = _schedule.ScheduleOverride
ScheduleWindow = _schedule.ScheduleWindow
TradingSchedule = _schedule.TradingSchedule


def test_with_overrides_returns_new_schedule_without_mutating_original() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    base_override = ScheduleOverride(
        start=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
        mode="maintenance",
        allow_trading=False,
    )
    schedule = TradingSchedule((window,), timezone_name="UTC", overrides=(base_override,))

    updated = schedule.with_overrides(
        (
            ScheduleOverride(
                start=datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
                mode="live",
                allow_trading=True,
            ),
        )
    )

    assert schedule is not updated
    assert len(schedule.overrides) == 1
    assert schedule.overrides[0].start == datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    assert len(updated.overrides) == 1
    assert updated.overrides[0].start == datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)


def test_with_overrides_parses_mapping_in_schedule_timezone() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((window,), timezone_name="Europe/Warsaw")

    updated = schedule.with_overrides(
        (
            {
                "start": "2024-01-01T10:00:00",
                "end": "2024-01-01T11:00:00",
                "mode": "maintenance",
                "allow_trading": False,
            },
        )
    )

    override = updated.overrides[0]
    assert override.start.tzinfo == schedule.timezone
    assert override.end.tzinfo == schedule.timezone
    assert override.start.hour == 10
    assert override.end.hour == 11


def test_with_overrides_keeps_constructor_sorting_and_tz_normalization() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((window,), timezone_name="Europe/Warsaw")

    updated = schedule.with_overrides(
        (
            {
                "start": "2024-01-01T11:00:00+00:00",
                "end": "2024-01-01T12:00:00+00:00",
                "mode": "maintenance",
                "allow_trading": False,
            },
            {
                "start": "2024-01-01T09:00:00+00:00",
                "end": "2024-01-01T10:00:00+00:00",
                "mode": "maintenance",
                "allow_trading": False,
            },
        )
    )

    assert updated.overrides[0].start < updated.overrides[1].start
    assert updated.overrides[0].start.tzinfo == schedule.timezone
    assert updated.overrides[1].start.tzinfo == schedule.timezone


def test_with_overrides_accepts_z_suffix_datetime() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((window,), timezone_name="UTC")

    updated = schedule.with_overrides(
        (
            {
                "start": "2024-01-01T10:00:00Z",
                "end": "2024-01-01T11:00:00Z",
                "mode": "maintenance",
                "allow_trading": False,
            },
        )
    )

    assert updated.overrides[0].start == datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert updated.overrides[0].end == datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)


def test_with_overrides_parses_string_allow_trading_flag() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((window,), timezone_name="UTC")

    updated = schedule.with_overrides(
        (
            {
                "start": "2024-01-01T10:00:00+00:00",
                "end": "2024-01-01T11:00:00+00:00",
                "mode": "maintenance",
                "allow_trading": "false",
            },
        )
    )

    assert updated.overrides[0].allow_trading is False


def test_with_overrides_treats_blank_string_allow_trading_as_false() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((window,), timezone_name="UTC")

    updated = schedule.with_overrides(
        (
            {
                "start": "2024-01-01T10:00:00+00:00",
                "end": "2024-01-01T11:00:00+00:00",
                "mode": "maintenance",
                "allow_trading": "   ",
            },
        )
    )

    assert updated.overrides[0].allow_trading is False


@pytest.mark.parametrize(
    ("allow_value", "expected"),
    [
        ("0", False),
        ("1", True),
        ("off", False),
        ("on", True),
    ],
)
def test_with_overrides_parses_common_string_boolean_variants(
    allow_value: str, expected: bool
) -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((window,), timezone_name="UTC")

    updated = schedule.with_overrides(
        (
            {
                "start": "2024-01-01T10:00:00+00:00",
                "end": "2024-01-01T11:00:00+00:00",
                "mode": "maintenance",
                "allow_trading": allow_value,
            },
        )
    )

    assert updated.overrides[0].allow_trading is expected
