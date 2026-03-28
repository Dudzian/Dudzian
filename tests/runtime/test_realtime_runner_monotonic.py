from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from bot_core.runtime.realtime import DailyTrendRealtimeRunner


def test_realtime_runner_sleep_policy_is_stable_when_wall_clock_goes_backwards() -> None:
    class _ControllerStub:
        interval = "1m"
        tick_seconds = 10.0

        def collect_signals(self, *, start: int, end: int) -> Sequence[object]:
            return ()

    class _TradingControllerStub:
        def maybe_report_health(self) -> None:
            return None

    timeline = iter(
        [
            datetime.fromtimestamp(100, tz=timezone.utc),  # cycle_start (run_forever)
            datetime.fromtimestamp(101, tz=timezone.utc),  # run_once() domain clock
            datetime.fromtimestamp(95, tz=timezone.utc),  # rollback in pacing checkpoint
            datetime.fromtimestamp(102, tz=timezone.utc),  # cycle_start (2nd cycle)
            datetime.fromtimestamp(103, tz=timezone.utc),  # run_once() (2nd cycle)
        ]
    )

    def fake_clock() -> datetime:
        try:
            return next(timeline)
        except StopIteration:
            return datetime.fromtimestamp(103, tz=timezone.utc)

    monotonic_timeline = iter([10.0, 12.0, 20.0])

    def fake_monotonic() -> float:
        try:
            return next(monotonic_timeline)
        except StopIteration:
            return 20.0

    sleeps: list[float] = []
    runner = DailyTrendRealtimeRunner(
        controller=_ControllerStub(),  # type: ignore[arg-type]
        trading_controller=_TradingControllerStub(),  # type: ignore[arg-type]
        clock=fake_clock,
        monotonic_clock=fake_monotonic,
        sleep=sleeps.append,
    )

    runner.run_forever(max_cycles=2)

    assert len(sleeps) == 1
    assert sleeps[0] == 8.0
    assert runner.last_cycle_started_at == datetime.fromtimestamp(102, tz=timezone.utc)


def test_realtime_runner_sleep_policy_is_stable_when_wall_clock_jumps_forward() -> None:
    class _ControllerStub:
        interval = "1m"
        tick_seconds = 10.0

        def collect_signals(self, *, start: int, end: int) -> Sequence[object]:
            return ()

    class _TradingControllerStub:
        def maybe_report_health(self) -> None:
            return None

    timeline = iter(
        [
            datetime.fromtimestamp(100, tz=timezone.utc),  # cycle_start (run_forever)
            datetime.fromtimestamp(101, tz=timezone.utc),  # run_once() domain clock
            datetime.fromtimestamp(900, tz=timezone.utc),  # large wall-clock jump forward
            datetime.fromtimestamp(901, tz=timezone.utc),  # cycle_start (2nd cycle)
            datetime.fromtimestamp(902, tz=timezone.utc),  # run_once() (2nd cycle)
        ]
    )

    def fake_clock() -> datetime:
        try:
            return next(timeline)
        except StopIteration:
            return datetime.fromtimestamp(902, tz=timezone.utc)

    monotonic_timeline = iter([10.0, 11.0, 20.0])

    def fake_monotonic() -> float:
        try:
            return next(monotonic_timeline)
        except StopIteration:
            return 20.0

    sleeps: list[float] = []
    runner = DailyTrendRealtimeRunner(
        controller=_ControllerStub(),  # type: ignore[arg-type]
        trading_controller=_TradingControllerStub(),  # type: ignore[arg-type]
        clock=fake_clock,
        monotonic_clock=fake_monotonic,
        sleep=sleeps.append,
    )

    runner.run_forever(max_cycles=2)

    assert len(sleeps) == 1
    assert sleeps[0] == 9.0
    assert runner.last_cycle_started_at == datetime.fromtimestamp(901, tz=timezone.utc)
