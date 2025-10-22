from __future__ import annotations

from datetime import datetime, timedelta, timezone, time
from typing import Any

import pytest

from bot_core.auto_trader import AutoTrader, ScheduleOverride, ScheduleWindow, TradingSchedule


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = _Var("1h")

    def is_demo_mode_active(self) -> bool:
        return True


class _Emitter:
    def __init__(self) -> None:
        self.logs: list[tuple[str, dict[str, Any]]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, message: str, *_, **payload: Any) -> None:
        self.logs.append((message, dict(payload)))

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, dict(payload)))


def _build_trader() -> tuple[AutoTrader, _Emitter]:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )
    return trader, emitter


def test_set_work_schedule_updates_state_and_audit() -> None:
    trader, emitter = _build_trader()
    window = ScheduleWindow(
        start=time(0, 0),
        end=time(0, 0),
        mode="maintenance",
        allow_trading=False,
    )
    schedule = TradingSchedule((window,), default_mode="demo")

    state = trader.set_work_schedule(schedule)

    assert state.mode == "maintenance"
    assert state.is_open is False
    assert trader.is_schedule_open() is False

    described = trader.get_schedule_state()
    assert described.mode == "maintenance"
    assert described.is_open is False

    entries = trader.get_decision_audit_entries(limit=5)
    assert entries, "Decision audit log should capture schedule updates"
    last_entry = entries[-1]
    assert last_entry["stage"] == "schedule_configured"
    payload = last_entry["payload"]
    assert payload["mode"] == "maintenance"
    assert payload["is_open"] is False
    assert payload["reason"] == "update"
    assert payload["window"]["allow_trading"] is False

    assert emitter.events, "Schedule updates should emit state events"
    event_name, event_payload = emitter.events[-1]
    assert event_name == "auto_trader.schedule_state"
    assert event_payload["mode"] == "maintenance"
    assert event_payload["is_open"] is False
    assert event_payload["reason"] == "update"
    assert event_payload["window"]["allow_trading"] is False


def test_set_work_schedule_accepts_none_and_resets() -> None:
    trader, emitter = _build_trader()
    window = ScheduleWindow(
        start=time(0, 0),
        end=time(0, 0),
        mode="maintenance",
        allow_trading=False,
    )
    schedule = TradingSchedule((window,), default_mode="demo")
    trader.set_work_schedule(schedule)

    emitter.events.clear()

    state = trader.set_work_schedule(None)

    assert state.is_open is True
    assert trader.is_schedule_open() is True
    assert state.mode == trader.get_schedule_state().mode

    entries = trader.get_decision_audit_entries(limit=2)
    assert entries[-1]["payload"]["reason"] == "reset"

    assert emitter.events, "Resetting schedule should emit state event"
    event_name, payload = emitter.events[-1]
    assert event_name == "auto_trader.schedule_state"
    assert payload["is_open"] is True
    assert payload["reason"] == "reset"


def test_set_work_schedule_rejects_invalid() -> None:
    trader, _ = _build_trader()
    with pytest.raises(TypeError):
        trader.set_work_schedule(object())  # type: ignore[arg-type]


def test_schedule_window_mapping_roundtrip() -> None:
    payload = {
        "start": "09:30",
        "end": "17:45:30",
        "mode": "live",
        "allow_trading": True,
        "days": ["mon", "tue", "wednesday"],
        "label": "session",
    }

    window = ScheduleWindow.from_mapping(payload)

    assert window.start == time(9, 30)
    assert window.end == time(17, 45, 30)
    assert window.mode == "live"
    assert window.allow_trading is True
    assert set(window.days) == {0, 1, 2}
    assert window.label == "session"

    roundtrip = window.to_mapping()
    assert roundtrip["start"].startswith("09:30")
    assert roundtrip["end"].startswith("17:45:30")
    assert roundtrip["mode"] == "live"
    assert roundtrip["allow_trading"] is True
    assert roundtrip["label"] == "session"
    assert roundtrip["days"] == [0, 1, 2]


def test_trading_schedule_payload_roundtrip() -> None:
    payload = {
        "timezone": "UTC",
        "default_mode": "demo",
        "windows": [
            {
                "start": "00:00",
                "end": "00:00",
                "mode": "maintenance",
                "allow_trading": False,
            }
        ],
        "overrides": [
            {
                "start": "2024-01-01T10:00:00+00:00",
                "end": "2024-01-01T12:00:00+00:00",
                "mode": "holiday",
                "allow_trading": False,
                "label": "new-year-maintenance",
            }
        ],
    }

    schedule = TradingSchedule.from_payload(payload)
    state = schedule.describe(datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc))

    assert state.mode == "holiday"
    assert state.is_open is False
    assert state.window is not None
    assert state.override is not None
    assert state.override.label == "new-year-maintenance"
    assert state.next_override is None

    roundtrip = schedule.to_payload()
    assert roundtrip["default_mode"] == "demo"
    assert roundtrip["windows"][0]["mode"] == "maintenance"
    assert roundtrip["windows"][0]["allow_trading"] is False
    assert roundtrip["overrides"][0]["mode"] == "holiday"
    assert roundtrip["overrides"][0]["label"] == "new-year-maintenance"


def test_set_work_schedule_accepts_mapping_payload() -> None:
    trader, emitter = _build_trader()
    schedule_payload = {
        "timezone": "UTC",
        "default_mode": "demo",
        "windows": [
            {
                "start": "00:00",
                "end": "00:00",
                "mode": "maintenance",
                "allow_trading": False,
                "label": "full"
            }
        ],
    }

    state = trader.set_work_schedule(schedule_payload, reason="config")

    assert state.mode == "maintenance"
    assert state.is_open is False
    assert trader.is_schedule_open() is False

    entries = trader.get_decision_audit_entries(limit=1)
    assert entries, "Schedule configuration should produce audit entry"
    assert entries[-1]["payload"]["reason"] == "config"

    assert emitter.events
    event_name, payload = emitter.events[-1]
    assert event_name == "auto_trader.schedule_state"
    assert payload["reason"] == "config"


def test_describe_work_schedule_returns_payload_with_state() -> None:
    trader, _ = _build_trader()
    now = datetime.now(timezone.utc)
    schedule_payload = {
        "timezone": "UTC",
        "default_mode": "live",
        "windows": [
            {
                "start": "07:00",
                "end": "19:00",
                "mode": "live",
                "allow_trading": True,
                "days": ["mon", "tue"],
                "label": "session",
            }
        ],
        "overrides": [
            {
                "start": (now - timedelta(minutes=5)).isoformat(),
                "end": (now + timedelta(minutes=25)).isoformat(),
                "mode": "maintenance",
                "allow_trading": False,
                "label": "window",
            }
        ],
    }

    trader.set_work_schedule(schedule_payload)

    described = trader.describe_work_schedule()

    assert described["timezone"] == "UTC"
    assert described["windows"][0]["label"] == "session"
    state = described["state"]
    assert state["mode"] == "maintenance"
    assert "override_active" in state


def test_describe_work_schedule_returns_default_when_unset() -> None:
    trader, _ = _build_trader()

    described = trader.describe_work_schedule()

    assert described["default_mode"] == "demo"
    assert described["windows"]
    state = described["state"]
    assert state["mode"] == "demo"
    assert state["is_open"] is True


def test_schedule_override_blocks_trading_and_serializes() -> None:
    trader, emitter = _build_trader()
    now = datetime.now(timezone.utc)
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    override = ScheduleOverride(
        start=now - timedelta(minutes=15),
        end=now + timedelta(minutes=45),
        mode="maintenance",
        allow_trading=False,
        label="ad-hoc",
    )
    schedule = TradingSchedule((window,), timezone_name="UTC", overrides=(override,))

    state = trader.set_work_schedule(schedule, reason="override")

    assert state.override is not None
    assert state.override.label == "ad-hoc"
    assert state.is_open is False
    assert state.override_active is True
    assert state.next_transition == override.end

    entries = trader.get_decision_audit_entries(limit=1)
    assert entries
    payload = entries[-1]["payload"]
    assert payload["override"]["label"] == "ad-hoc"
    assert payload["override_active"] is True
    assert payload.get("time_until_next_override_s") is None

    assert emitter.events
    event_name, event_payload = emitter.events[-1]
    assert event_name == "auto_trader.schedule_state"
    assert event_payload["override"]["mode"] == "maintenance"
    assert event_payload["override_active"] is True


def test_schedule_next_override_shortens_transition() -> None:
    window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    first_override = ScheduleOverride(
        start=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
        mode="maintenance",
        allow_trading=False,
        label="midday",
    )
    second_override = ScheduleOverride(
        start=datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
        mode="maintenance",
        allow_trading=False,
        label="next-day",
    )
    schedule = TradingSchedule(
        (window,),
        timezone_name="UTC",
        overrides=(first_override, second_override),
        default_mode="live",
    )

    before_override = schedule.describe(datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc))
    assert before_override.override is None
    assert before_override.next_override is not None
    assert before_override.next_override.label == "midday"
    assert before_override.next_transition == first_override.start

    during_override = schedule.describe(datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc))
    assert during_override.override is not None
    assert during_override.override.label == "midday"
    assert during_override.is_open is False
    assert during_override.next_transition == first_override.end
    assert during_override.next_override is not None
    assert during_override.next_override.label == "next-day"
