from __future__ import annotations

from datetime import datetime, timedelta, timezone, time
from typing import Any

import pytest

from bot_core.auto_trader import (
    AutoTrader,
    ScheduleOverride,
    ScheduleState,
    ScheduleWindow,
    TradingSchedule,
)


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


def _last_schedule_event(emitter: _Emitter) -> tuple[str, dict[str, Any]]:
    schedule_events = [event for event in emitter.events if event[0] == "auto_trader.schedule_state"]
    assert schedule_events, "Expected at least one schedule state event"
    return schedule_events[-1]


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
    assert last_entry["decision_id"]
    payload = last_entry["payload"]
    assert payload["mode"] == "maintenance"
    assert payload["is_open"] is False
    assert payload["reason"] == "update"
    assert payload["window"]["allow_trading"] is False

    assert emitter.events, "Schedule updates should emit state events"
    event_name, event_payload = _last_schedule_event(emitter)
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
    assert entries[-1]["decision_id"]

    assert emitter.events, "Resetting schedule should emit state event"
    event_name, payload = _last_schedule_event(emitter)
    assert event_name == "auto_trader.schedule_state"
    assert payload["is_open"] is True
    assert payload["reason"] == "reset"


def test_get_work_schedule_exposes_current_configuration() -> None:
    trader, _ = _build_trader()

    default_schedule = trader.get_work_schedule()

    assert isinstance(default_schedule, TradingSchedule)
    default_state = default_schedule.describe()
    assert default_state.mode == "demo"
    assert default_state.is_open is True

    custom_window = ScheduleWindow(
        start=time(9, 0),
        end=time(17, 0),
        mode="live",
        allow_trading=True,
        days={0, 1, 2, 3, 4},
    )
    custom_schedule = TradingSchedule((custom_window,), default_mode="maintenance")

    trader.set_work_schedule(custom_schedule)

    assert trader.get_work_schedule() is custom_schedule


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


def test_schedule_window_to_mapping_exposes_duration_when_requested() -> None:
    window = ScheduleWindow(
        start=time(9, 0),
        end=time(17, 0),
        mode="live",
        allow_trading=True,
        days={0, 1, 2, 3, 4},
    )

    payload = window.to_mapping(include_duration=True)

    assert payload["duration_s"] == 8 * 3600
    assert payload["mode"] == "live"


def test_schedule_override_to_mapping_respects_timezone_and_duration() -> None:
    local_tz = timezone(timedelta(hours=2))
    start = datetime(2024, 5, 1, 12, 0, tzinfo=local_tz)
    override = ScheduleOverride(
        start=start,
        end=start + timedelta(hours=3),
        mode="maintenance",
        allow_trading=False,
        label="service",
    )

    payload = override.to_mapping(include_duration=True, timezone_hint=timezone.utc)

    assert payload["duration_s"] == 3 * 3600
    assert payload["start"].endswith("+00:00")
    assert payload["label"] == "service"


def test_schedule_state_to_mapping_serializes_runtime_snapshot() -> None:
    window = ScheduleWindow(
        start=time(0, 0),
        end=time(23, 59, 59),
        mode="live",
        allow_trading=True,
    )
    schedule = TradingSchedule((window,), default_mode="demo")
    now = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    override = ScheduleOverride(
        start=now - timedelta(hours=1),
        end=now + timedelta(hours=1),
        mode="maintenance",
        allow_trading=False,
        label="window-service",
    )
    schedule = schedule.with_overrides((override,))

    state = schedule.describe(now=now)
    payload = state.to_mapping()

    assert isinstance(state, ScheduleState)
    assert payload["mode"] == "maintenance"
    assert payload["override_active"] is True
    assert payload["override"]["label"] == "window-service"
    assert payload["window"]["duration_s"] >= 23 * 3600
    assert "time_until_transition_s" in payload
    assert payload["is_open"] is False


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
    assert entries[-1]["decision_id"]

    assert emitter.events
    event_name, payload = _last_schedule_event(emitter)
    assert event_name == "auto_trader.schedule_state"
    assert payload["reason"] == "config"


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
    assert entries[-1]["decision_id"]
    assert payload["override"]["label"] == "ad-hoc"
    assert payload["override_active"] is True
    assert payload.get("time_until_next_override_s") is None

    assert emitter.events
    event_name, event_payload = _last_schedule_event(emitter)
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


def test_apply_schedule_override_appends_and_audits() -> None:
    trader, emitter = _build_trader()
    trader.clear_decision_audit_log()

    base_window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    schedule = TradingSchedule((base_window,), timezone_name="UTC", default_mode="live")
    trader.set_work_schedule(schedule, reason="base")
    trader.clear_decision_audit_log()
    emitter.events.clear()

    now = datetime.now(timezone.utc)
    override_payload = {
        "start": (now - timedelta(minutes=5)).isoformat(),
        "end": (now + timedelta(hours=1)).isoformat(),
        "mode": "maintenance",
        "allow_trading": False,
        "label": "manual-maintenance",
    }

    state = trader.apply_schedule_override(override_payload, reason="manual")

    assert state.override is not None
    assert state.override.label == "manual-maintenance"
    assert state.is_open is False

    entries = trader.get_decision_audit_entries(limit=2)
    assert entries[-1]["stage"] == "schedule_override_applied"
    assert entries[-1]["decision_id"] == entries[-2]["decision_id"]
    payload = entries[-1]["payload"]
    assert payload["reason"] == "manual"
    assert payload["override_replace"] is False
    assert payload["overrides_applied"][0]["label"] == "manual-maintenance"

    assert emitter.events
    event_name, event_payload = _last_schedule_event(emitter)
    assert event_name == "auto_trader.schedule_state"
    assert event_payload["reason"] == "manual"
    assert event_payload["override"]["label"] == "manual-maintenance"


def test_apply_schedule_override_replace_existing() -> None:
    trader, _ = _build_trader()

    now = datetime.now(timezone.utc)
    base_window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    existing_override = ScheduleOverride(
        start=now - timedelta(minutes=30),
        end=now + timedelta(minutes=30),
        mode="maintenance",
        allow_trading=False,
        label="existing",
    )
    schedule = TradingSchedule(
        (base_window,),
        timezone_name="UTC",
        default_mode="live",
        overrides=(existing_override,),
    )
    trader.set_work_schedule(schedule, reason="seed")
    trader.clear_decision_audit_log()

    new_override = ScheduleOverride(
        start=now + timedelta(hours=1),
        end=now + timedelta(hours=2),
        mode="maintenance",
        allow_trading=False,
        label="replacement",
    )

    state = trader.apply_schedule_override(new_override, reason="replace", replace=True)

    assert state.override is None or state.override.label == "replacement"
    assert state.next_override is None or state.next_override.label == "replacement"

    entries = trader.get_decision_audit_entries(limit=1)
    payload = entries[-1]["payload"]
    assert entries[-1]["decision_id"]
    assert payload["reason"] == "replace"
    assert payload["override_replace"] is True
    assert payload["overrides_applied"][0]["label"] == "replacement"


def test_clear_schedule_overrides_by_label() -> None:
    trader, emitter = _build_trader()

    now = datetime.now(timezone.utc)
    base_window = ScheduleWindow(start=time(0, 0), end=time(0, 0), mode="live", allow_trading=True)
    first_override = ScheduleOverride(
        start=now + timedelta(minutes=10),
        end=now + timedelta(hours=1),
        mode="maintenance",
        allow_trading=False,
        label="maintenance-a",
    )
    second_override = ScheduleOverride(
        start=now + timedelta(hours=2),
        end=now + timedelta(hours=3),
        mode="maintenance",
        allow_trading=False,
        label="maintenance-b",
    )
    schedule = TradingSchedule(
        (base_window,),
        timezone_name="UTC",
        default_mode="live",
        overrides=(first_override, second_override),
    )
    trader.set_work_schedule(schedule, reason="seed")
    trader.clear_decision_audit_log()
    emitter.events.clear()

    state = trader.clear_schedule_overrides(labels=["maintenance-a"], reason="cleanup")

    assert state.next_override is not None
    assert state.next_override.label == "maintenance-b"

    entries = trader.get_decision_audit_entries(limit=1)
    assert entries[-1]["stage"] == "schedule_override_cleared"
    assert entries[-1]["decision_id"]
    payload = entries[-1]["payload"]
    assert payload["reason"] == "cleanup"
    assert payload["cleared_labels"] == ["maintenance-a"]
    remaining = payload["remaining_overrides"]
    assert len(remaining) == 1
    assert remaining[0]["label"] == "maintenance-b"

    assert emitter.events
    event_name, event_payload = _last_schedule_event(emitter)
    assert event_name == "auto_trader.schedule_state"
    assert event_payload["reason"] == "cleanup"


def test_list_schedule_overrides_tracks_updates() -> None:
    trader, _ = _build_trader()

    start = datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc)
    override = ScheduleOverride(
        start=start,
        end=start + timedelta(hours=3),
        mode="maintenance",
        allow_trading=False,
        label="maintenance-window",
    )

    trader.apply_schedule_override(override, reason="patch")

    overrides = trader.list_schedule_overrides()
    assert overrides
    assert overrides[-1].label == "maintenance-window"
    assert overrides[-1].allow_trading is False

    trader.clear_schedule_overrides(labels="maintenance-window", reason="cleanup")

    assert trader.list_schedule_overrides() == ()
