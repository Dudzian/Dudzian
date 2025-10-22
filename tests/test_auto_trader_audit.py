from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bot_core.auto_trader import AutoTrader
from bot_core.auto_trader.audit import DecisionAuditLog, DecisionAuditRecord


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


def _build_trader(log: DecisionAuditLog) -> AutoTrader:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        decision_audit_log=log,
    )
    return trader


def test_decision_audit_log_listeners_receive_records() -> None:
    log = DecisionAuditLog()
    received: list[DecisionAuditRecord] = []

    def listener(record: DecisionAuditRecord) -> None:
        received.append(record)

    log.add_listener(listener)
    log.record("stage1", "BTCUSDT", mode="demo", payload={"step": 1})

    assert len(received) == 1
    assert received[0].stage == "stage1"
    assert received[0].symbol == "BTCUSDT"
    assert received[0].decision_id is None

    assert log.remove_listener(listener) is True
    log.record("stage2", "BTCUSDT", mode="demo")
    assert len(received) == 1


def test_decision_audit_log_load_notifies_listeners_when_requested() -> None:
    source = DecisionAuditLog()
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    source.record("stage1", "BTCUSDT", mode="demo", timestamp=base)
    payload = source.export()

    target = DecisionAuditLog()
    received: list[str] = []

    def listener(record: DecisionAuditRecord) -> None:
        received.append(record.stage)

    target.add_listener(listener)
    loaded = target.load(payload, notify_listeners=True)

    assert loaded == 1
    assert received == ["stage1"]


def test_decision_audit_log_query_filters_and_limits() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    log.record(
        "initialised",
        "BTCUSDT",
        mode="demo",
        payload={"step": 1},
        timestamp=base,
        decision_id="cycle-1",
    )
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="live",
        payload={"approved": True},
        risk_snapshot={"value": 1},
        timestamp=base + timedelta(seconds=10),
        decision_id="cycle-2",
    )
    log.record(
        "risk_evaluated",
        "ETHUSDT",
        mode="live",
        payload={"approved": False},
        portfolio_snapshot={"positions": {}},
        timestamp=base + timedelta(seconds=20),
    )

    stage_filtered = log.query_dicts(stage="risk_evaluated", symbol={"BTCUSDT"})
    assert len(stage_filtered) == 1
    assert stage_filtered[0]["symbol"] == "BTCUSDT"
    assert stage_filtered[0]["decision_id"] == "cycle-2"

    time_filtered = log.query_dicts(
        stage=["risk_evaluated"],
        since=(base + timedelta(seconds=15)).isoformat(),
    )
    assert len(time_filtered) == 1
    assert time_filtered[0]["symbol"] == "ETHUSDT"

    assert log.query_dicts(has_risk_snapshot=True)[0]["stage"] == "risk_evaluated"
    assert log.query_dicts(has_portfolio_snapshot=True)[0]["symbol"] == "ETHUSDT"

    limited = log.query_dicts(limit=2)
    assert [entry["stage"] for entry in limited] == [
        "risk_evaluated",
        "risk_evaluated",
    ]

    reversed_entries = log.query_dicts(limit=2, reverse=True)
    assert [entry["symbol"] for entry in reversed_entries] == [
        "ETHUSDT",
        "BTCUSDT",
    ]

    decision_filtered = log.query_dicts(decision_id="cycle-2")
    assert len(decision_filtered) == 1
    assert decision_filtered[0]["stage"] == "risk_evaluated"
    assert decision_filtered[0]["decision_id"] == "cycle-2"


def test_decision_audit_log_dataframe_filters_and_timezone() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    log.record(
        "initialised",
        "BTCUSDT",
        mode="demo",
        payload={"step": 1},
        timestamp=base,
    )
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="live",
        payload={"approved": True},
        risk_snapshot={"value": 1},
        timestamp=base + timedelta(seconds=5),
        decision_id="cycle-200",
    )
    log.record(
        "risk_evaluated",
        "ETHUSDT",
        mode="live",
        payload={"approved": False},
        portfolio_snapshot={"positions": {}},
        timestamp=base + timedelta(seconds=10),
    )

    frame = log.to_dataframe(stage="risk_evaluated", timezone_hint=timezone.utc)

    assert list(frame["stage"]) == ["risk_evaluated", "risk_evaluated"]
    assert list(frame["decision_id"]) == ["cycle-200", None]
    assert frame["timestamp"].iloc[0].tzinfo == timezone.utc
    assert frame.attrs["audit_filters"]["stage"] == frozenset({"risk_evaluated"})
    assert frame.attrs["audit_filters"]["limit"] == 20
    assert frame.attrs["audit_filters"]["decision_id"] is None


def test_decision_audit_log_summary_counts_and_timestamps() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    log.record(
        "initialised",
        "BTCUSDT",
        mode="demo",
        payload={"step": 1},
        timestamp=base,
        decision_id="cycle-1",
    )
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="live",
        payload={"approved": True},
        risk_snapshot={"value": 1},
        timestamp=base + timedelta(seconds=5),
        decision_id="cycle-1",
    )
    log.record(
        "risk_evaluated",
        "ETHUSDT",
        mode="live",
        payload={"approved": False},
        portfolio_snapshot={"positions": {}},
        timestamp=base + timedelta(seconds=10),
        decision_id="cycle-2",
    )

    summary = log.summarize()
    assert summary["count"] == 3
    assert summary["stages"]["risk_evaluated"] == 2
    assert summary["symbols"]["BTCUSDT"] == 2
    assert summary["modes"]["live"] == 2
    assert summary["decision_ids"] == {"cycle-1": 2, "cycle-2": 1}
    assert summary["unique_decision_ids"] == 2
    assert summary["with_risk_snapshot"] == 1
    assert summary["with_portfolio_snapshot"] == 1
    assert summary["first_timestamp"].startswith("2024-01-01T00:00:00")
    assert summary["last_timestamp"].startswith("2024-01-01T00:00:10")

    filtered = log.summarize(stage="risk_evaluated", symbol="ETHUSDT")
    assert filtered["count"] == 1
    assert filtered["with_portfolio_snapshot"] == 1
    assert filtered["symbols"] == {"ETHUSDT": 1}
    assert filtered["decision_ids"] == {"cycle-2": 1}
    assert filtered["unique_decision_ids"] == 1


def test_decision_audit_log_trace_decision_enriches_timeline() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    log.record(
        "initialised",
        "BTCUSDT",
        mode="demo",
        payload={"step": 1},
        timestamp=base,
        decision_id="cycle-900",
    )
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="demo",
        payload={"approved": True},
        risk_snapshot={"score": 0.9},
        timestamp=base + timedelta(seconds=5),
        decision_id="cycle-900",
    )
    log.record(
        "order_submitted",
        "BTCUSDT",
        mode="demo",
        payload={"order_id": "abc"},
        portfolio_snapshot={"positions": {"BTC": 1}},
        metadata={"request_id": "req-1"},
        timestamp=base + timedelta(seconds=12),
        decision_id="cycle-900",
    )

    timeline = log.trace_decision("cycle-900", timezone_hint=timezone.utc)

    assert len(timeline) == 3
    assert timeline[0]["step_index"] == 0
    assert timeline[0]["elapsed_since_first_s"] == 0.0
    assert timeline[1]["elapsed_since_previous_s"] == 5.0
    assert timeline[2]["elapsed_since_first_s"] == 12.0
    assert timeline[1]["payload"] == {"approved": True}
    assert timeline[2]["portfolio_snapshot"] == {"positions": {"BTC": 1}}
    assert timeline[2]["metadata"] == {"request_id": "req-1"}
    assert timeline[0]["timestamp"].endswith("+00:00")

    risk_only = log.trace_decision(
        "cycle-900",
        stage="risk_evaluated",
        has_risk_snapshot=True,
        include_payload=False,
        include_metadata=False,
    )

    assert len(risk_only) == 1
    assert "payload" not in risk_only[0]
    assert "metadata" not in risk_only[0]
    assert risk_only[0]["elapsed_since_first_s"] == 0.0

    assert log.trace_decision("missing-id") == ()
    assert log.trace_decision(None) == ()


def test_decision_audit_log_group_by_decision_respects_filters() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    log.record(
        "initialised",
        "BTCUSDT",
        mode="demo",
        timestamp=base,
        decision_id="cycle-1",
    )
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="demo",
        timestamp=base + timedelta(seconds=5),
        decision_id="cycle-1",
    )
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="demo",
        timestamp=base + timedelta(seconds=10),
    )
    log.record(
        "risk_evaluated",
        "ETHUSDT",
        mode="demo",
        timestamp=base + timedelta(seconds=15),
        decision_id="cycle-2",
    )

    grouped = log.group_by_decision(
        stage="risk_evaluated",
        limit=2,
        reverse=True,
        include_unidentified=True,
        timezone_hint=timezone.utc,
    )

    assert list(grouped.keys()) == ["cycle-2", None]
    assert grouped["cycle-2"][0]["decision_id"] == "cycle-2"
    assert grouped["cycle-2"][0]["timestamp"].endswith("+00:00")
    assert grouped[None][0]["decision_id"] is None

    filtered = log.group_by_decision(
        stage="risk_evaluated",
        limit=2,
        reverse=True,
        include_unidentified=False,
    )

    assert list(filtered.keys()) == ["cycle-2"]
    assert filtered["cycle-2"][0]["symbol"] == "ETHUSDT"


def test_decision_audit_log_export_and_load_roundtrip(tmp_path: Path) -> None:
    log = DecisionAuditLog(max_entries=16, max_age_s=600)
    base = datetime.now(timezone.utc)

    log.record(
        "initialised",
        "BTCUSDT",
        mode="demo",
        payload={"step": 1},
        timestamp=base,
        decision_id="cycle-a",
    )
    log.record(
        "risk_evaluated",
        "ETHUSDT",
        mode="live",
        payload={"approved": False},
        risk_snapshot={"ratio": 0.9},
        timestamp=base + timedelta(seconds=15),
        decision_id="cycle-b",
    )

    exported = log.export()
    assert exported["retention"] == {"max_entries": 16, "max_age_s": 600.0}

    target = tmp_path / "audit.json"
    log.dump(target)
    assert target.exists()

    restored = DecisionAuditLog()
    loaded = restored.load(exported)
    assert loaded == 2
    assert restored.export()["entries"] == exported["entries"]


def test_decision_audit_log_export_includes_filters() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    log.record("initialised", "BTCUSDT", mode="demo", timestamp=base)
    log.record(
        "risk_evaluated",
        "BTCUSDT",
        mode="live",
        payload={"approved": True},
        risk_snapshot={"score": 0.7},
        timestamp=base + timedelta(seconds=5),
    )

    payload = log.export(
        limit=1,
        reverse=True,
        stage="risk_evaluated",
        symbol=["BTCUSDT"],
        has_risk_snapshot=True,
    )

    assert len(payload["entries"]) == 1
    filters = payload["filters"]
    assert filters["limit"] == 1
    assert filters["reverse"] is True
    assert filters["stage"] == ("risk_evaluated",)
    assert filters["symbol"] == ("BTCUSDT",)
    assert filters["has_risk_snapshot"] is True
    assert filters["timezone_hint"] == "UTC"
    assert "decision_id" in filters
    assert filters["decision_id"] is None


def test_decision_audit_log_dump_preserves_filters(tmp_path: Path) -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    log.record("risk_evaluated", "BTCUSDT", mode="live", timestamp=base)

    target = tmp_path / "audit.json"
    log.dump(target, stage="risk_evaluated", limit=1)

    with target.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    assert payload["filters"]["stage"] == ["risk_evaluated"]
    assert payload["filters"]["limit"] == 1
    assert payload["filters"].get("decision_id") is None


def test_decision_audit_log_load_merge_preserves_existing_entries() -> None:
    base = datetime.now(timezone.utc)
    source = DecisionAuditLog()
    source.record("initialised", "BTCUSDT", mode="demo", timestamp=base)
    snapshot = source.export()

    target = DecisionAuditLog()
    target.record("schedule_configured", "<schedule>", mode="demo")

    loaded = target.load(snapshot, merge=True)
    assert loaded == 1

    entries = target.query_dicts(limit=None)
    assert len(entries) == 2
    assert {entry["stage"] for entry in entries} == {
        "schedule_configured",
        "initialised",
    }


def test_auto_trader_exposes_filtered_audit_entries() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)

    trader._record_decision_audit_stage(
        "schedule_configured",
        symbol="<schedule>",
        payload={"mode": "demo"},
    )
    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
        risk_snapshot={"value": 1},
    )

    assert trader.get_decision_audit_entries(limit=5)

    filtered = trader.get_decision_audit_entries(
        limit=1,
        stage="risk_evaluated",
        has_risk_snapshot=True,
    )
    assert len(filtered) == 1
    assert filtered[0]["stage"] == "risk_evaluated"
    assert filtered[0]["payload"]["approved"] is True
    assert filtered[0]["decision_id"]

    decision_filtered = trader.get_decision_audit_entries(
        decision_id=filtered[0]["decision_id"],
        limit=5,
    )
    assert len(decision_filtered) == 1
    assert decision_filtered[0]["stage"] == "risk_evaluated"

    none_result = trader.get_decision_audit_entries(
        limit=1,
        stage="risk_evaluated",
        has_portfolio_snapshot=True,
    )
    assert none_result == ()


def test_auto_trader_decision_audit_listener_delegation() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)
    received: list[str] = []

    def listener(record: DecisionAuditRecord) -> None:
        received.append(record.stage)

    assert trader.add_decision_audit_listener(listener) is True

    trader._record_decision_audit_stage(
        "schedule_configured",
        symbol="<schedule>",
        payload={"mode": "demo"},
    )
    assert received == ["schedule_configured"]

    assert trader.remove_decision_audit_listener(listener) is True
    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
    )
    assert received == ["schedule_configured"]


def test_auto_trader_decision_audit_listener_without_log() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)
    trader._decision_audit_log = None  # type: ignore[assignment]

    def listener(record: DecisionAuditRecord) -> None:  # pragma: no cover - local stub
        raise AssertionError("Listener should not be invoked")

    assert trader.add_decision_audit_listener(listener) is False
    assert trader.remove_decision_audit_listener(listener) is False


def test_auto_trader_emits_decision_audit_events() -> None:
    log = DecisionAuditLog()
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        decision_audit_log=log,
    )

    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
    )

    decision_events = [
        (name, payload)
        for name, payload in emitter.events
        if name == "auto_trader.decision_audit"
    ]
    assert decision_events, "Recording an audit entry should emit a decision event"
    _, payload = decision_events[-1]
    assert payload["stage"] == "risk_evaluated"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["payload"]["approved"] is True


def test_auto_trader_load_decision_audit_log_notifies_listeners() -> None:
    source = DecisionAuditLog()
    source.record("risk_evaluated", "BTCUSDT", mode="demo")
    payload = source.export()

    target_log = DecisionAuditLog()
    trader = _build_trader(target_log)
    received: list[str] = []

    def listener(record: DecisionAuditRecord) -> None:
        received.append(record.stage)

    trader.add_decision_audit_listener(listener)
    loaded = trader.load_decision_audit_log(payload, notify_listeners=True)

    assert loaded == 1
    assert received == ["risk_evaluated"]


def test_auto_trader_get_grouped_decision_audit_entries_delegates() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)
    decision_id = "cycle-group"

    trader._record_decision_audit_stage(
        "initialised",
        symbol="BTCUSDT",
        payload={"step": 1},
        decision_id=decision_id,
    )
    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
        decision_id=decision_id,
    )

    groups = trader.get_grouped_decision_audit_entries(decision_id=decision_id)

    assert list(groups.keys()) == [decision_id]
    assert len(groups[decision_id]) == 2
    assert groups[decision_id][0]["stage"] == "initialised"

    trader._decision_audit_log = None  # type: ignore[assignment]

    assert trader.get_grouped_decision_audit_entries() == {}


def test_auto_trader_exposes_audit_dataframe() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)

    trader._record_decision_audit_stage(
        "schedule_configured",
        symbol="<schedule>",
        payload={"mode": "demo"},
    )
    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
    )

    frame = trader.get_decision_audit_dataframe(stage="schedule_configured")

    assert not frame.empty
    assert frame.attrs["audit_filters"]["stage"] == frozenset({"schedule_configured"})
    assert list(frame["stage"]) == ["schedule_configured"]


def test_auto_trader_audit_dataframe_without_log_returns_empty_frame() -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )

    frame = trader.get_decision_audit_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert frame.attrs["audit_filters"]["limit"] == 20
    assert "decision_id" in frame


def test_auto_trader_exposes_audit_summary() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)

    trader._record_decision_audit_stage(
        "schedule_configured",
        symbol="<schedule>",
        payload={"mode": "demo"},
    )
    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
        risk_snapshot={"value": 1},
    )

    summary = trader.get_decision_audit_summary()
    assert summary["count"] == 2
    assert summary["stages"]["risk_evaluated"] == 1
    assert summary["with_risk_snapshot"] == 1
    assert summary["unique_decision_ids"] == 2

    filtered = trader.get_decision_audit_summary(stage="risk_evaluated")
    assert filtered["count"] == 1
    assert filtered["symbols"] == {"BTCUSDT": 1}
    assert filtered["unique_decision_ids"] == 1


def test_auto_trader_get_decision_audit_trace_delegates_to_log() -> None:
    log = DecisionAuditLog()
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    log.record(
        "initialised",
        "ETHUSDT",
        mode="demo",
        timestamp=base,
        decision_id="cycle-xyz",
    )
    log.record(
        "risk_evaluated",
        "ETHUSDT",
        mode="demo",
        risk_snapshot={"score": 0.5},
        timestamp=base + timedelta(seconds=3),
        decision_id="cycle-xyz",
    )

    trader = _build_trader(log)

    trace = trader.get_decision_audit_trace(
        "cycle-xyz",
        include_payload=False,
        include_snapshots=False,
    )

    assert len(trace) == 2
    assert trace[0]["step_index"] == 0
    assert "payload" not in trace[0]
    assert "risk_snapshot" not in trace[1]
    assert trace[1]["elapsed_since_previous_s"] == 3.0

    trader_without_log = AutoTrader(
        _Emitter(),
        _GUI(),
        symbol_getter=lambda: "ETHUSDT",
        enable_auto_trade=False,
    )

    assert trader_without_log.get_decision_audit_trace("cycle-xyz") == ()


def test_decision_audit_log_prunes_entries_by_max_age() -> None:
    base = datetime.now(timezone.utc)
    log = DecisionAuditLog(max_age_s=15)
    log.record(
        "stage1",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 1},
        timestamp=base - timedelta(seconds=21),
    )
    log.record(
        "stage2",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 2},
        timestamp=base - timedelta(seconds=5),
    )

    entries = log.query_dicts(limit=None)
    assert len(entries) == 1
    assert entries[0]["payload"]["idx"] == 2


def test_decision_audit_log_manual_trim_and_retention() -> None:
    base = datetime.now(timezone.utc)
    log = DecisionAuditLog(max_entries=10)
    log.record(
        "stage1",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 1},
        timestamp=base - timedelta(seconds=30),
    )
    log.record(
        "stage2",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 2},
        timestamp=base - timedelta(seconds=5),
    )

    removed = log.trim(before=base - timedelta(seconds=10))
    assert removed == 1
    assert len(log.query_dicts(limit=None)) == 1

    log.set_retention(max_entries=1)
    log.record(
        "stage3",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 3},
        timestamp=base,
    )
    entries = log.query_dicts(limit=None)
    assert len(entries) == 1
    assert entries[0]["payload"]["idx"] == 3


def test_auto_trader_trim_decision_audit_log_delegates() -> None:
    base = datetime.now(timezone.utc)
    log = DecisionAuditLog()
    trader = _build_trader(log)
    log.record(
        "stage1",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 1},
        timestamp=base - timedelta(seconds=30),
    )
    log.record(
        "stage2",
        "BTCUSDT",
        mode="demo",
        payload={"idx": 2},
        timestamp=base - timedelta(seconds=5),
    )

    removed = trader.trim_decision_audit_log(before=base - timedelta(seconds=10))
    assert removed == 1

    remaining = trader.get_decision_audit_entries(limit=None)
    assert len(remaining) == 1
    assert remaining[0]["payload"]["idx"] == 2


def test_auto_trader_export_and_load_audit_log_roundtrip() -> None:
    source_log = DecisionAuditLog()
    trader = _build_trader(source_log)

    trader._record_decision_audit_stage(
        "initialised",
        symbol="BTCUSDT",
        payload={"step": 1},
    )

    payload = trader.export_decision_audit_log(stage="initialised")
    assert payload["entries"]
    assert payload["filters"]["stage"] == ("initialised",)
    assert payload["filters"]["timezone_hint"] == "UTC"
    assert payload["filters"].get("decision_id") is None

    other_trader = _build_trader(DecisionAuditLog())
    loaded = other_trader.load_decision_audit_log(payload)
    assert loaded == len(payload["entries"])

    assert other_trader.get_decision_audit_entries(limit=None) == trader.get_decision_audit_entries(
        limit=None
    )


def test_auto_trader_export_and_load_audit_without_log() -> None:
    trader = _build_trader(DecisionAuditLog())
    trader._decision_audit_log = None  # type: ignore[assignment]

    payload = trader.export_decision_audit_log()
    assert payload["entries"] == []
    assert payload["filters"].get("decision_id") is None
    assert payload["filters"]["limit"] is None
    assert payload["filters"]["timezone_hint"] == "UTC"
    assert trader.load_decision_audit_log(payload) == 0
