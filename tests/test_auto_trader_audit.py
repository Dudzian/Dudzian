from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from bot_core.auto_trader import AutoTrader
from bot_core.auto_trader.audit import DecisionAuditLog


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


def test_decision_audit_log_query_filters_and_limits() -> None:
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
        timestamp=base + timedelta(seconds=10),
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

    none_result = trader.get_decision_audit_entries(
        limit=1,
        stage="risk_evaluated",
        has_portfolio_snapshot=True,
    )
    assert none_result == ()
