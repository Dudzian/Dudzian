from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from bot_core.auto_trader import AutoTrader, RiskDecision
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


def test_auto_trader_audit_records_feature_metadata() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)
    frame = pd.DataFrame({"close": [1.0, 1.2], "volume": [100.0, 110.0]})

    assert trader._ai_feature_columns(frame) == ["close", "volume"]

    trader._record_decision_audit_stage(
        "risk_evaluated",
        symbol="BTCUSDT",
        payload={"approved": True},
    )

    entry = trader.get_decision_audit_entries(limit=1)[0]
    metadata = entry["metadata"]
    assert metadata is not None
    assert metadata["feature_columns"] == ["close", "volume"]
    assert metadata["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in metadata


def test_risk_evaluation_history_records_feature_columns() -> None:
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
    frame = pd.DataFrame({"close": [1.0, 1.1], "volume": [95.0, 97.0]})
    assert trader._ai_feature_columns(frame) == ["close", "volume"]

    decision = RiskDecision(
        should_trade=True,
        fraction=1.0,
        state="ready",
        details={
            "decision_engine": {
                "features": {"close": 1.1, "volume": 97.0},
                **trader._feature_column_metadata(["close", "volume"]),
            }
        },
    )

    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=None,
        service=None,
        error=None,
    )

    evaluations = trader._risk_evaluations  # type: ignore[attr-defined]
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    metadata = evaluation["metadata"]
    assert metadata["feature_columns"] == ["close", "volume"]
    assert metadata["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in metadata

    events = [payload for event, payload in emitter.events if event == "auto_trader.risk_evaluation"]
    assert len(events) == 1
    assert events[0]["metadata"]["feature_columns"] == ["close", "volume"]


def test_load_risk_evaluations_preserves_feature_metadata() -> None:
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

    metadata = {
        "feature_columns": ["close", "volume"],
        "feature_columns_source": "configured",
        "configured_feature_columns": ["close", "volume", "ema"],
    }
    payload = {
        "version": 1,
        "entries": [
            {
                "timestamp": 1_700_000_000.0,
                "approved": True,
                "normalized": True,
                "decision": {"state": "ready"},
                "metadata": metadata,
            }
        ],
        "filters": {},
        "retention": {},
        "trimmed_by_ttl": 0,
        "history_size": 1,
    }

    loaded = trader.load_risk_evaluations(payload, notify_listeners=True)
    assert loaded == 1

    evaluations = trader._risk_evaluations  # type: ignore[attr-defined]
    assert len(evaluations) == 1
    record = evaluations[0]
    assert record["metadata"] == metadata

    events = [payload for event, payload in emitter.events if event == "auto_trader.risk_evaluation"]
    assert len(events) == 1
    assert events[0]["metadata"] == metadata


def test_lifecycle_snapshot_includes_feature_columns() -> None:
    log = DecisionAuditLog()
    trader = _build_trader(log)
    frame = pd.DataFrame({"close": [1.0, 1.2], "volume": [100.0, 110.0]})

    # Warm up feature resolver to capture snapshot of available columns.
    assert trader._ai_feature_columns(frame) == ["close", "volume"]

    trader.summarize_guardrail_timeline = lambda **_: {}  # type: ignore[assignment]
    trader.summarize_risk_decision_timeline = lambda **_: {}  # type: ignore[assignment]

    snapshot = trader.build_lifecycle_snapshot()
    risk_section = snapshot["risk_decisions"]

    assert risk_section["feature_columns"] == ["close", "volume"]
    assert risk_section["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in risk_section
