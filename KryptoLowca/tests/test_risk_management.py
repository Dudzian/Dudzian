from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from bot_core.alerts import AlertSeverity, get_alert_dispatcher
from bot_core.risk.events import RiskDecisionLog
from KryptoLowca.managers.risk_manager_adapter import RiskManager


def _sample_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": np.linspace(100, 110, 120),
            "high": np.linspace(101, 111, 120),
            "low": np.linspace(99, 109, 120),
        }
    )


def _sample_portfolio() -> Mapping[str, object]:
    return {"capital": 10_000.0, "positions": []}


def test_risk_manager_records_decision(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    decision_log = RiskDecisionLog(jsonl_path=log_path, max_entries=10)
    manager = RiskManager(
        config={
            "max_risk_per_trade": 0.02,
            "max_drawdown_pct": 0.2,
            "max_daily_loss_pct": 0.1,
            "max_positions": 5,
        },
        mode="paper",
        decision_log=decision_log,
    )

    fraction, details = manager.calculate_position_size(
        "BTC/USDT",
        0.75,
        _sample_market(),
        _sample_portfolio(),
        return_details=True,
    )

    assert 0.0 <= fraction <= 1.0
    assert details["recommended_size"] == fraction
    tail = decision_log.tail(limit=1)
    assert tail, "RiskDecisionLog powinien zawierać wpis"
    entry = tail[0]
    assert entry["symbol"] == "BTC/USDT"
    assert "allowed" in entry
    assert entry["metadata"]["source"] == "risk_manager_adapter"
    assert entry["metadata"]["mode"] == "paper"


def test_risk_manager_emits_alert_on_denial() -> None:
    dispatcher = get_alert_dispatcher()
    received = []

    def _handler(event) -> None:
        if event.source == "risk":
            received.append(event)

    token = dispatcher.register(_handler, name="risk-manager-test")
    try:
        manager = RiskManager(
            config={"max_risk_per_trade": 0.01, "max_positions": 1},
            mode="paper",
        )
        manager.calculate_position_size(
            "BTC/USDT",
            0.5,
            _sample_market(),
            {"capital": 0.0},
        )
    finally:
        dispatcher.unregister(token)

    assert received, "Brak zarejestrowanych alertów ryzyka"
    severities = {event.severity for event in received}
    assert AlertSeverity.WARNING in severities


class _DummyDB:
    def __init__(self) -> None:
        self.logged: list[Mapping[str, object]] = []

    async def log_risk_limit(self, payload: Mapping[str, object]) -> None:
        self.logged.append(dict(payload))


def test_risk_manager_logs_snapshot_to_db() -> None:
    db = _DummyDB()
    manager = RiskManager(
        config={
            "max_risk_per_trade": 0.05,
            "max_daily_loss_pct": 0.2,
            "max_positions": 5,
        },
        mode="paper",
        db_manager=db,
    )

    fraction = manager.calculate_position_size(
        "ETH/USDT",
        0.6,
        _sample_market(),
        _sample_portfolio(),
        return_details=False,
    )

    assert db.logged, "Powinien zostać zapisany snapshot limitu ryzyka"
    snapshot = db.logged[0]
    assert snapshot["symbol"] == "ETH/USDT"
    assert snapshot["mode"] == "paper"
    assert pytest.approx(snapshot["recommended_size"]) == fraction
    assert "profile" in snapshot["details"]

