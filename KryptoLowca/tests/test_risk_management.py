"""Testy integracji z modułem ryzyka i bazą danych `bot_core`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.database.manager import DatabaseManager
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import InMemoryRiskRepository, ThresholdRiskEngine
from bot_core.risk.profiles.balanced import BalancedProfile


def _account(equity: float) -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity * 0.9,
        maintenance_margin=equity * 0.1,
    )


def _order(notional: float) -> OrderRequest:
    price = 20_000.0
    quantity = notional / price
    return OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=quantity,
        order_type="limit",
        price=price,
        atr=150.0,
        stop_price=price * 0.98,
        metadata={"atr": 150.0, "stop_price": price * 0.98},
    )


@pytest.mark.asyncio
async def test_threshold_risk_engine_enforces_daily_loss() -> None:
    engine = ThresholdRiskEngine(repository=InMemoryRiskRepository())
    engine.register_profile(BalancedProfile())

    # Pierwsza transakcja powinna przejść – brak ekspozycji i limity w normie.
    account = _account(100_000.0)
    request = _order(4_000.0)
    decision = engine.apply_pre_trade_checks(request, account=account, profile_name="balanced")
    assert decision.allowed is True

    # Aktualizujemy stan o stratę dzienną przekraczającą dopuszczalny limit.
    engine.on_fill(
        profile_name="balanced",
        symbol="BTC/USDT",
        side="buy",
        position_value=4_000.0,
        pnl=-5_000.0,
        timestamp=datetime.now(timezone.utc),
    )

    # Symulujemy spadek kapitału i sprawdzamy, że kolejne zlecenie zostaje zablokowane.
    degraded_account = _account(94_000.0)
    denial = engine.apply_pre_trade_checks(
        _order(2_000.0),
        account=degraded_account,
        profile_name="balanced",
    )

    assert denial.allowed is False
    assert "limit" in (denial.reason or "").lower()
    assert engine.should_liquidate(profile_name="balanced") is True


@pytest.mark.asyncio
async def test_database_manager_persists_logs(tmp_path: Path) -> None:
    db_path = tmp_path / "risk.db"
    manager = DatabaseManager(f"sqlite+aiosqlite:///{db_path}")
    await manager.init_db()

    user_id = await manager.ensure_user("tester@example.com")
    entry_id = await manager.log(user_id, "info", "Risk metrics recalculated", category="risk")
    assert entry_id > 0

    rows = await manager.fetch_logs(level="info", source="risk")
    assert rows, "Oczekiwano co najmniej jednego logu ryzyka"
    payload = rows[0]
    assert payload["message"] == "Risk metrics recalculated"
    assert payload["level"] == "INFO"

    # Zapisanie audytu ryzyka i pobranie go z bazy.
    audit_id = await manager.log_risk_audit(
        {
            "symbol": "BTC/USDT",
            "state": "warning",
            "reason": "daily_loss_limit",
            "fraction": 0.05,
            "mode": "paper",
            "ts": datetime.now(timezone.utc).timestamp(),
            "details": {"max_daily_loss_pct": 0.015},
        }
    )
    assert audit_id > 0


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

