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

    audits = await manager.fetch_risk_audits(symbol="BTC/USDT")
    assert audits and audits[0]["reason"] == "daily_loss_limit"

