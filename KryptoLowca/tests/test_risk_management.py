import asyncio

import numpy as np
import pandas as pd
import pytest

from KryptoLowca.alerts import AlertSeverity, get_alert_dispatcher
from KryptoLowca.managers.database_manager import DatabaseManager
from KryptoLowca.managers.risk_manager_adapter import RiskManager
from KryptoLowca.risk_management import RiskLevel, RiskManagement


async def test_risk_manager_logs_snapshot(tmp_path):
    db_path = tmp_path / "risk.db"
    db = DatabaseManager(f"sqlite+aiosqlite:///{db_path}")
    await db.init_db()

    adapter = RiskManager(config={}, db_manager=db, mode="paper")

    df = pd.DataFrame(
        {
            "close": np.linspace(100, 105, 180),
            "volume": np.linspace(1_000_000, 900_000, 180),
        }
    )

    size, details = adapter.calculate_position_size(
        "BTC/USDT",
        {"strength": 0.6, "confidence": 0.7},
        df,
        {"ETH/USDT": {"size": 0.1, "volatility": 0.25}},
        return_details=True,
    )

    assert 0.0 <= size <= 1.0
    assert "recommended_size" in details

    await asyncio.sleep(0)

    rows = await db.fetch_risk_limits(symbol="BTC/USDT")
    assert rows, "Oczekiwano zapisu limitów ryzyka w bazie"
    latest = rows[0]
    assert latest["recommended_size"] == pytest.approx(size)
    assert latest["mode"] == "paper"


def test_risk_metrics_and_alert_dispatch():
    dispatcher = get_alert_dispatcher()
    received = []

    def _handler(event):
        if event.source == "risk":
            received.append(event)

    token = dispatcher.register(_handler, name="test-risk")
    try:
        rm = RiskManagement({"max_portfolio_risk": 0.2, "max_risk_per_trade": 0.05})
        rm.portfolio_value_history = [100, 102, 98, 95, 90, 87, 85, 84, 82, 80, 78, 77]

        portfolio = {
            "BTC/USDT": {"size": 0.12, "volatility": 0.3},
            "ETH/USDT": {"size": 0.11, "volatility": 0.28},
        }
        base_prices = np.linspace(100, 110, 300)
        df = pd.DataFrame(
            {
                "close": base_prices,
                "volume": np.linspace(800_000, 700_000, 300),
            }
        )
        market = {symbol: df.copy() for symbol in portfolio}

        metrics = rm.calculate_risk_metrics(portfolio, market)
        assert isinstance(metrics.var_95, float)
        assert isinstance(metrics.risk_level, RiskLevel)

        emergency = rm.emergency_risk_check(60_000, 100_000, portfolio)
        assert isinstance(emergency, dict)
        assert emergency["actions_required"], "Powinny istnieć działania awaryjne"

        assert any(evt.source == "risk" for evt in received), "Alerty ryzyka powinny zostać zarejestrowane"
        assert any(
            evt.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL) for evt in received if evt.source == "risk"
        )
    finally:
        dispatcher.unregister(token)
