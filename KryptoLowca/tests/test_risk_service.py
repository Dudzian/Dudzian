import datetime

import pytest

from KryptoLowca.core.services.risk_service import RiskService
from KryptoLowca.strategies.base import StrategyContext, StrategyMetadata, StrategySignal


@pytest.fixture
def strategy_context() -> StrategyContext:
    metadata = StrategyMetadata(name="Test", description="")
    metadata.risk_level = "balanced"
    return StrategyContext(
        symbol="BTC/USDT",
        timeframe="1m",
        portfolio_value=1000.0,
        position=0.0,
        timestamp=datetime.datetime.utcnow(),
        metadata=metadata,
        extra={"mode": "demo"},
    )


def test_risk_service_limits_portfolio_exposure(strategy_context: StrategyContext) -> None:
    service = RiskService(
        max_position_notional_pct=0.5,
        max_daily_loss_pct=0.5,
        max_portfolio_risk_pct=0.6,
    )
    signal = StrategySignal(symbol="BTC/USDT", action="BUY", confidence=0.9, size=400.0)
    market_state = {
        "price": 100.0,
        "daily_loss_pct": 0.0,
        "portfolio_exposure_pct": 0.55,
        "portfolio_value": 1000.0,
    }

    assessment = service.assess(signal, strategy_context, market_state)

    assert assessment.allow is True
    assert assessment.size == pytest.approx(50.0)


def test_risk_service_rejects_when_position_limit_reached(strategy_context: StrategyContext) -> None:
    service = RiskService(max_positions=1)
    signal = StrategySignal(symbol="BTC/USDT", action="BUY", confidence=0.8, size=50.0)
    market_state = {
        "price": 100.0,
        "daily_loss_pct": 0.0,
        "open_positions": 1,
        "portfolio_value": 1000.0,
    }

    assessment = service.assess(signal, strategy_context, market_state)

    assert assessment.allow is False
    assert assessment.reason == "Przekroczony limit liczby pozycji"


def test_risk_service_stops_on_emergency_drawdown(strategy_context: StrategyContext) -> None:
    service = RiskService(emergency_stop_drawdown_pct=0.2)
    signal = StrategySignal(symbol="BTC/USDT", action="BUY", confidence=0.5, size=10.0)
    market_state = {
        "price": 100.0,
        "daily_loss_pct": 0.0,
        "drawdown_pct": 0.25,
        "portfolio_value": 1000.0,
    }

    assessment = service.assess(signal, strategy_context, market_state)

    assert assessment.allow is False
    assert assessment.reason == "Osiągnięto limit awaryjnego drawdownu"
