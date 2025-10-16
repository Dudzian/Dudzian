from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from bot_core.tco import TCOAnalyzer, TradeCostEvent


def _event(
    *,
    strategy: str,
    risk_profile: str,
    quantity: str,
    price: str,
    commission: str,
    slippage: str,
    funding: str,
    other: str = "0",
) -> TradeCostEvent:
    return TradeCostEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        strategy=strategy,
        risk_profile=risk_profile,
        instrument="BTC/USDT",
        exchange="binance",
        side="buy",
        quantity=Decimal(quantity),
        price=Decimal(price),
        commission=Decimal(commission),
        slippage=Decimal(slippage),
        funding=Decimal(funding),
        other=Decimal(other),
    )


def test_analyzer_aggregates_costs_and_alerts() -> None:
    events = [
        _event(
            strategy="mean_reversion",
            risk_profile="balanced",
            quantity="0.5",
            price="20000",
            commission="5",
            slippage="2",
            funding="0.5",
        ),
        _event(
            strategy="mean_reversion",
            risk_profile="aggressive",
            quantity="0.4",
            price="21000",
            commission="6",
            slippage="3",
            funding="1.0",
        ),
    ]
    analyzer = TCOAnalyzer(cost_limit_bps=Decimal("5"))
    report = analyzer.analyze(events, metadata={"environment": "paper"})

    assert report.metadata["events_count"] == 2
    assert report.metadata["strategy_count"] == 1
    total = report.total
    assert total.trade_count == 2
    assert float(total.notional) == float(Decimal("0.5") * Decimal("20000") + Decimal("0.4") * Decimal("21000"))
    assert float(total.breakdown.commission) == 11.0
    assert float(total.breakdown.slippage) == 5.0
    assert float(total.breakdown.funding) == 1.5
    assert report.alerts  # koszt przekracza limit 5 bps

    summary = report.strategies["mean_reversion"]
    assert summary.total.trade_count == 2
    assert "balanced" in summary.profiles
    balanced = summary.profiles["balanced"]
    assert balanced.trade_count == 1
    assert balanced.breakdown.commission == Decimal("5")

    payload = report.to_dict()
    assert payload["metadata"]["environment"] == "paper"
    assert payload["total"]["trade_count"] == 2
