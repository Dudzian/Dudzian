"""Testy referencyjnej ceny w PaperTradingExecutionService."""
from __future__ import annotations

import pytest

from bot_core.execution import ExecutionContext, MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import OrderRequest


def _service_with_market() -> PaperTradingExecutionService:
    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
        )
    }
    balances = {"USDT": 10_000.0, "BTC": 0.0}
    return PaperTradingExecutionService(
        markets,
        initial_balances=balances,
        slippage_bps=0.0,
    )


def test_executes_with_price_from_context_resolver() -> None:
    last_price = 25_100.0

    def resolver(symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return last_price

    service = _service_with_market()
    context = ExecutionContext(
        portfolio_id="resolver-test",
        risk_profile="balanced",
        environment="paper",
        metadata={},
        price_resolver=resolver,
    )
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        order_type="market",
        price=None,
    )

    result = service.execute(request, context)

    assert result.status == "filled"
    assert result.avg_price == pytest.approx(last_price)


def test_executes_with_price_from_service_resolver_when_context_missing() -> None:
    last_price = 18_450.0

    def resolver(symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return last_price

    service = PaperTradingExecutionService(
        {
            "BTCUSDT": MarketMetadata(
                base_asset="BTC",
                quote_asset="USDT",
                min_quantity=0.001,
                min_notional=10.0,
            )
        },
        initial_balances={"USDT": 5_000.0, "BTC": 0.0},
        slippage_bps=0.0,
        price_resolver=resolver,
    )
    context = ExecutionContext(
        portfolio_id="resolver-test",
        risk_profile="balanced",
        environment="paper",
        metadata={},
    )
    request = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=0.2,
        order_type="market",
        price=None,
    )

    result = service.execute(request, context)

    assert result.status == "filled"
    assert result.avg_price == pytest.approx(last_price)


def test_raises_when_no_reference_price_available() -> None:
    service = _service_with_market()
    context = ExecutionContext(
        portfolio_id="resolver-test",
        risk_profile="balanced",
        environment="paper",
        metadata={},
    )
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        order_type="market",
        price=None,
    )

    with pytest.raises(ValueError):
        service.execute(request, context)
