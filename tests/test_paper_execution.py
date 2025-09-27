"""Testy symulatora paper trading."""
from __future__ import annotations

import pytest

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.execution import (  # type: ignore[import-not-found]
    ExecutionContext,
    InsufficientBalanceError,
    MarketMetadata,
    PaperTradingExecutionService,
)
from bot_core.exchanges.base import OrderRequest


def _default_service(**kwargs: object) -> PaperTradingExecutionService:
    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }
    balances = {"USDT": 100_000.0, "BTC": 1.0}
    return PaperTradingExecutionService(markets, initial_balances=balances, **kwargs)


def _default_context() -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="paper-1",
        risk_profile="balanced",
        environment="paper",
        metadata={},
    )


def test_buy_order_consumes_quote_balance() -> None:
    service = _default_service()
    context = _default_context()
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.5,
        order_type="market",
        price=20_000.0,
    )

    result = service.execute(request, context)

    assert result.status == "filled"
    balances = service.balances()
    assert balances["BTC"] > 1.0
    assert balances["USDT"] < 100_000.0


def test_sell_requires_position() -> None:
    service = _default_service()
    context = _default_context()
    request = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=5.0,
        order_type="market",
        price=20_000.0,
    )

    with pytest.raises(InsufficientBalanceError):
        service.execute(request, context)


def test_slippage_affects_price_directionally() -> None:
    service = _default_service(slippage_bps=10.0)
    context = _default_context()

    buy_request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        order_type="market",
        price=20_000.0,
    )
    sell_request = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=0.1,
        order_type="market",
        price=20_000.0,
    )

    buy_result = service.execute(buy_request, context)
    sell_result = service.execute(sell_request, context)

    assert buy_result.avg_price and sell_result.avg_price
    assert buy_result.avg_price > 20_000.0
    assert sell_result.avg_price < 20_000.0


def test_ledger_contains_audit_entries() -> None:
    service = _default_service()
    context = _default_context()

    service.execute(
        OrderRequest(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.2,
            order_type="limit",
            price=19_500.0,
        ),
        context,
    )

    entries = list(service.ledger())
    assert entries
    first = entries[0]
    assert first["symbol"] == "BTCUSDT"
    assert first["status"] == "filled"
    assert first["fee"] > 0
