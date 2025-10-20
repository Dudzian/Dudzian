from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import (
    InsufficientBalanceError,
    MarketMetadata,
    PaperTradingExecutionService,
)
from bot_core.exchanges.base import OrderRequest


@pytest.fixture()
def paper_service(tmp_path: Path) -> Callable[..., tuple[PaperTradingExecutionService, Callable[..., ExecutionContext]]]:
    """Buduje symulator paper tradingu w konfiguracji zgodnej z bot_core."""

    def factory(
        *,
        balances: Mapping[str, float] | None = None,
        ledger_subdir: str = "ledger",
        min_notional: float = 25.0,
    ) -> tuple[PaperTradingExecutionService, Callable[..., ExecutionContext]]:
        directory = tmp_path / ledger_subdir
        service = PaperTradingExecutionService(
            markets={
                "BTC/USDT": MarketMetadata(
                    base_asset="BTC",
                    quote_asset="USDT",
                    min_quantity=0.001,
                    min_notional=min_notional,
                    step_size=0.001,
                    tick_size=0.1,
                )
            },
            initial_balances=balances or {"USDT": 10_000.0},
            maker_fee=0.0,
            taker_fee=0.0,
            slippage_bps=0.0,
            ledger_directory=directory,
            ledger_retention_days=30,
        )

        def make_context(**metadata: object) -> ExecutionContext:
            return ExecutionContext(
                portfolio_id="portfolio-test",
                risk_profile="balanced",
                environment="paper",
                metadata=metadata,
            )

        return service, make_context

    return factory


def test_market_order_flow_records_balances_and_ledger(
    paper_service: Callable[..., tuple[PaperTradingExecutionService, Callable[..., ExecutionContext]]]
) -> None:
    service, make_context = paper_service(ledger_subdir="flow")
    context = make_context()

    order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.5,
        order_type="market",
        price=100.0,
    )

    result = service.execute(order, context)

    assert result.status == "filled"
    balances = service.balances()
    assert pytest.approx(balances["BTC"], rel=1e-9) == 0.5
    assert pytest.approx(balances["USDT"], rel=1e-9) == 10_000.0 - 50.0

    ledger_entries = list(service.ledger())
    assert ledger_entries and ledger_entries[-1]["order_id"] == result.order_id
    assert ledger_entries[-1]["status"] == "filled"

    files = service.ledger_files()
    assert files, "powinien zostać utworzony plik ledger"
    payloads = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line]
    assert any(entry["order_id"] == result.order_id for entry in payloads)


def test_short_sell_allocates_margin(
    paper_service: Callable[..., tuple[PaperTradingExecutionService, Callable[..., ExecutionContext]]]
) -> None:
    service, make_context = paper_service(ledger_subdir="short")
    context = make_context(leverage=3)

    sell_order = OrderRequest(
        symbol="BTC/USDT",
        side="sell",
        quantity=1.0,
        order_type="limit",
        price=120.0,
    )

    result = service.execute(sell_order, context)

    assert result.status == "filled"
    shorts = service.short_positions()
    assert "BTC/USDT" in shorts
    position = shorts["BTC/USDT"]
    assert position["margin"] > 0.0
    assert position["quantity"] == pytest.approx(1.0)

    balances = service.balances()
    assert balances["USDT"] > 10_000.0


def test_min_notional_is_enforced(
    paper_service: Callable[..., tuple[PaperTradingExecutionService, Callable[..., ExecutionContext]]]
) -> None:
    service, make_context = paper_service(ledger_subdir="min", balances={"USDT": 1_000.0}, min_notional=200.0)
    context = make_context()

    tiny_order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=10.0,
    )

    with pytest.raises(ValueError):
        service.execute(tiny_order, context)


def test_buy_rejects_when_balance_insufficient(
    paper_service: Callable[..., tuple[PaperTradingExecutionService, Callable[..., ExecutionContext]]]
) -> None:
    service, make_context = paper_service(ledger_subdir="insufficient", balances={"USDT": 50.0})
    context = make_context()

    large_order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        price=100.0,
    )

    with pytest.raises(InsufficientBalanceError):
        service.execute(large_order, context)
