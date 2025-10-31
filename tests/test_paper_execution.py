"""Testy symulatora paper trading."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


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


def test_short_requires_sufficient_margin() -> None:
    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }
    service = PaperTradingExecutionService(markets, initial_balances={"USDT": 0.0, "BTC": 0.0})
    context = ExecutionContext(
        portfolio_id="paper-1",
        risk_profile="balanced",
        environment="paper",
        metadata={"leverage": "1"},
    )
    request = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=0.5,
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
    assert "leverage" in first and first["leverage"] >= 1.0
    assert "position_value" in first and first["position_value"] > 0


def test_short_trade_with_leverage_and_fees() -> None:
    taker_fee = 0.0006
    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }
    service = PaperTradingExecutionService(markets, initial_balances={"USDT": 10_000.0, "BTC": 0.0})
    context = ExecutionContext(
        portfolio_id="paper-1",
        risk_profile="aggressive",
        environment="paper",
        metadata={"leverage": "3"},
    )

    sell_request = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=1.0,
        order_type="market",
        price=20_000.0,
    )

    sell_result = service.execute(sell_request, context)
    assert sell_result.status == "filled"
    sell_price = sell_result.avg_price or 0.0
    assert sell_result.raw_response["fee"] == pytest.approx(sell_price * taker_fee, rel=1e-5)

    balances_after_sell = service.balances()
    expected_after_sell = 10_000.0 + sell_price * (1 - taker_fee) - (sell_price / 3)
    assert balances_after_sell["USDT"] == pytest.approx(expected_after_sell, rel=1e-5)

    positions = service.short_positions()
    assert "BTCUSDT" in positions
    short = positions["BTCUSDT"]
    assert short["quantity"] == pytest.approx(1.0)
    assert short["margin"] == pytest.approx(sell_price / 3, rel=1e-5)
    assert short["leverage"] == pytest.approx(3.0, rel=1e-4)
    assert short["entry_price"] == pytest.approx(sell_price, rel=1e-5)

    ledger_entry = list(service.ledger())[-1]
    assert ledger_entry["leverage"] == pytest.approx(3.0, rel=1e-4)
    assert ledger_entry["position_value"] == pytest.approx(sell_price, rel=1e-5)

    buy_request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        price=19_000.0,
    )

    buy_result = service.execute(buy_request, context)
    assert buy_result.status == "filled"
    buy_price = buy_result.avg_price or 0.0
    assert buy_result.raw_response["fee"] == pytest.approx(buy_price * taker_fee, rel=1e-5)

    assert not service.short_positions()
    balances_after_close = service.balances()
    expected_after_close = 10_000.0 + sell_price * (1 - taker_fee) - buy_price * (1 + taker_fee)
    assert balances_after_close["USDT"] == pytest.approx(expected_after_close, rel=1e-5)

    final_entry = list(service.ledger())[-1]
    assert final_entry["position_value"] == pytest.approx(0.0, abs=1e-8)
    assert final_entry["leverage"] == pytest.approx(1.0, abs=1e-8)


def test_ledger_persists_entries_to_disk(tmp_path: Path) -> None:
    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
        )
    }
    ledger_dir = tmp_path / "ledger"
    time_iter = itertools.repeat(1_700_000_000.0)

    service = PaperTradingExecutionService(
        markets,
        initial_balances={"USDT": 50_000.0},
        time_source=lambda: next(time_iter),
        ledger_directory=ledger_dir,
        ledger_fsync=True,
    )
    context = _default_context()
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.5,
        order_type="market",
        price=20_000.0,
    )

    service.execute(request, context)

    files = sorted(ledger_dir.glob("*.jsonl"))
    assert files, "Powinien powstać plik ledger JSONL"
    payload = files[0].read_text(encoding="utf-8").strip()
    assert payload
    record = json.loads(payload)
    assert record["symbol"] == "BTCUSDT"
    assert record["status"] == "filled"
    assert float(record["quantity"]) == pytest.approx(0.5)


def test_ledger_retention_removes_old_files(tmp_path: Path) -> None:
    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
        )
    }
    ledger_dir = tmp_path / "ledger"
    day_one = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    day_four = datetime(2024, 1, 4, tzinfo=timezone.utc).timestamp()
    time_values = iter(
        [
            day_one,
            day_one,
            day_one,
            day_four,
            day_four,
            day_four,
        ]
    )

    service = PaperTradingExecutionService(
        markets,
        initial_balances={"USDT": 50_000.0},
        time_source=lambda: next(time_values),
        ledger_directory=ledger_dir,
        ledger_retention_days=2,
    )
    context = _default_context()

    service.execute(
        OrderRequest(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.2,
            order_type="market",
            price=20_000.0,
        ),
        context,
    )

    files_after_first = sorted(ledger_dir.glob("*.jsonl"))
    assert files_after_first, "Pierwszy zapis powinien utworzyć plik ledger"

    service.execute(
        OrderRequest(
            symbol="BTCUSDT",
            side="sell",
            quantity=0.1,
            order_type="market",
            price=21_000.0,
        ),
        context,
    )

    files_after_second = sorted(ledger_dir.glob("*.jsonl"))
    assert files_after_second, "Powinien istnieć co najmniej jeden plik ledger"
    assert len(files_after_second) <= 2
    filenames = {path.name for path in files_after_second}
    assert all(name.startswith("ledger-") for name in filenames)
    assert any("20240104" in name for name in filenames)
    assert not any("20240101" in name for name in filenames), "Plik starszy niż retencja powinien zostać usunięty"
