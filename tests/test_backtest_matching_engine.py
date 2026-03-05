from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot_core.backtest.models import BacktestFill
from bot_core.backtest.simulation import MatchingEngine


def _ts() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_matching_engine_respects_latency(latency_one_no_cost_scenario) -> None:
    engine = MatchingEngine(latency_one_no_cost_scenario)
    engine.submit_market_order(side="buy", size=1.0, index=0, timestamp=_ts())

    assert engine.process_bar(index=0, timestamp=_ts(), bar={"close": 100.0}) == []

    fills = engine.process_bar(index=1, timestamp=_ts(), bar={"close": 101.0})
    assert len(fills) == 1
    fill = fills[0]
    assert isinstance(fill, BacktestFill)
    assert fill.side == "buy"
    assert fill.partial is False
    assert fill.price == pytest.approx(101.0)


def test_matching_engine_partial_fills_until_complete(partial_fill_scenario) -> None:
    engine = MatchingEngine(partial_fill_scenario)
    engine.submit_market_order(side="sell", size=1.0, index=0, timestamp=_ts().replace(tzinfo=None))

    total_filled = 0.0
    completed = False
    for idx in range(60):
        fills = engine.process_bar(index=idx, timestamp=_ts(), bar={"close": 50.0 - idx})
        if not fills:
            continue
        total_filled += sum(fill.size for fill in fills)
        if fills[-1].partial is False:
            completed = True
            break

    assert completed is True
    assert total_filled == pytest.approx(1.0)


def test_matching_engine_applies_slippage_and_fees(slippage_fee_scenario) -> None:
    cfg = slippage_fee_scenario.to_matching_config()
    engine = MatchingEngine(slippage_fee_scenario)
    engine.submit_market_order(side="buy", size=2.0, index=5, timestamp=_ts())

    fills = engine.process_bar(index=5, timestamp=_ts(), bar={"close": 40.0})
    assert len(fills) == 1
    fill = fills[0]
    expected_slippage = 40.0 * (cfg.slippage_bps / 10_000.0)
    expected_price = 40.0 + expected_slippage
    expected_fee = abs(expected_price * 2.0) * (cfg.fee_bps / 10_000.0)
    assert fill.slippage == pytest.approx(expected_slippage * fill.size)
    assert fill.price == pytest.approx(expected_price)
    assert fill.fee == pytest.approx(expected_fee)
    assert fill.timestamp.tzinfo is timezone.utc
