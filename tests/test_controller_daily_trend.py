from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import mock

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from bot_core.config.models import ControllerRuntimeConfig
from bot_core.data.base import OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv.backfill import OHLCVBackfillService
from bot_core.data.ohlcv.cache import CachedOHLCVSource
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.controller import ControllerSignal, DailyTrendController
from bot_core.strategies.base import MarketSnapshot, StrategySignal
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy
from tests._daily_trend_helpers import FixtureSource, InMemoryStorage, build_core_config

def test_daily_trend_controller_executes_signal() -> None:
    day_ms = 86_400_000
    start_time = 1_600_000_000_000
    candles = [
        [float(start_time + i * day_ms), 100.0 + i, 100.5 + i, 99.5 + i, 100.0 + i, 10.0]
        for i in range(5)
    ]
    candles.append([float(start_time + 5 * day_ms), 107.0, 110.0, 106.0, 108.0, 12.0])

    storage = InMemoryStorage()
    source = FixtureSource(rows=candles)
    cached = CachedOHLCVSource(storage=storage, upstream=source)
    backfill = OHLCVBackfillService(cached, chunk_limit=10)

    settings = DailyTrendMomentumSettings(
        fast_ma=3,
        slow_ma=5,
        breakout_lookback=4,
        momentum_window=3,
        atr_window=3,
        atr_multiplier=1.5,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    strategy = DailyTrendMomentumStrategy(settings)

    runtime_cfg = ControllerRuntimeConfig(tick_seconds=60.0, interval="1d")
    core_cfg = build_core_config(runtime_cfg, "paper", "paper_risk")

    risk_engine = ThresholdRiskEngine()
    profile = ManualProfile(
        name="paper_risk",
        max_positions=5,
        max_leverage=5.0,
        drawdown_limit=1.0,
        daily_loss_limit=1.0,
        max_position_pct=1.0,
        target_volatility=0.0,
        stop_loss_atr_multiple=2.0,
    )
    risk_engine.register_profile(profile)

    execution_service = PaperTradingExecutionService(
        {"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT", min_notional=0.0)},
        initial_balances={"USDT": 100_000.0},
        maker_fee=0.0,
        taker_fee=0.0,
        slippage_bps=0.0,
    )

    account_snapshot = AccountSnapshot(
        balances={"USDT": 100_000.0},
        total_equity=100_000.0,
        available_margin=100_000.0,
        maintenance_margin=0.0,
    )

    controller = DailyTrendController(
        core_config=core_cfg,
        environment_name="paper",
        controller_name="daily_trend",
        symbols=("BTCUSDT",),
        backfill_service=backfill,
        data_source=cached,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_service=execution_service,
        account_loader=lambda: account_snapshot,
        execution_context=ExecutionContext(
            portfolio_id="paper-demo",
            risk_profile="paper_risk",
            environment=Environment.PAPER.value,
            metadata={},
        ),
        position_size=0.1,
    )

    results = controller.run_cycle(start=candles[0][0], end=candles[-1][0])

    assert len(results) == 1
    result = results[0]
    assert result.status == "filled"
    assert result.filled_quantity == 0.1
    assert controller.tick_seconds == runtime_cfg.tick_seconds
    assert controller.interval == runtime_cfg.interval
    assert risk_engine.should_liquidate(profile_name="paper_risk") is False


def test_daily_trend_controller_scales_quantity_after_risk_adjustment() -> None:
    storage = InMemoryStorage()
    source = FixtureSource(
        rows=[
            [1_700_000_000_000.0, 20_000.0, 20_050.0, 19_950.0, 20_000.0, 5.0],
            [1_700_086_400_000.0, 20_100.0, 20_150.0, 20_000.0, 20_120.0, 5.0],
        ]
    )
    cached = CachedOHLCVSource(storage=storage, upstream=source)
    backfill = OHLCVBackfillService(cached, chunk_limit=10)

    runtime_cfg = ControllerRuntimeConfig(tick_seconds=60.0, interval="1d")
    core_cfg = build_core_config(runtime_cfg, "paper", "paper_risk")

    risk_engine = ThresholdRiskEngine()
    profile = ManualProfile(
        name="paper_risk",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.5,
        max_position_pct=0.03,
        target_volatility=0.0,
        stop_loss_atr_multiple=1.5,
    )
    risk_engine.register_profile(profile)

    execution_service = PaperTradingExecutionService(
        {"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT", min_notional=0.0)},
        initial_balances={"USDT": 100_000.0},
        maker_fee=0.0,
        taker_fee=0.0,
        slippage_bps=0.0,
    )

    account_snapshot = AccountSnapshot(
        balances={"USDT": 100_000.0},
        total_equity=100_000.0,
        available_margin=100_000.0,
        maintenance_margin=0.0,
    )

    controller = DailyTrendController(
        core_config=core_cfg,
        environment_name="paper",
        controller_name="daily_trend",
        symbols=("BTCUSDT",),
        backfill_service=backfill,
        data_source=cached,
        strategy=DailyTrendMomentumStrategy(DailyTrendMomentumSettings()),
        risk_engine=risk_engine,
        execution_service=execution_service,
        account_loader=lambda: account_snapshot,
        execution_context=ExecutionContext(
            portfolio_id="paper-demo",
            risk_profile="paper_risk",
            environment=Environment.PAPER.value,
            metadata={},
        ),
        position_size=1.0,
    )

    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=1_700_086_400_000,
        open=20_100.0,
        high=20_150.0,
        low=20_000.0,
        close=20_120.0,
        volume=4.0,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"quantity": 1.0, "price": 20_120.0, "order_type": "market"},
    )

    results = controller._handle_signals(snapshot, (signal,))

    assert len(results) == 1
    expected_qty = pytest.approx((0.03 * account_snapshot.total_equity) / 20_120.0, rel=1e-6)
    assert results[0].filled_quantity == expected_qty
    ledger_entries = list(execution_service.ledger())
    assert ledger_entries[-1]["quantity"] == pytest.approx(expected_qty, rel=1e-6)


def test_collect_signals_enriches_metadata() -> None:
    day_ms = 86_400_000
    start_time = 1_600_000_000_000
    candles = [
        [float(start_time + i * day_ms), 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 10.0]
        for i in range(6)
    ]

    storage = InMemoryStorage()
    source = FixtureSource(rows=candles)
    cached = CachedOHLCVSource(storage=storage, upstream=source)
    backfill = OHLCVBackfillService(cached, chunk_limit=10)

    settings = DailyTrendMomentumSettings(
        fast_ma=3,
        slow_ma=5,
        breakout_lookback=4,
        momentum_window=3,
        atr_window=3,
        atr_multiplier=1.5,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    strategy = DailyTrendMomentumStrategy(settings)

    runtime_cfg = ControllerRuntimeConfig(tick_seconds=60.0, interval="1d")
    core_cfg = build_core_config(runtime_cfg, "paper", "paper_risk")

    risk_engine = ThresholdRiskEngine()
    profile = ManualProfile(
        name="paper_risk",
        max_positions=5,
        max_leverage=5.0,
        drawdown_limit=1.0,
        daily_loss_limit=1.0,
        max_position_pct=1.0,
        target_volatility=0.0,
        stop_loss_atr_multiple=2.0,
    )
    risk_engine.register_profile(profile)

    execution_service = PaperTradingExecutionService(
        {"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT", min_notional=0.0)},
        initial_balances={"USDT": 100_000.0},
        maker_fee=0.0,
        taker_fee=0.0,
        slippage_bps=0.0,
    )

    account_snapshot = AccountSnapshot(
        balances={"USDT": 100_000.0},
        total_equity=100_000.0,
        available_margin=100_000.0,
        maintenance_margin=0.0,
    )

    controller = DailyTrendController(
        core_config=core_cfg,
        environment_name="paper",
        controller_name="daily_trend",
        symbols=("BTCUSDT",),
        backfill_service=backfill,
        data_source=cached,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_service=execution_service,
        account_loader=lambda: account_snapshot,
        execution_context=ExecutionContext(
            portfolio_id="paper-demo",
            risk_profile="paper_risk",
            environment=Environment.PAPER.value,
            metadata={},
        ),
        position_size=0.25,
    )

    collected = controller.collect_signals(start=candles[0][0], end=candles[-1][0])

    assert collected, "Oczekiwano co najmniej jednego sygnału z strategii"
    assert isinstance(collected[0], ControllerSignal)
    signal = collected[0].signal
    assert signal.metadata["quantity"] == pytest.approx(0.25)
    assert float(signal.metadata["price"]) == pytest.approx(collected[0].snapshot.close)
    assert signal.metadata["order_type"] == "market"


def test_handle_signals_preserves_metadata_for_adjustments() -> None:
    runtime_cfg = ControllerRuntimeConfig(tick_seconds=60.0, interval="1d")
    core_cfg = build_core_config(runtime_cfg, "paper", "paper_risk")

    risk_engine = ThresholdRiskEngine()
    profile = ManualProfile(
        name="paper_risk",
        max_positions=5,
        max_leverage=5.0,
        drawdown_limit=1.0,
        daily_loss_limit=1.0,
        max_position_pct=1.0,
        target_volatility=0.02,
        stop_loss_atr_multiple=1.5,
    )
    risk_engine.register_profile(profile)

    execution_service = PaperTradingExecutionService(
        {"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT", min_notional=0.0)},
        initial_balances={"USDT": 100_000.0},
        maker_fee=0.0,
        taker_fee=0.0,
        slippage_bps=0.0,
    )

    storage = InMemoryStorage()
    source = FixtureSource(
        rows=[
            [1_700_000_000_000.0, 20_000.0, 20_100.0, 19_900.0, 20_050.0, 5.0],
            [1_700_086_400_000.0, 20_050.0, 20_200.0, 19_950.0, 20_100.0, 6.0],
        ]
    )
    cached = CachedOHLCVSource(storage=storage, upstream=source)
    backfill = OHLCVBackfillService(cached, chunk_limit=10)

    account_snapshot = AccountSnapshot(
        balances={"USDT": 100_000.0},
        total_equity=100_000.0,
        available_margin=100_000.0,
        maintenance_margin=0.0,
    )

    controller = DailyTrendController(
        core_config=core_cfg,
        environment_name="paper",
        controller_name="daily_trend",
        symbols=("BTCUSDT",),
        backfill_service=backfill,
        data_source=cached,
        strategy=DailyTrendMomentumStrategy(DailyTrendMomentumSettings()),
        risk_engine=risk_engine,
        execution_service=execution_service,
        account_loader=lambda: account_snapshot,
        execution_context=ExecutionContext(
            portfolio_id="paper-demo",
            risk_profile="paper_risk",
            environment=Environment.PAPER.value,
            metadata={},
        ),
        position_size=1.0,
    )

    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=1_700_086_400_000,
        open=20_050.0,
        high=20_200.0,
        low=19_950.0,
        close=20_100.0,
        volume=5.0,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={
            "quantity": 5.0,
            "price": 20_100.0,
            "order_type": "market",
            "atr": 500.0,
            "stop_price": 19_100.0,
        },
    )

    with mock.patch.object(risk_engine, "apply_pre_trade_checks", wraps=risk_engine.apply_pre_trade_checks) as patched:
        results = controller._handle_signals(snapshot, (signal,))

    assert patched.call_count == 2
    first_request = patched.call_args_list[0].args[0]
    second_request = patched.call_args_list[1].args[0]

    assert first_request.metadata is not None
    assert second_request.metadata is not None
    assert first_request.metadata["atr"] == pytest.approx(500.0)
    assert first_request.metadata["stop_price"] == pytest.approx(19_100.0)
    assert second_request.metadata["atr"] == pytest.approx(500.0)
    assert second_request.metadata["stop_price"] == pytest.approx(19_100.0)

    assert results
