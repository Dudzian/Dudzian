from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from bot_core.alerts import DefaultAlertRouter
from bot_core.alerts.audit import InMemoryAlertAuditLog
from bot_core.config.models import ControllerRuntimeConfig
from bot_core.data.base import OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv.backfill import OHLCVBackfillService
from bot_core.data.ohlcv.cache import CachedOHLCVSource
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment, OrderResult
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.controller import ControllerSignal, DailyTrendController, TradingController
from bot_core.runtime.realtime import DailyTrendRealtimeRunner
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy
from tests._alert_channel_helpers import CollectingChannel
from tests._daily_trend_helpers import FixtureSource, InMemoryStorage, build_core_config


def _build_controller(position_size: float = 0.2) -> tuple[DailyTrendController, PaperTradingExecutionService, AccountSnapshot, Sequence[Sequence[float]]]:
    day_ms = 86_400_000
    start_time = 1_600_000_000_000
    candles = [
        [float(start_time + i * day_ms), 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 10.0]
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
        position_size=position_size,
    )

    return controller, execution_service, account_snapshot, candles


class _StubRealtimeController:
    def __init__(self, interval: str, tick_seconds: float) -> None:
        self.interval = interval
        self.tick_seconds = tick_seconds
        self.calls: list[tuple[int, int]] = []

    def collect_signals(self, *, start: int, end: int) -> Sequence[ControllerSignal]:
        self.calls.append((start, end))
        return []


class _StubTradingController:
    def __init__(self) -> None:
        self.health_checks = 0

    def maybe_report_health(self) -> None:
        self.health_checks += 1

    def process_signals(self, signals: Sequence[ControllerSignal]) -> list[OrderResult]:
        return []


class _FailingRealtimeController(_StubRealtimeController):
    def __init__(self, interval: str, tick_seconds: float) -> None:
        super().__init__(interval, tick_seconds)
        self.failures = 0

    def collect_signals(self, *, start: int, end: int) -> Sequence[ControllerSignal]:
        del start, end
        self.failures += 1
        raise RuntimeError("boom")


def test_realtime_runner_uses_monthly_interval_lookback() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    controller = _StubRealtimeController(interval="1M", tick_seconds=60.0)
    trading_controller = _StubTradingController()

    runner = DailyTrendRealtimeRunner(
        controller=controller,
        trading_controller=trading_controller,
        history_bars=5,
        clock=lambda: now,
    )

    results = runner.run_once()

    assert results == []
    assert trading_controller.health_checks == 1
    assert controller.calls

    start_ms, end_ms = controller.calls[0]
    expected_end_ms = int(now.timestamp() * 1000)
    expected_lookback_ms = int(2_592_000.0 * 5 * 1000)

    assert end_ms == expected_end_ms
    assert end_ms - start_ms == expected_lookback_ms


def test_realtime_runner_triggers_alerts_and_execution() -> None:
    controller, execution_service, account_snapshot, candles = _build_controller()

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = CollectingChannel()
    router.register(channel)

    trading_controller = TradingController(
        risk_engine=controller.risk_engine,
        execution_service=execution_service,
        alert_router=router,
        account_snapshot_provider=controller.account_loader,
        portfolio_id=controller.execution_context.portfolio_id,
        environment=controller.execution_context.environment,
        risk_profile=controller.execution_context.risk_profile,
        health_check_interval=0.0,
    )

    runner = DailyTrendRealtimeRunner(
        controller=controller,
        trading_controller=trading_controller,
        history_bars=10,
        clock=lambda: datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc),
    )

    results = runner.run_once()

    assert len(results) == 1
    assert results[0].status == "filled"
    assert results[0].filled_quantity == pytest.approx(controller.position_size)

    severities = [message.severity for message in channel.messages]
    assert "info" in severities
    assert channel.messages[0].category == "strategy"

    balances = execution_service.balances()
    assert balances["USDT"] < account_snapshot.available_margin


def test_run_forever_respects_stop_condition_and_sleep() -> None:
    controller = _StubRealtimeController(interval="1d", tick_seconds=180.0)
    trading_controller = _StubTradingController()

    class _Clock:
        def __init__(self) -> None:
            self.current = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def __call__(self) -> datetime:
            value = self.current
            self.current = value + timedelta(seconds=60)
            return value

    sleep_calls: list[float] = []

    runner = DailyTrendRealtimeRunner(
        controller=controller,
        trading_controller=trading_controller,
        clock=_Clock(),
        sleep=lambda seconds: sleep_calls.append(seconds),
        history_bars=3,
    )

    runner.run_forever(stop_condition=lambda: len(controller.calls) >= 2)

    assert len(controller.calls) == 2
    assert trading_controller.health_checks == 2
    assert sleep_calls == [pytest.approx(60.0)]


def test_run_forever_uses_error_handler_and_continues() -> None:
    controller = _FailingRealtimeController(interval="1d", tick_seconds=180.0)
    trading_controller = _StubTradingController()

    class _Clock:
        def __init__(self) -> None:
            self.current = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def __call__(self) -> datetime:
            value = self.current
            self.current = value + timedelta(seconds=60)
            return value

    sleep_calls: list[float] = []
    captured: list[Exception] = []

    runner = DailyTrendRealtimeRunner(
        controller=controller,
        trading_controller=trading_controller,
        clock=_Clock(),
        sleep=lambda seconds: sleep_calls.append(seconds),
        on_cycle_error=lambda exc: captured.append(exc),
        min_sleep_seconds=5.0,
    )

    runner.run_forever(max_cycles=2)

    assert controller.failures == 2
    assert len(captured) == 2
    assert all(isinstance(exc, RuntimeError) for exc in captured)
    assert sleep_calls == [pytest.approx(60.0)]
