from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from bot_core.alerts import DefaultAlertRouter
from bot_core.alerts.audit import InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel, AlertMessage
from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    EnvironmentConfig,
    InstrumentUniverseConfig,
    RiskProfileConfig,
)
from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv.backfill import OHLCVBackfillService
from bot_core.data.ohlcv.cache import CachedOHLCVSource
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment, OrderResult
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.controller import DailyTrendController, TradingController
from bot_core.runtime.realtime import DailyTrendRealtimeRunner
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy


class CollectingChannel(AlertChannel):
    name = "collector"

    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:
        return {"status": "ok"}


class _InMemoryStorage(CacheStorage):
    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Sequence[Sequence[float]]]] = {}
        self._metadata: dict[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._store[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        try:
            rows = self._store[key]["rows"]
        except KeyError:
            return None
        if not rows:
            return None
        return float(rows[-1][0])


@dataclass(slots=True)
class _FixtureSource(DataSource):
    rows: Sequence[Sequence[float]]

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        filtered = [row for row in self.rows if request.start <= float(row[0]) <= request.end]
        limit = request.limit or len(filtered)
        return OHLCVResponse(
            columns=("open_time", "open", "high", "low", "close", "volume"),
            rows=filtered[:limit],
        )

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover
        del symbols, intervals


def _core_config(runtime: ControllerRuntimeConfig, environment_name: str, risk_profile: str) -> CoreConfig:
    return CoreConfig(
        environments={
            environment_name: EnvironmentConfig(
                name=environment_name,
                exchange="paper",
                environment=Environment.PAPER,
                keychain_key="paper",
                data_cache_path="./var/data",
                risk_profile=risk_profile,
                alert_channels=(),
            )
        },
        risk_profiles={
            risk_profile: RiskProfileConfig(
                name=risk_profile,
                max_daily_loss_pct=1.0,
                max_position_pct=1.0,
                target_volatility=0.0,
                max_leverage=10.0,
                stop_loss_atr_multiple=2.0,
                max_open_positions=10,
                hard_drawdown_pct=1.0,
            )
        },
        instrument_universes={
            "default": InstrumentUniverseConfig(name="default", description="", instruments=())
        },
        strategies={},
        reporting={},
        sms_providers={},
        telegram_channels={},
        email_channels={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
        runtime_controllers={"daily_trend": runtime},
    )


def _build_controller(position_size: float = 0.2) -> tuple[DailyTrendController, PaperTradingExecutionService, AccountSnapshot, Sequence[Sequence[float]]]:
    day_ms = 86_400_000
    start_time = 1_600_000_000_000
    candles = [
        [float(start_time + i * day_ms), 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 10.0]
        for i in range(5)
    ]
    candles.append([float(start_time + 5 * day_ms), 107.0, 110.0, 106.0, 108.0, 12.0])

    storage = _InMemoryStorage()
    source = _FixtureSource(rows=candles)
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
    core_cfg = _core_config(runtime_cfg, "paper", "paper_risk")

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
