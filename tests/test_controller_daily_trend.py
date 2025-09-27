from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from bot_core.exchanges.base import AccountSnapshot, Environment
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.controller import DailyTrendController
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy


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
                keychain_key="paper",  # nieużywane w teście
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


def test_daily_trend_controller_executes_signal() -> None:
    day_ms = 86_400_000
    start_time = 1_600_000_000_000
    candles = [
        [float(start_time + i * day_ms), 100.0 + i, 100.5 + i, 99.5 + i, 100.0 + i, 10.0]
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
