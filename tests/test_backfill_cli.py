import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config.loader import load_core_config
from bot_core.config.models import (
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
)
from bot_core.exchanges.base import Environment
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter

import scripts.backfill as backfill


def test_build_public_source_supports_all_exchanges_from_universe():
    config = load_core_config("config/core.yaml")
    universe = config.instrument_universes["core_multi_exchange"]

    expected_adapters = {
        "binance_spot": BinanceSpotAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "kraken_spot": KrakenSpotAdapter,
        "kraken_futures": KrakenFuturesAdapter,
        "zonda_spot": ZondaSpotAdapter,
    }

    exchanges = {
        exchange_name
        for instrument in universe.instruments
        for exchange_name in instrument.exchange_symbols.keys()
    }

    for exchange in exchanges:
        source = backfill._build_public_source(exchange, Environment.PAPER)
        assert isinstance(source.exchange_adapter, expected_adapters[exchange])
        assert source.exchange_adapter.credentials.key_id == "public"
        assert source.exchange_adapter.credentials.environment == Environment.PAPER


def test_build_interval_plans_assigns_refresh_seconds_and_lookbacks():
    universe = InstrumentUniverseConfig(
        name="test",
        description="test",
        instruments=(
            InstrumentConfig(
                name="BTC_USDT",
                base_asset="BTC",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"binance_spot": "BTCUSDT"},
                backfill_windows=(
                    InstrumentBackfillWindow(interval="1d", lookback_days=365),
                    InstrumentBackfillWindow(interval="1h", lookback_days=30),
                ),
            ),
        ),
    )

    plans, symbols = backfill._build_interval_plans(
        universe=universe,
        exchange_name="binance_spot",
        incremental_lookback_days=7,
        refresh_overrides={"1h": 120},
    )

    assert symbols == {"BTCUSDT"}
    assert plans["1d"].refresh_seconds == backfill._DEFAULT_REFRESH_SECONDS["1d"]
    assert plans["1d"].incremental_lookback_ms == 7 * backfill._MILLISECONDS_IN_DAY

    assert plans["1h"].refresh_seconds == 120
    assert plans["1h"].incremental_lookback_ms == 7 * backfill._MILLISECONDS_IN_DAY


class _DummyScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.stopped = False

    def add_job(self, **kwargs):
        self.jobs.append(kwargs)

    async def run_forever(self):
        return

    def stop(self):
        self.stopped = True


def test_run_scheduler_uses_interval_specific_frequency():
    scheduler = _DummyScheduler()
    plans = {
        "1d": backfill._IntervalPlan(
            symbols={"BTCUSDT"},
            backfill_start_ms=0,
            incremental_lookback_ms=backfill._MILLISECONDS_IN_DAY,
            refresh_seconds=backfill._DEFAULT_REFRESH_SECONDS["1d"],
        ),
        "1h": backfill._IntervalPlan(
            symbols={"ETHUSDT"},
            backfill_start_ms=0,
            incremental_lookback_ms=3 * backfill._MILLISECONDS_IN_DAY,
            refresh_seconds=900,
        ),
    }

    asyncio.run(
        backfill._run_scheduler(
            scheduler=scheduler,
            plans=plans,
            refresh_seconds=600,
        )
    )

    assert scheduler.stopped is True
    assert len(scheduler.jobs) == 2

    job_daily = next(job for job in scheduler.jobs if job["interval"] == "1d")
    job_hourly = next(job for job in scheduler.jobs if job["interval"] == "1h")

    assert job_daily["frequency_seconds"] == backfill._DEFAULT_REFRESH_SECONDS["1d"]
    assert job_daily["lookback_ms"] == backfill._MILLISECONDS_IN_DAY

    assert job_hourly["frequency_seconds"] == 900
    assert job_hourly["lookback_ms"] == 3 * backfill._MILLISECONDS_IN_DAY
