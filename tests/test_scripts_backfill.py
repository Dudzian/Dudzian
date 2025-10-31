"""Testy skryptu backfill sprawdzające harmonogram odświeżania."""
from __future__ import annotations

import asyncio
from pathlib import Path


from bot_core.config.models import (
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
)
from scripts.backfill import _MILLISECONDS_IN_DAY, _build_interval_plans, _run_scheduler


class _DummyScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, object]] = []
        self.run_called = False
        self.stop_called = False

    def add_job(
        self,
        *,
        symbols: tuple[str, ...],
        interval: str,
        lookback_ms: int,
        frequency_seconds: int,
        jitter_seconds: int = 0,
        name: str,
    ) -> None:
        self.jobs.append(
            {
                "symbols": symbols,
                "interval": interval,
                "lookback_ms": lookback_ms,
                "frequency_seconds": frequency_seconds,
                "jitter_seconds": jitter_seconds,
                "name": name,
            }
        )

    async def run_forever(self) -> None:
        self.run_called = True

    def stop(self) -> None:
        self.stop_called = True

def test_scheduler_uses_interval_specific_frequency_and_lookback() -> None:
    universe = InstrumentUniverseConfig(
        name="test_universe",
        description="",
        instruments=(
            InstrumentConfig(
                name="BTC_USDT",
                base_asset="BTC",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"binance_spot": "BTCUSDT"},
                backfill_windows=(
                    InstrumentBackfillWindow(interval="1d", lookback_days=10),
                    InstrumentBackfillWindow(interval="1h", lookback_days=2),
                ),
            ),
        ),
    )

    plans, _ = _build_interval_plans(
        universe=universe,
        exchange_name="binance_spot",
        incremental_lookback_days=5,
    )

    scheduler = _DummyScheduler()
    asyncio.run(
        _run_scheduler(
            scheduler=scheduler,
            plans=plans,
            refresh_seconds=60,
        )
    )

    assert scheduler.run_called is True
    assert scheduler.stop_called is True
    jobs_by_interval = {job["interval"]: job for job in scheduler.jobs}
    assert set(jobs_by_interval) == {"1d", "1h"}

    assert jobs_by_interval["1d"]["frequency_seconds"] == plans["1d"].refresh_seconds
    assert jobs_by_interval["1h"]["frequency_seconds"] == plans["1h"].refresh_seconds
    assert jobs_by_interval["1d"]["frequency_seconds"] != jobs_by_interval["1h"]["frequency_seconds"]

    assert jobs_by_interval["1d"]["lookback_ms"] == plans["1d"].incremental_lookback_ms
    assert jobs_by_interval["1h"]["lookback_ms"] == plans["1h"].incremental_lookback_ms
    assert jobs_by_interval["1d"]["lookback_ms"] != jobs_by_interval["1h"]["lookback_ms"]

    assert plans["1d"].refresh_seconds == 24 * 60 * 60
    assert plans["1h"].refresh_seconds == 15 * 60
    assert plans["1d"].incremental_lookback_ms == 5 * _MILLISECONDS_IN_DAY
    assert plans["1h"].incremental_lookback_ms == 2 * _MILLISECONDS_IN_DAY
