from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.config.models import InstrumentBackfillWindow, InstrumentConfig, InstrumentUniverseConfig
from bot_core.data.ohlcv.manifest_report import generate_manifest_report, summarize_status
from bot_core.data.ohlcv.sqlite_storage import SQLiteCacheStorage


@pytest.fixture()
def sample_universe() -> InstrumentUniverseConfig:
    return InstrumentUniverseConfig(
        name="test",
        description="",
        instruments=(
            InstrumentConfig(
                name="BTC/USDT",
                base_asset="BTC",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"binance_spot": "BTCUSDT"},
                backfill_windows=(
                    InstrumentBackfillWindow(interval="1d", lookback_days=365),
                    InstrumentBackfillWindow(interval="1h", lookback_days=30),
                ),
            ),
            InstrumentConfig(
                name="ETH/USDT",
                base_asset="ETH",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"binance_spot": "ETHUSDT"},
                backfill_windows=(
                    InstrumentBackfillWindow(interval="1d", lookback_days=365),
                ),
            ),
        ),
    )


def test_report_marks_missing_metadata(tmp_path: Path, sample_universe: InstrumentUniverseConfig) -> None:
    manifest = tmp_path / "manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    metadata = storage.metadata()
    metadata["last_timestamp::BTCUSDT::1d"] = str(int(datetime(2024, 5, 1, tzinfo=timezone.utc).timestamp() * 1000))
    metadata["row_count::BTCUSDT::1d"] = "200"

    entries = generate_manifest_report(
        manifest_path=manifest,
        universe=sample_universe,
        exchange_name="binance_spot",
        as_of=datetime(2024, 5, 10, tzinfo=timezone.utc),
        warning_thresholds={"1d": 1440},
    )

    btc_daily = next(e for e in entries if e.symbol == "BTCUSDT" and e.interval == "1d")
    assert btc_daily.status == "warning"
    assert pytest.approx(btc_daily.gap_minutes or 0.0, rel=1e-3) == 12960.0  # 9 dni

    btc_hourly = next(e for e in entries if e.symbol == "BTCUSDT" and e.interval == "1h")
    assert btc_hourly.status == "missing_metadata"
    assert btc_hourly.last_timestamp_iso is None

    eth_daily = next(e for e in entries if e.symbol == "ETHUSDT" and e.interval == "1d")
    assert eth_daily.status == "missing_metadata"


def test_report_handles_invalid_timestamp(tmp_path: Path, sample_universe: InstrumentUniverseConfig) -> None:
    manifest = tmp_path / "manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    metadata = storage.metadata()
    metadata["last_timestamp::BTCUSDT::1d"] = "not-a-number"
    metadata["row_count::BTCUSDT::1d"] = "50"

    entries = generate_manifest_report(
        manifest_path=manifest,
        universe=sample_universe,
        exchange_name="binance_spot",
    )

    entry = next(e for e in entries if e.symbol == "BTCUSDT" and e.interval == "1d")
    assert entry.status == "invalid_metadata"
    assert entry.last_timestamp_iso == "not-a-number"


def test_summarize_status_counts_entries(sample_universe: InstrumentUniverseConfig, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    metadata = storage.metadata()
    now_ms = int(datetime(2024, 5, 10, tzinfo=timezone.utc).timestamp() * 1000)
    metadata["last_timestamp::BTCUSDT::1d"] = str(now_ms)
    metadata["row_count::BTCUSDT::1d"] = "10"

    entries = generate_manifest_report(
        manifest_path=manifest,
        universe=sample_universe,
        exchange_name="binance_spot",
        as_of=datetime(2024, 5, 10, tzinfo=timezone.utc),
    )

    summary = summarize_status(entries)
    assert summary["ok"] == 1
