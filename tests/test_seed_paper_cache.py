from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.data.ohlcv import ParquetCacheStorage, SQLiteCacheStorage

from scripts.seed_paper_cache import GeneratedSeries, generate_smoke_cache


def _write_config(path: Path, data_path: Path, *, cache_namespace: str | None = None) -> None:
    data_source_block = ""
    if cache_namespace:
        data_source_block = (
            "    data_source:\n"
            f"      cache_namespace: {cache_namespace}\n"
        )

    path.write_text(
        f"""
risk_profiles:
  balanced:
    max_daily_loss_pct: 0.01
    max_position_pct: 0.05
    target_volatility: 0.1
    max_leverage: 2.0
    stop_loss_atr_multiple: 1.5
    max_open_positions: 5
    hard_drawdown_pct: 0.1

instrument_universes:
  test_universe:
    description: smoke
    instruments:
      AAA_USDT:
        base_asset: AAA
        quote_asset: USDT
        categories: []
        exchanges:
          binance_spot: AAAUSDT
        backfill:
          - interval: "1d"
            lookback_days: 30
      BBB_USDT:
        base_asset: BBB
        quote_asset: USDT
        categories: []
        exchanges:
          binance_spot: BBBUSDT
        backfill:
          - interval: "1d"
            lookback_days: 30

environments:
  test_env:
    exchange: binance_spot
    environment: paper
    keychain_key: test
    credential_purpose: trading
    data_cache_path: {data_path!s}
{data_source_block}    risk_profile: balanced
    alert_channels: []
    ip_allowlist: []
    required_permissions: [trade]
    forbidden_permissions: []
    instrument_universe: test_universe
    adapter_settings:
      paper_trading:
        valuation_asset: USDT
        position_size: 0.1
        initial_balances:
          USDT: 1000.0
""",
        encoding="utf-8",
    )


def test_generate_smoke_cache_writes_parquet_and_manifest(tmp_path):
    data_path = tmp_path / "cache"
    config_path = tmp_path / "core.yaml"
    _write_config(config_path, data_path)

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    results = generate_smoke_cache(
        config_path=config_path,
        environment_name="test_env",
        interval="1d",
        days=5,
        start_date=start_date,
        seed=42,
    )

    assert results
    assert all(isinstance(entry, GeneratedSeries) for entry in results)
    symbols = {entry.symbol for entry in results}
    assert symbols == {"AAAUSDT", "BBBUSDT"}

    parquet_storage = ParquetCacheStorage(data_path / "ohlcv_parquet", namespace="binance_spot")
    manifest_storage = SQLiteCacheStorage(data_path / "ohlcv_manifest.sqlite", store_rows=False)

    for entry in results:
        payload = parquet_storage.read(f"{entry.symbol}::{entry.interval}")
        rows = payload["rows"]
        assert len(rows) == entry.candles
        assert rows[0][0] == entry.start_timestamp
        assert rows[-1][0] == entry.end_timestamp
        key = f"row_count::{entry.symbol}::{entry.interval}"
        metadata = manifest_storage.metadata()
        assert metadata[key] == str(entry.candles)
        last_key = f"last_timestamp::{entry.symbol}::{entry.interval}"
        assert metadata[last_key] == str(entry.end_timestamp)


def test_generate_smoke_cache_respects_cache_namespace(tmp_path):
    data_path = tmp_path / "cache"
    config_path = tmp_path / "core_namespace.yaml"
    _write_config(config_path, data_path, cache_namespace="offline_namespace")

    start_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
    results = generate_smoke_cache(
        config_path=config_path,
        environment_name="test_env",
        interval="1d",
        days=3,
        start_date=start_date,
        seed=1,
    )

    assert results

    parquet_storage = ParquetCacheStorage(
        data_path / "ohlcv_parquet",
        namespace="offline_namespace",
    )

    for entry in results:
        payload = parquet_storage.read(f"{entry.symbol}::{entry.interval}")
        assert payload["rows"]
