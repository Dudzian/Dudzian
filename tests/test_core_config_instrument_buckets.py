from pathlib import Path

from bot_core.config.loader import load_core_config


def test_load_core_config_reads_instrument_buckets(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          conservative:
            max_daily_loss_pct: 0.01
            max_position_pct: 0.03
            target_volatility: 0.07
            max_leverage: 2.0
            stop_loss_atr_multiple: 1.0
            max_open_positions: 2
            hard_drawdown_pct: 0.05
            instrument_buckets: [spot_core]
        instrument_universes:
          spot_universe:
            description: Core spot pairs
            instruments:
              BTC_USDT:
                base_asset: BTC
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: BTCUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 30
              ETH_USDT:
                base_asset: ETH
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: ETHUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 30
        instrument_buckets:
          spot_core:
            universe: spot_universe
            symbols: [BTC_USDT, ETH_USDT]
            max_position_pct: 0.02
            max_notional_usd: 50000
            tags: [core, major]
        environments:
          binance_paper:
            exchange: binance_spot
            environment: paper
            keychain_key: paper
            data_cache_path: ./cache
            risk_profile: conservative
            alert_channels: []
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert "spot_core" in config.instrument_buckets
    bucket = config.instrument_buckets["spot_core"]
    assert bucket.universe == "spot_universe"
    assert bucket.symbols == ("BTC_USDT", "ETH_USDT")
    assert abs(bucket.max_position_pct - 0.02) < 1e-9
    assert abs(bucket.max_notional_usd - 50000.0) < 1e-9
    assert set(bucket.tags) == {"core", "major"}

    profile = config.risk_profiles["conservative"]
    assert profile.instrument_buckets == ("spot_core",)
