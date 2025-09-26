from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from KryptoLowca.config_manager import ConfigManager, StrategyConfig, ValidationError
from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest
from KryptoLowca.backtest.mini_backtester import MiniBacktester


class DummyExchange:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._ticker_price = 101.0

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 10, since: int | None = None) -> List[List[float]]:
        self.calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "since": since,
        })
        base_ts = 1_600_000_000_000
        candles: List[List[float]] = []
        price = 100.0
        for i in range(limit):
            # Celowo pomijamy trzecią świecę, aby sprawdzić wypełnianie luk
            if i == 2:
                base_ts += 2 * 60_000
            ts = base_ts + i * 60_000
            candles.append([ts, price, price * 1.001, price * 0.999, price * 1.0005, 5.0 + i])
            price *= 1.0005
        return candles

    def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        return {"last": self._ticker_price}


@pytest.fixture()
def provider() -> MarketDataProvider:
    return MarketDataProvider(DummyExchange(), cache_ttl_s=60.0)


def test_market_data_provider_caches_and_fills_gaps(provider: MarketDataProvider) -> None:
    request = MarketDataRequest(symbol="BTC/USDT", timeframe="1m", limit=5)
    df_first = provider.get_historical(request)
    assert len(df_first) == 5
    assert df_first.index.is_monotonic_increasing
    # drugi raz – powinien użyć cache (bez dodatkowego wywołania fetch_ohlcv)
    exchange = provider._exchange  # type: ignore[attr-defined]
    calls_before = len(exchange.calls)  # type: ignore[attr-defined]
    df_second = provider.get_historical(request)
    calls_after = len(exchange.calls)  # type: ignore[attr-defined]
    assert calls_before == calls_after
    pd.testing.assert_frame_equal(df_first, df_second)
    # walidacja latest price – fallback do fetch_ticker
    assert provider.get_latest_price("BTC/USDT") == pytest.approx(101.0)


def _build_signal_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC")
    prices = [100, 101, 102, 103, 104, 105]
    signals = [1, 1, 1, 0, 1, 0]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [10.0] * len(prices),
            "signal": signals,
        },
        index=idx,
    )
    return df


def test_mini_backtester_applies_limits_and_costs() -> None:
    strategy = StrategyConfig(
        preset="SAFE",
        max_position_notional_pct=0.05,
        trade_risk_pct=0.05,
        default_sl=0.001,
        default_tp=0.01,
        max_leverage=1.0,
    ).validate()
    backtester = MiniBacktester(strategy, fee_rate=0.001, slippage_rate=0.001)
    df = _build_signal_frame()
    report = backtester.run(df, signal_column="signal", price_column="close", initial_balance=1_000.0)
    assert report.trades, "Backtester nie otworzył żadnej pozycji"
    assert report.final_balance > 0
    assert report.fees_paid > 0
    assert report.slippage_paid > 0
    assert report.reduce_only_triggers >= 1  # ograniczenie pozycji powoduje tryb reduce-only
    assert any("limit" in reason for reason in report.violations)


@pytest.mark.asyncio()
async def test_config_manager_preflight_backtest(tmp_path: Path) -> None:
    cfg = ConfigManager(tmp_path / "config.yml")
    strategy = StrategyConfig(
        preset="SAFE",
        mode="live",
        max_leverage=1.0,
        max_position_notional_pct=0.05,
        trade_risk_pct=0.02,
        default_sl=0.01,
        default_tp=0.02,
        violation_cooldown_s=60,
        reduce_only_after_violation=True,
        compliance_confirmed=True,
        api_keys_configured=True,
        acknowledged_risk_disclaimer=True,
    ).validate()
    cfg._current_config["strategy"] = asdict(strategy)

    df = _build_signal_frame()
    report = cfg.run_backtest_on_dataframe(df, signal_column="signal")
    assert report.total_return_pct >= 0

    provider = MarketDataProvider(DummyExchange(), cache_ttl_s=1.0)
    # wstrzykujemy kolumnę sygnału do danych z provider'a
    original_get = provider.get_historical

    def _patched(request: MarketDataRequest) -> pd.DataFrame:
        data = original_get(request)
        signals = [1] * len(data)
        if signals:
            signals[-1] = 0
        data["signal"] = signals
        prices = [100.0 + 2.0 * idx for idx in range(len(data))]
        data["close"] = prices
        data["open"] = prices
        data["high"] = [p * 1.001 for p in prices]
        data["low"] = [p * 0.999 for p in prices]
        return data

    provider.get_historical = _patched  # type: ignore[assignment]
    request = MarketDataRequest(symbol="BTC/USDT", timeframe="1m", limit=5)
    report_provider = await cfg.preflight_backtest(provider, request, signal_column="signal")
    assert report_provider.trades

    # symulujemy negatywny wynik – powinien wywołać ValidationError w trybie LIVE
    losing_df = df.copy()
    losing_df["close"] = [100, 99, 98, 97, 96, 95]
    with pytest.raises(ValidationError):
        cfg.run_backtest_on_dataframe(losing_df, signal_column="signal")
