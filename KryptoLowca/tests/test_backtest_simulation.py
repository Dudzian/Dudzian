from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import pytest

from KryptoLowca.config_manager import ConfigManager, StrategyConfig, ValidationError
from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest
from KryptoLowca.strategies.base import BaseStrategy, StrategyContext, StrategyMetadata, StrategySignal, registry

from KryptoLowca.backtest.simulation import BacktestEngine, MatchingConfig


class DummyExchange:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._ticker_price = 101.0

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 10,
        since: int | None = None,
    ) -> List[List[float]]:
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
            ts = base_ts + i * 60_000
            candles.append([ts, price, price * 1.001, price * 0.999, price * 1.0005, 5.0 + i])
            price *= 1.0005
        return candles

    def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        return {"last": self._ticker_price}


@pytest.fixture()
def provider() -> MarketDataProvider:
    return MarketDataProvider(DummyExchange(), cache_ttl_s=60.0)


def test_market_data_provider_caches(provider: MarketDataProvider) -> None:
    request = MarketDataRequest(symbol="BTC/USDT", timeframe="1m", limit=5)
    df_first = provider.get_historical(request)
    assert len(df_first) == 5
    exchange = provider._exchange  # type: ignore[attr-defined]
    calls_before = len(exchange.calls)  # type: ignore[attr-defined]
    df_second = provider.get_historical(request)
    calls_after = len(exchange.calls)  # type: ignore[attr-defined]
    assert calls_before == calls_after
    pd.testing.assert_frame_equal(df_first, df_second)
    assert provider.get_latest_price("BTC/USDT") == pytest.approx(101.0)


def _build_dataframe(periods: int = 120, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    base = pd.DataFrame(index=idx)
    prices: List[float] = []
    for i in range(periods):
        if i < periods // 2:
            prices.append(100 + i * 0.2)
        else:
            prices.append(100 + (periods - i) * 0.2)
    base["open"] = prices
    base["high"] = [p * 1.001 for p in prices]
    base["low"] = [p * 0.999 for p in prices]
    base["close"] = [p * 1.0005 for p in prices]
    base["volume"] = [5.0 + i * 0.01 for i in range(periods)]
    return base


@registry.register
class TestTrendStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="TestTrendStrategy", description="Synthetic test strategy")
    __test__ = False

    async def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        history: pd.DataFrame = market_payload["ohlcv"]
        if len(history) < 5:
            return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.0)
        closes = history["close"].tail(5)
        trend = closes.diff().mean()
        if trend > 0 and context.position <= 0:
            return StrategySignal(symbol=context.symbol, action="BUY", confidence=0.8)
        if trend < 0 and context.position > 0:
            return StrategySignal(symbol=context.symbol, action="SELL", confidence=0.8)
        return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.1)


def test_backtest_engine_generates_metrics() -> None:
    df = _build_dataframe()
    engine = BacktestEngine(
        strategy_name="TestTrendStrategy",
        data=df,
        symbol="BTC/USDT",
        timeframe="1m",
        initial_balance=1_000.0,
        matching=MatchingConfig(latency_bars=1, slippage_bps=1.0, fee_bps=5.0, liquidity_share=1.0),
        context_extra={
            "trade_risk_pct": 0.1,
            "max_position_notional_pct": 0.5,
            "max_leverage": 2.0,
        },
    )
    report = engine.run()
    assert report.trades
    assert report.metrics is not None
    assert report.metrics.total_return_pct >= -100.0
    assert report.metrics.max_drawdown_pct >= 0.0


@pytest.mark.asyncio()
async def test_config_manager_preflight_backtest(tmp_path: Path, provider: MarketDataProvider) -> None:
    cfg = ConfigManager(tmp_path / "config.yml")
    strategy = StrategyConfig(
        preset="TestTrendStrategy",
        mode="demo",
        max_leverage=2.0,
        max_position_notional_pct=0.5,
        trade_risk_pct=0.1,
        default_sl=0.01,
        default_tp=0.02,
    ).validate()
    cfg._current_config["strategy"] = asdict(strategy)

    df = _build_dataframe()
    report = cfg.run_backtest_on_dataframe(
        df,
        symbol="BTC/USDT",
        timeframe="1m",
        strategy_name="TestTrendStrategy",
        initial_balance=1_000.0,
    )
    assert report.metrics is not None

    original_get = provider.get_historical

    def _patched(request: MarketDataRequest) -> pd.DataFrame:
        return _build_dataframe(periods=request.limit or 200)

    provider.get_historical = _patched  # type: ignore[assignment]
    request = MarketDataRequest(symbol="BTC/USDT", timeframe="1m", limit=50)
    report_provider = await cfg.preflight_backtest(
        provider,
        request,
        strategy_name="TestTrendStrategy",
        initial_balance=500.0,
    )
    assert report_provider.metrics is not None

    losing_df = df.copy()
    losing_df["close"] = [90 - i for i in range(len(losing_df))]
    with pytest.raises(ValidationError):
        cfg.run_backtest_on_dataframe(
            losing_df,
            symbol="BTC/USDT",
            timeframe="1m",
            strategy_name="TestTrendStrategy",
            initial_balance=1_000.0,
        )


def test_backtest_benchmark() -> None:
    df = _build_dataframe(periods=10_000)
    engine = BacktestEngine(
        strategy_name="TestTrendStrategy",
        data=df,
        symbol="ETH/USDT",
        timeframe="1m",
        initial_balance=10_000.0,
        matching=MatchingConfig(),
        context_extra={
            "trade_risk_pct": 0.02,
            "max_position_notional_pct": 0.2,
            "max_leverage": 1.5,
        },
    )
    start = time.perf_counter()
    report = engine.run()
    duration = time.perf_counter() - start
    assert report.metrics is not None
    assert duration < 5.0
