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

from KryptoLowca.backtest.reporting import export_report, render_html_report
from KryptoLowca.backtest.simulation import BacktestEngine, MatchingConfig
from KryptoLowca.core.services.paper_adapter import PaperTradingAdapter


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


def _build_flat_dataframe(periods: int = 30, freq: str = "1min", price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-02-01", periods=periods, freq=freq, tz="UTC")
    df = pd.DataFrame(index=idx)
    df["open"] = price
    df["high"] = price
    df["low"] = price
    df["close"] = price
    df["volume"] = 10.0
    return df


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


@registry.register
class TestShortStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="TestShortStrategy", description="Alternating long/short")
    __test__ = False

    async def prepare(self, context: StrategyContext, data_provider: Any) -> None:  # type: ignore[override]
        await super().prepare(context, data_provider)
        self._step = 0

    async def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        history: pd.DataFrame = market_payload["ohlcv"]
        if len(history) < 8:
            return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.0)
        self._step += 1
        last_close = history["close"].iloc[-1]
        prev_close = history["close"].iloc[-2]
        direction_down = last_close < prev_close
        if context.position == 0:
            action = "SELL" if direction_down else "BUY"
            return StrategySignal(symbol=context.symbol, action=action, confidence=0.9)
        if context.position > 0 and direction_down:
            return StrategySignal(symbol=context.symbol, action="SELL", confidence=0.9)
        if context.position < 0 and not direction_down:
            return StrategySignal(symbol=context.symbol, action="BUY", confidence=0.9)
        return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.2)


@registry.register
class TestHoldStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="TestHoldStrategy", description="Buy and hold")
    __test__ = False

    async def prepare(self, context: StrategyContext, data_provider: Any) -> None:  # type: ignore[override]
        await super().prepare(context, data_provider)
        self._entered = False

    async def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        history: pd.DataFrame = market_payload["ohlcv"]
        if len(history) < 5:
            return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.0)
        if not self._entered:
            self._entered = True
            return StrategySignal(symbol=context.symbol, action="BUY", confidence=0.95)
        return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.1)


@registry.register
class ZeroVolumeStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="ZeroVolumeStrategy", description="Handles zero liquidity bars")
    __test__ = False

    async def prepare(self, context: StrategyContext, data_provider: Any) -> None:  # type: ignore[override]
        await super().prepare(context, data_provider)
        self._entered = False

    async def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        history: pd.DataFrame = market_payload["ohlcv"]
        if len(history) < 6:
            return StrategySignal(symbol=context.symbol, action="HOLD", confidence=0.0)
        if not self._entered:
            self._entered = True
            return StrategySignal(symbol=context.symbol, action="BUY", confidence=0.9)
        if self._entered and len(history) > 12 and context.position > 0:
            return StrategySignal(symbol=context.symbol, action="SELL", confidence=0.9)
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


def test_backtest_engine_supports_short_positions() -> None:
    df = _build_dataframe(periods=200)
    engine = BacktestEngine(
        strategy_name="TestShortStrategy",
        data=df,
        symbol="BTC/USDT",
        timeframe="1m",
        initial_balance=1_000.0,
        matching=MatchingConfig(latency_bars=1, slippage_bps=2.0, fee_bps=8.0, liquidity_share=1.0),
        allow_short=True,
        context_extra={
            "trade_risk_pct": 0.05,
            "max_position_notional_pct": 0.5,
            "max_leverage": 2.0,
        },
    )
    report = engine.run()
    assert report.metrics is not None
    assert any(trade.direction == "SHORT" for trade in report.trades)
    assert any(fill.side == "sell" for fill in report.fills)


def test_forced_closure_uses_matching_costs() -> None:
    df = _build_dataframe(periods=80)
    engine = BacktestEngine(
        strategy_name="TestHoldStrategy",
        data=df,
        symbol="ETH/USDT",
        timeframe="1m",
        initial_balance=2_000.0,
        matching=MatchingConfig(latency_bars=0, slippage_bps=5.0, fee_bps=15.0, liquidity_share=1.0),
        context_extra={
            "trade_risk_pct": 0.1,
            "max_position_notional_pct": 0.5,
        },
    )
    report = engine.run()
    assert report.trades, "Forced closure should complete trade"
    last_fill = report.fills[-1]
    assert last_fill.fee > 0
    assert abs(last_fill.slippage) > 0


def test_fees_reduce_final_balance() -> None:
    df = _build_flat_dataframe(periods=40, price=100.0)
    engine = BacktestEngine(
        strategy_name="TestHoldStrategy",
        data=df,
        symbol="BTC/USDT",
        timeframe="1m",
        initial_balance=1_000.0,
        matching=MatchingConfig(latency_bars=0, slippage_bps=0.0, fee_bps=10.0, liquidity_share=1.0),
        context_extra={"trade_risk_pct": 0.1, "max_position_notional_pct": 0.5},
    )
    report = engine.run()
    assert report.metrics is not None
    total_fees = sum(fill.fee for fill in report.fills)
    assert total_fees > 0
    assert report.final_balance == pytest.approx(report.starting_balance - total_fees, rel=1e-6)


def test_backtest_handles_zero_volume_bars() -> None:
    df = _build_dataframe(periods=60)
    zero_idx = df.index[20]
    df.loc[zero_idx, "volume"] = 0.0
    engine = BacktestEngine(
        strategy_name="ZeroVolumeStrategy",
        data=df,
        symbol="SOL/USDT",
        timeframe="1m",
        initial_balance=1_500.0,
        matching=MatchingConfig(latency_bars=0, slippage_bps=1.0, fee_bps=5.0, liquidity_share=0.5),
        context_extra={
            "trade_risk_pct": 0.05,
            "max_position_notional_pct": 0.5,
        },
    )
    report = engine.run()
    assert report.trades
    zero_ts = zero_idx.to_pydatetime()
    assert all(fill.timestamp != zero_ts for fill in report.fills)
    assert any("zerowym wolumenem" in warning for warning in report.warnings)


def test_backtest_warns_on_missing_candles() -> None:
    df = _build_dataframe(periods=60)
    df = df.drop(df.index[15])
    engine = BacktestEngine(
        strategy_name="TestTrendStrategy",
        data=df,
        symbol="BTC/USDT",
        timeframe="1m",
        initial_balance=2_000.0,
        matching=MatchingConfig(),
        context_extra={"trade_risk_pct": 0.05, "max_position_notional_pct": 0.5},
    )
    report = engine.run()
    assert any("luki w danych" in warning for warning in report.warnings)


def test_reporting_snapshot_export(tmp_path: Path) -> None:
    df = _build_dataframe(periods=120)
    engine = BacktestEngine(
        strategy_name="TestTrendStrategy",
        data=df,
        symbol="BTC/USDT",
        timeframe="1m",
        initial_balance=5_000.0,
        matching=MatchingConfig(latency_bars=1, slippage_bps=1.0, fee_bps=5.0, liquidity_share=1.0),
        context_extra={
            "trade_risk_pct": 0.02,
            "max_position_notional_pct": 0.5,
        },
    )
    report = engine.run()
    html = render_html_report(report, title="Regression Snapshot")
    assert "Regression Snapshot" in html
    paths = export_report(report, tmp_path, title="Regression Snapshot")
    assert paths["html"].read_text(encoding="utf-8") == html
    if "pdf" in paths:
        assert paths["pdf"].exists()


def test_paper_trading_adapter_multi_symbol_multi_timeframe() -> None:
    adapter = PaperTradingAdapter(initial_balance=10_000.0, matching=MatchingConfig(latency_bars=0, liquidity_share=1.0))
    base_time = pd.Timestamp("2024-02-01", tz="UTC")

    for idx in range(3):
        ts = base_time + pd.Timedelta(minutes=idx)
        btc_bar = {
            "open": 100 + idx,
            "high": 100.5 + idx,
            "low": 99.5 + idx,
            "close": 100.2 + idx,
            "volume": 5.0,
            "timestamp": ts.to_pydatetime(),
            "index": idx,
        }
        eth_bar = {
            "open": 50 + idx,
            "high": 50.5 + idx,
            "low": 49.5 + idx,
            "close": 50.1 + idx,
            "volume": 3.0,
            "timestamp": (ts + pd.Timedelta(minutes=idx)).to_pydatetime(),
            "index": idx,
        }
        adapter.submit_order(symbol="BTC/USDT", side="buy", size=0.1, bar_index=idx)
        adapter.update_market_data("BTC/USDT", "1m", {"ohlcv": btc_bar})
        adapter.submit_order(symbol="ETH/USDT", side="sell", size=0.2, bar_index=idx)
        adapter.update_market_data("ETH/USDT", "5m", {"ohlcv": eth_bar})

    btc_snapshot = adapter.portfolio_snapshot("BTC/USDT")
    eth_snapshot = adapter.portfolio_snapshot("ETH/USDT")
    assert btc_snapshot["value"] != eth_snapshot["value"]
    assert btc_snapshot["position"] > 0
    assert eth_snapshot["position"] < 0
