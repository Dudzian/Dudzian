from pathlib import Path

import pytest

from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig
from bot_core.market_intel import MarketIntelAggregator
from bot_core.market_intel.sqlite_builder import (
    MarketIntelSqliteBuilder,
    OrderBookLevel,
    OrderBookSnapshot,
    FundingSnapshot,
    SentimentSnapshot,
    OHLCVBar,
)


class _StubProvider:
    def __init__(self) -> None:
        self._base = {
            "BTCUSDT": 20_000.0,
            "ETHUSDT": 1_500.0,
        }

    def fetch_order_book(self, symbol: str, *, depth: int) -> OrderBookSnapshot:
        base = self._base[symbol]
        bids = [
            OrderBookLevel(price=base - 10.0, quantity=1.25),
            OrderBookLevel(price=base - 20.0, quantity=1.35),
        ]
        asks = [
            OrderBookLevel(price=base + 10.0, quantity=1.15),
            OrderBookLevel(price=base + 20.0, quantity=1.45),
        ]
        return OrderBookSnapshot(bids=tuple(bids[:depth]), asks=tuple(asks[:depth]))

    def fetch_funding(self, symbol: str) -> FundingSnapshot:
        return FundingSnapshot(rate_bps=12.5 if symbol == "BTCUSDT" else 8.0)

    def fetch_sentiment(self, symbol: str) -> SentimentSnapshot:
        return SentimentSnapshot(score=0.35 if symbol == "BTCUSDT" else 0.15)

    def fetch_ohlcv(self, symbol: str, *, bars: int) -> tuple[OHLCVBar, ...]:
        base = self._base[symbol]
        closes = [base * (1.0 + 0.0005 * i) for i in range(bars)]
        return tuple(OHLCVBar(close=value) for value in closes)

    def resolve_weight(self, symbol: str) -> float:
        return 1.0 if symbol == "BTCUSDT" else 0.8


@pytest.fixture()
def sqlite_config(tmp_path: Path) -> MarketIntelConfig:
    db_path = tmp_path / "market_metrics.sqlite"
    sqlite_cfg = MarketIntelSqliteConfig(path=str(db_path))
    return MarketIntelConfig(
        enabled=True,
        output_directory=str(tmp_path / "metrics"),
        manifest_path=str(tmp_path / "manifest.json"),
        sqlite=sqlite_cfg,
        required_symbols=("BTCUSDT", "ETHUSDT"),
        default_weight=1.25,
    )


def test_builder_populates_sqlite_and_aggregator_reads(sqlite_config: MarketIntelConfig) -> None:
    provider = _StubProvider()
    builder = MarketIntelSqliteBuilder(
        sqlite_config,
        provider=provider,
        depth_levels=2,
        volatility_lookback=32,
    )

    baselines = builder.collect()
    assert {baseline.symbol for baseline in baselines} == {"BTCUSDT", "ETHUSDT"}

    btc_baseline = next(item for item in baselines if item.symbol == "BTCUSDT")
    assert btc_baseline.mid_price == pytest.approx(20_000.0)
    assert btc_baseline.avg_spread_bps > 0.0
    assert btc_baseline.avg_depth_usd > 0.0
    assert btc_baseline.realized_volatility > 0.0
    assert btc_baseline.weight == pytest.approx(1.0)

    db_path = builder.write_database(baselines)
    assert db_path.exists()
    builder.validate_checksums(baselines)

    aggregator = MarketIntelAggregator(sqlite_config)
    rows = aggregator.build()
    assert {row.symbol for row in rows} == {"BTCUSDT", "ETHUSDT"}
    btc_row = next(item for item in rows if item.symbol == "BTCUSDT")
    assert btc_row.mid_price == pytest.approx(btc_baseline.mid_price)


def test_builder_rejects_missing_orderbook(sqlite_config: MarketIntelConfig) -> None:
    class _FailingProvider(_StubProvider):
        def fetch_order_book(self, symbol: str, *, depth: int) -> OrderBookSnapshot:  # type: ignore[override]
            return OrderBookSnapshot(bids=(), asks=())

    builder = MarketIntelSqliteBuilder(
        sqlite_config,
        provider=_FailingProvider(),
        depth_levels=1,
        volatility_lookback=8,
    )

    with pytest.raises(ValueError):
        builder.collect()
