"""Agregator danych Market Intelligence (Stage6) — warstwa zgodności nazw."""

from bot_core.market_intel import aggregator as _agg

_optional_exports = []

# Zawsze dostępne
MarketIntelAggregator = _agg.MarketIntelAggregator

# Opcjonalne nazwy różniące się między gałęziami

if hasattr(_agg, "MarketIntelQuery"):
    MarketIntelQuery = _agg.MarketIntelQuery  # type: ignore[attr-defined]
    _optional_exports.append("MarketIntelQuery")

if hasattr(_agg, "MarketIntelSnapshot"):
    MarketIntelSnapshot = _agg.MarketIntelSnapshot  # type: ignore[attr-defined]
    _optional_exports.append("MarketIntelSnapshot")

if hasattr(_agg, "MarketIntelBaseline"):
    MarketIntelBaseline = _agg.MarketIntelBaseline  # type: ignore[attr-defined]
    _optional_exports.append("MarketIntelBaseline")

if hasattr(_agg, "MarketIntelSourceInfo"):
    MarketIntelSourceInfo = _agg.MarketIntelSourceInfo  # type: ignore[attr-defined]
    _optional_exports.append("MarketIntelSourceInfo")

try:  # nowe API Stage6 – builder SQLite
    from bot_core.market_intel.sqlite_builder import (  # type: ignore[attr-defined]
        MarketIntelSqliteBuilder,
        MarketIntelDataProvider,
        OrderBookLevel,
        OrderBookSnapshot,
        FundingSnapshot,
        SentimentSnapshot,
        OHLCVBar,
    )

    _optional_exports.extend(
        [
            "MarketIntelSqliteBuilder",
            "MarketIntelDataProvider",
            "OrderBookLevel",
            "OrderBookSnapshot",
            "FundingSnapshot",
            "SentimentSnapshot",
            "OHLCVBar",
        ]
    )
except Exception:  # pragma: no cover - gałęzie bez buildera
    pass

__all__ = ["MarketIntelAggregator", *_optional_exports]
