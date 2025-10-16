"""Agregator danych Market Intelligence (Stage6) — warstwa zgodności nazw."""

from bot_core.market_intel import aggregator as _agg

# Zawsze dostępne
MarketIntelAggregator = _agg.MarketIntelAggregator

# Opcjonalne nazwy różniące się między gałęziami
_optional_exports = []

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

__all__ = ["MarketIntelAggregator", *_optional_exports]
