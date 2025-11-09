"""Zestaw danych demonstracyjnych decision logu na potrzeby pierwszego uruchomienia."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping

DecisionRecord = Mapping[str, str]

_DEMO_TEMPLATE: tuple[DecisionRecord, ...] = (
    {
        "event": "order_submitted",
        "timestamp": datetime(2025, 1, 7, 12, 0, tzinfo=timezone.utc).isoformat(),
        "environment": "demo",
        "portfolio": "alpha",
        "risk_profile": "dynamic",
        "strategy": "momentum_v2",
        "schedule": "auto",
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "submitted",
        "quantity": "0.15",
        "price": "43650.25",
        "decision_state": "trade",
        "decision_should_trade": "true",
        "decision_confidence": "0.84",
        "ai_model": "omega-transformer",
        "ai_version": "2.3.1",
        "market_regime_state": "bull",
        "market_regime_risk_level": "moderate",
        "market_regime_volatility": "0.41",
        "latency_ms": "48",
    },
    {
        "event": "order_filled",
        "timestamp": datetime(2025, 1, 7, 12, 0, 2, tzinfo=timezone.utc).isoformat(),
        "environment": "demo",
        "portfolio": "alpha",
        "risk_profile": "dynamic",
        "strategy": "momentum_v2",
        "schedule": "auto",
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "filled",
        "quantity": "0.15",
        "price": "43652.10",
        "decision_state": "trade",
        "decision_should_trade": "true",
        "decision_signal_strength": "0.72",
        "ai_model": "omega-transformer",
        "ai_version": "2.3.1",
        "market_regime_state": "bull",
        "market_regime_risk_level": "moderate",
        "market_regime_volatility": "0.39",
        "latency_ms": "41",
    },
    {
        "event": "order_closed",
        "timestamp": datetime(2025, 1, 7, 16, 45, tzinfo=timezone.utc).isoformat(),
        "environment": "demo",
        "portfolio": "alpha",
        "risk_profile": "dynamic",
        "strategy": "momentum_v2",
        "schedule": "auto",
        "symbol": "BTC/USDT",
        "side": "sell",
        "status": "filled",
        "quantity": "0.15",
        "price": "45210.75",
        "decision_state": "take_profit",
        "decision_should_trade": "true",
        "decision_signal_strength": "0.67",
        "ai_model": "omega-transformer",
        "ai_version": "2.3.1",
        "market_regime_state": "bull",
        "market_regime_risk_level": "moderate",
        "market_regime_volatility": "0.36",
        "latency_ms": "37",
    },
    {
        "event": "risk_update",
        "timestamp": datetime(2025, 1, 8, 7, 15, tzinfo=timezone.utc).isoformat(),
        "environment": "demo",
        "portfolio": "alpha",
        "risk_profile": "dynamic",
        "strategy": "momentum_v2",
        "schedule": "auto",
        "status": "ok",
        "decision_state": "rebalance",
        "decision_should_trade": "false",
        "decision_risk_delta": "-0.12",
        "ai_model": "omega-transformer",
        "ai_version": "2.3.1",
        "market_regime_state": "transition",
        "market_regime_risk_level": "cautious",
        "market_regime_volatility": "0.58",
        "latency_ms": "52",
    },
)


def load_demo_decisions() -> Iterable[DecisionRecord]:
    """Zwraca defensywną kopię wpisów demonstracyjnych."""

    for record in _DEMO_TEMPLATE:
        yield dict(record)


__all__ = ["load_demo_decisions"]

