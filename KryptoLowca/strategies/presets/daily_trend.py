"""Wbudowany preset dziennego trend following dla kont demo."""
from __future__ import annotations

from datetime import datetime, timezone

from KryptoLowca.strategies.marketplace import StrategyPreset

DAILY_TREND_PRESET = StrategyPreset(
    preset_id="DAILY_TREND",
    name="Daily Trend Guard",
    description=(
        "Konserwatywna strategia podążania za trendem dziennym. Wymusza demo "
        "mode oraz ogranicza rozmiar pozycji do 1.5% kapitału, co jest "
        "zgodne z dobrymi praktykami dla konta o wartości 5 000 USD."
    ),
    risk_level="balanced",
    recommended_min_balance=5_000.0,
    timeframe="1d",
    exchanges=["binance", "kraken", "zonda"],
    tags=["trend", "spot", "daily"],
    config={
        "strategy": {
            "preset": "DAILY_TREND",
            "mode": "demo",
            "max_leverage": 1.0,
            "max_position_notional_pct": 0.015,
            "trade_risk_pct": 0.0075,
            "default_sl": 0.02,
            "default_tp": 0.045,
            "violation_cooldown_s": 600,
            "reduce_only_after_violation": True,
        },
        "trade": {
            "risk_per_trade": 0.0075,
            "max_leverage": 1.0,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "max_open_positions": 3,
        },
        "exchange": {
            "exchange_name": "binance",
            "testnet": True,
            "require_demo_mode": True,
            "rate_limit_per_minute": 1200,
            "rate_limit_alert_threshold": 0.75,
        },
    },
    version="1.1.0",
    last_updated=datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc),
    compatibility={"app": ">=2.8.0", "schema": "1.0"},
    compliance={
        "required_flags": [
            "compliance_confirmed",
            "acknowledged_risk_disclaimer",
        ],
        "allow_live": False,
    },
)

__all__ = ["DAILY_TREND_PRESET"]
