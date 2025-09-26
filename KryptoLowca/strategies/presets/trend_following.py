"""Wbudowany preset intraday trend following dla Binance/Kraken."""
from __future__ import annotations

from datetime import datetime, timezone

from KryptoLowca.strategies.marketplace import StrategyPreset

INTRADAY_TREND_PRESET = StrategyPreset(
    preset_id="INTRADAY_TREND",
    name="Intraday Momentum Scout",
    description=(
        "Agresywniejsza wersja trend following na interwale 1h. Została "
        "przygotowana do pracy z kontem demo Binance Testnet i Kraken Demo, "
        "z dodatkowymi limitami bezpieczeństwa (cooldown po naruszeniu ryzyka)."
    ),
    risk_level="assertive",
    recommended_min_balance=10_000.0,
    timeframe="1h",
    exchanges=["binance", "kraken"],
    tags=["trend", "momentum", "intraday"],
    config={
        "strategy": {
            "preset": "INTRADAY_TREND",
            "mode": "demo",
            "max_leverage": 1.0,
            "max_position_notional_pct": 0.025,
            "trade_risk_pct": 0.0125,
            "default_sl": 0.018,
            "default_tp": 0.055,
            "violation_cooldown_s": 900,
            "reduce_only_after_violation": True,
        },
        "trade": {
            "risk_per_trade": 0.0125,
            "max_leverage": 1.0,
            "stop_loss_pct": 0.018,
            "take_profit_pct": 0.055,
            "max_open_positions": 4,
        },
        "exchange": {
            "exchange_name": "kraken",
            "testnet": True,
            "require_demo_mode": True,
            "rate_limit_per_minute": 900,
            "rate_limit_alert_threshold": 0.8,
        },
    },
    version="2.0.0",
    last_updated=datetime(2024, 5, 15, 9, 30, tzinfo=timezone.utc),
    compatibility={"app": ">=2.9.0", "schema": "1.1"},
    compliance={
        "required_flags": [
            "compliance_confirmed",
            "api_keys_configured",
            "acknowledged_risk_disclaimer",
        ],
        "allow_live": False,
    },
)

__all__ = ["INTRADAY_TREND_PRESET"]
