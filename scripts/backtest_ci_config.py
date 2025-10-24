"""Configuration for CI backtest regression harness."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

BACKTEST_CI_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "trend_following",
        "dataset": REPO_ROOT / "data" / "sample_ohlcv" / "trend.csv",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "initial_balance": 10_000.0,
        "risk_profile": "balanced",
        "required_data": ["open", "high", "low", "close", "volume"],
        "matching": {
            "latency_bars": 1,
            "slippage_bps": 5.0,
            "fee_bps": 10.0,
            "liquidity_share": 1.0,
        },
        "context_extra": {
            "trade_risk_pct": 0.01,
            "max_position_notional_pct": 0.035,
            "max_leverage": 1.0,
        },
        "strategy": {
            "fast_window": 8,
            "slow_window": 21,
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.03,
        },
    }
]

__all__ = ["BACKTEST_CI_SCENARIOS", "REPO_ROOT"]
