"""Adapter futures Bybit zintegrowany z CCXT oraz watchdogiem."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule
from bot_core.exchanges.health import Watchdog


class BybitFuturesAdapter(WatchdogCCXTAdapter):
    """Adapter Bybit futures obsługujący kontrakty perpetual."""

    name = "bybit_futures"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment,
        settings: Mapping[str, Any] | None = None,
        client=None,
        metrics_registry=None,
        watchdog: Watchdog | None = None,
    ) -> None:
        defaults: dict[str, Any] = {
            "ccxt_config": {
                "timeout": 20_000,
                "options": {
                    "defaultType": "swap",
                    "hedgeMode": True,
                },
            },
            "fetch_ohlcv_params": {"category": "linear"},
            "create_order_params": {"category": "linear"},
            "cancel_order_params": {"category": "linear"},
            "rate_limit_rules": (
                RateLimitRule(rate=20, per=1.0),
                RateLimitRule(rate=240, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 5,
                "base_delay": 0.2,
                "max_delay": 2.5,
                "jitter": (0.05, 0.35),
            },
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        combined_settings.setdefault(
            "sandbox_mode", environment in (Environment.PAPER, Environment.TESTNET)
        )
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "bybit"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
            watchdog=watchdog,
        )


__all__ = ["BybitFuturesAdapter"]
