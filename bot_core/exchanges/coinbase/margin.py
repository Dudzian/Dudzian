"""Adapter margin Coinbase Advanced Trade z watchdogiem."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule
from bot_core.exchanges.health import Watchdog


class CoinbaseMarginAdapter(WatchdogCCXTAdapter):
    """Adapter Coinbase margin wykorzystujący CCXT."""

    name = "coinbase_margin"

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
                "timeout": 12_000,
                "options": {
                    "defaultType": "margin",
                },
            },
            "fetch_ohlcv_params": {"product_type": "margin"},
            "create_order_params": {"product_type": "margin"},
            "cancel_order_params": {"product_type": "margin"},
            "rate_limit_rules": (
                RateLimitRule(rate=10, per=1.0),
                RateLimitRule(rate=120, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 4,
                "base_delay": 0.3,
                "max_delay": 2.0,
                "jitter": (0.05, 0.3),
            },
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        combined_settings.setdefault(
            "sandbox_mode", environment in (Environment.PAPER, Environment.TESTNET)
        )
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "coinbase"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
            watchdog=watchdog,
        )


__all__ = ["CoinbaseMarginAdapter"]
