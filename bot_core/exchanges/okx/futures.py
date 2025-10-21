"""Adapter futures OKX z obsługą watchdog i metryk."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.health import Watchdog


class OKXFuturesAdapter(WatchdogCCXTAdapter):
    """Adapter kontraktów futures/swap dla OKX."""

    name = "okx_futures"

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
                    "marginMode": "cross",
                },
            },
            "fetch_ohlcv_params": {"instType": "SWAP"},
            "create_order_params": {"instType": "SWAP"},
            "cancel_order_params": {"instType": "SWAP"},
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        combined_settings.setdefault(
            "sandbox_mode", environment in (Environment.PAPER, Environment.TESTNET)
        )
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "okx"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
            watchdog=watchdog,
        )


__all__ = ["OKXFuturesAdapter"]
