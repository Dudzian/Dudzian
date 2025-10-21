"""Adapter futures Coinbase (Advanced Trade) z integracją watchdog."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.health import Watchdog


class CoinbaseFuturesAdapter(WatchdogCCXTAdapter):
    """Adapter Coinbase futures oparty na CCXT."""

    name = "coinbase_futures"

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
                    "defaultType": "swap",
                },
            },
            "fetch_ohlcv_params": {"product_type": "futures"},
            "create_order_params": {"product_type": "futures"},
            "cancel_order_params": {"product_type": "futures"},
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


__all__ = ["CoinbaseFuturesAdapter"]
