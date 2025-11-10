"""Adapter CCXT dla rynku spot KuCoin."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule


class KuCoinSpotAdapter(CCXTSpotAdapter):
    """Adapter KuCoin spot oparty na CCXT."""

    name = "kucoin_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, Any] | None = None,
        client=None,
        metrics_registry=None,
    ) -> None:
        defaults: dict[str, Any] = {
            "ccxt_config": {
                "timeout": 15_000,
                "options": {"defaultType": "spot"},
            },
            "fetch_ohlcv_params": {"type": "spot"},
            "cancel_order_params": {"type": "spot"},
            "rate_limit_rules": (
                RateLimitRule(rate=30, per=3.0),
                RateLimitRule(rate=1_800, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 4,
                "base_delay": 0.2,
                "max_delay": 2.0,
                "jitter": (0.1, 0.3),
            },
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        combined_settings.setdefault(
            "sandbox_mode", environment in (Environment.PAPER, Environment.TESTNET)
        )
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "kucoin"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
        )


__all__ = ["KuCoinSpotAdapter"]
