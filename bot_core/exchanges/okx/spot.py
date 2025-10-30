"""Adapter CCXT dla rynku spot OKX."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule


class OKXSpotAdapter(CCXTSpotAdapter):
    """Adapter OKX wykorzystujący CCXT oraz snapshoty REST."""

    name = "okx_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, Any] | None = None,
        client=None,
        metrics_registry=None,
    ) -> None:
        combined_settings = merge_adapter_settings(
            {
                "ccxt_config": {"timeout": 20_000},
                "fetch_ohlcv_params": {"price": "mark"},
                "rate_limit_rules": (
                    RateLimitRule(rate=30, per=1.0),
                    RateLimitRule(rate=300, per=60.0),
                ),
                "retry_policy": {
                    "max_attempts": 5,
                    "base_delay": 0.2,
                    "max_delay": 2.0,
                    "jitter": (0.05, 0.3),
                },
            },
            settings or {},
        )
        combined_settings.setdefault("sandbox_mode", environment in (Environment.PAPER, Environment.TESTNET))
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "okx"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
        )


__all__ = ["OKXSpotAdapter"]

