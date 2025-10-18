"""Adapter CCXT dla Bitfinex Spot."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter


class BitfinexSpotAdapter(CCXTSpotAdapter):
    """Adapter Bitfinex wykorzystujący klienta CCXT."""

    name = "bitfinex_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, Any] | None = None,
        client=None,
        metrics_registry=None,
    ) -> None:
        combined_settings: dict[str, Any] = {
            "ccxt_config": {"timeout": 15_000},
            "fetch_ohlcv_params": {"sort": 1},
        }
        if settings:
            combined_settings.update(settings)
            if "ccxt_config" in settings:
                merged = dict({"timeout": 15_000})
                merged.update(settings["ccxt_config"])
                combined_settings["ccxt_config"] = merged
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "bitfinex"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
        )


__all__ = ["BitfinexSpotAdapter"]

