"""Adapter CCXT dla rynku spot MEXC."""

from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter, merge_adapter_settings


class MexcSpotAdapter(CCXTSpotAdapter):
    """Adapter CCXT obsługujący handel spot na MEXC."""

    name = "mexc_spot"

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
                "ccxt_config": {
                    "timeout": 20_000,
                    "options": {
                        "defaultType": "spot",
                    },
                },
                "fetch_ohlcv_params": {
                    "type": "spot",
                },
                "cancel_order_params": {
                    "type": "spot",
                },
            },
            settings or {},
        )

        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "mexc"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
        )


__all__ = ["MexcSpotAdapter"]
