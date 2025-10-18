"""Adapter CCXT dla rynku spot OKX."""
from __future__ import annotations

from typing import Any, Mapping

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter


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
        combined_settings: dict[str, Any] = {
            "ccxt_config": {"timeout": 20_000},
            "fetch_ohlcv_params": {"price": "mark"},
        }
        if settings:
            combined_settings.update(settings)
            if "ccxt_config" in settings:
                merged = dict({"timeout": 20_000})
                merged.update(settings["ccxt_config"])
                combined_settings["ccxt_config"] = merged
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

