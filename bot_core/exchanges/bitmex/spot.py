"""Adapter CCXT dla BitMEX (tryb spot syntetyczny) z long-pollem."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from bot_core.exchanges._long_poll_ccxt import CCXTLongPollMixin
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule


class BitmexSpotAdapter(CCXTLongPollMixin, CCXTSpotAdapter):
    """Adapter BitMEX wykorzystujący CCXT oraz lokalny stream long-pollowy."""

    name = "bitmex_spot"

    _STREAM_DEFAULTS: Mapping[str, Any] = {
        "base_url": "http://127.0.0.1:8765",
        "public_path": "/stream/bitmex_spot/public",
        "private_path": "/stream/bitmex_spot/private",
        "poll_interval": 0.5,
    }

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
                "timeout": 20_000,
                "options": {"defaultType": "spot"},
            },
            "rate_limit_rules": (
                RateLimitRule(rate=30, per=1.0),
                RateLimitRule(rate=600, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 4,
                "base_delay": 0.25,
                "max_delay": 1.5,
                "jitter": (0.05, 0.25),
            },
            "stream": dict(self._STREAM_DEFAULTS),
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        combined_settings.setdefault("stream", {})
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "bitmex"),
            environment=environment,
            settings=combined_settings,
            client=client,
            metrics_registry=metrics_registry,
        )

    def _stream_defaults(self) -> Mapping[str, Any]:
        return self._STREAM_DEFAULTS

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_long_poll_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if not self.credentials.secret:
            raise PermissionError("Adapter BitMEX wymaga secret do kanałów prywatnych.")
        return self._build_long_poll_stream("private", channels)


__all__ = ["BitmexSpotAdapter"]

