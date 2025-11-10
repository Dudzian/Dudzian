"""Adapter CCXT futures dla BitMEX."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from bot_core.exchanges._long_poll_ccxt import CCXTLongPollMixin
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule
from bot_core.exchanges.health import Watchdog


class BitmexFuturesAdapter(CCXTLongPollMixin, WatchdogCCXTAdapter):
    """Adapter futures BitMEX zintegrowany z watchdogiem i long-pollem."""

    name = "bitmex_futures"

    _STREAM_DEFAULTS: Mapping[str, Any] = {
        "base_url": "http://127.0.0.1:8765",
        "public_path": "/stream/bitmex_futures/public",
        "private_path": "/stream/bitmex_futures/private",
        "poll_interval": 0.35,
    }

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
                "options": {"defaultType": "future"},
            },
            "rate_limit_rules": (
                RateLimitRule(rate=40, per=1.0),
                RateLimitRule(rate=600, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 5,
                "base_delay": 0.3,
                "max_delay": 2.0,
                "jitter": (0.05, 0.35),
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
            watchdog=watchdog,
        )

    def _stream_defaults(self) -> Mapping[str, Any]:
        return self._STREAM_DEFAULTS

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_long_poll_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if not self.credentials.secret:
            raise PermissionError("Adapter BitMEX futures wymaga secret do kanałów prywatnych.")
        return self._build_long_poll_stream("private", channels)


__all__ = ["BitmexFuturesAdapter"]

