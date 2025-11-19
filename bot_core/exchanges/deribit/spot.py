"""Adapter CCXT dla Deribit spot z obsługą long-polla."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from bot_core.exchanges._long_poll_ccxt import CCXTLongPollMixin
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter, merge_adapter_settings
from bot_core.exchanges.rate_limiter import RateLimitRule
from bot_core.exchanges.streaming import LocalLongPollStream


class DeribitSpotAdapter(CCXTLongPollMixin, CCXTSpotAdapter):
    """Lekki adapter CCXT dla Deribit z konfiguracją long-polla."""

    name = "deribit_spot"

    _STREAM_DEFAULTS: Mapping[str, Any] = {
        "base_url": "http://127.0.0.1:8765",
        "public_path": "/stream/deribit_spot/public",
        "private_path": "/stream/deribit_spot/private",
        "poll_interval": 0.75,
    }

    _STREAM_ENDPOINTS: Mapping[Environment, str] = {
        Environment.LIVE: "https://stream.hyperion.dudzian.ai/exchanges",
        Environment.TESTNET: "https://stream.sandbox.dudzian.ai/exchanges",
        Environment.PAPER: "https://stream.sandbox.dudzian.ai/exchanges",
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
                RateLimitRule(rate=20, per=1.0),
                RateLimitRule(rate=600, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 4,
                "base_delay": 0.3,
                "max_delay": 1.5,
                "jitter": (0.05, 0.25),
            },
            "stream": dict(self._STREAM_DEFAULTS),
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        stream_settings = combined_settings.setdefault("stream", {})

        user_stream_settings: Mapping[str, Any] = {}
        raw_stream = (settings or {}).get("stream") if isinstance(settings, Mapping) else None
        if isinstance(raw_stream, Mapping):
            user_stream_settings = raw_stream

        environment_base = self._STREAM_ENDPOINTS.get(environment or Environment.LIVE)
        if environment_base:
            override_key = f"{(environment or Environment.LIVE).value}_base_url"
            env_override = stream_settings.get(override_key)
            if env_override:
                stream_settings["base_url"] = env_override
            elif "base_url" not in user_stream_settings:
                stream_settings["base_url"] = environment_base
        stream_settings.setdefault("base_url", self._STREAM_DEFAULTS["base_url"])
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "deribit"),
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
            raise PermissionError("Adapter Deribit wymaga secret do kanałów prywatnych.")
        return self._build_long_poll_stream("private", channels)

    def stream_order_book(self, symbol: str, *, depth: int = 50) -> LocalLongPollStream:
        return self.stream_public_data(channels=[f"order_book:{symbol}:{depth}"])

    def stream_ticker(self, symbol: str) -> LocalLongPollStream:
        return self.stream_public_data(channels=[f"ticker:{symbol}"])

    def stream_fills(self, symbol: str) -> LocalLongPollStream:
        return self.stream_private_data(channels=[f"fills:{symbol}"])

    def fetch_order_book(self, symbol: str, *, limit: int | None = None, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        return self._call_client("fetch_order_book", symbol, limit=limit, params=params or None)

    def fetch_ticker(self, symbol: str, *, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        return self._call_client("fetch_ticker", symbol, params=params or None)

    def fetch_my_trades(
        self,
        symbol: str,
        *,
        limit: int | None = None,
        since: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        return self._call_client("fetch_my_trades", symbol, since=since, limit=limit, params=params or None)


__all__ = ["DeribitSpotAdapter"]

