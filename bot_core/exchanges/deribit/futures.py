"""Adapter CCXT futures dla Deribit."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from bot_core.exchanges._long_poll_ccxt import CCXTLongPollMixin
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.error_mapping import raise_for_deribit_error
from bot_core.exchanges.errors import ExchangeAPIError
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.rate_limiter import RateLimitRule


class DeribitFuturesAdapter(CCXTLongPollMixin, WatchdogCCXTAdapter):
    """Adapter futures Deribit integrujący watchdog oraz long-poll."""

    name = "deribit_futures"

    _STREAM_DEFAULTS: Mapping[str, Any] = {
        "base_url": "https://stream.hyperion.dudzian.ai/exchanges",
        "public_path": "/deribit/futures/public",
        "private_path": "/deribit/futures/private",
        "poll_interval": 0.5,
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
                RateLimitRule(rate=1200, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 5,
                "base_delay": 0.4,
                "max_delay": 3.0,
                "jitter": (0.05, 0.35),
            },
            "stream": dict(self._STREAM_DEFAULTS),
        }
        combined_settings = merge_adapter_settings(defaults, settings or {})
        stream_settings = combined_settings.setdefault("stream", {})

        user_stream_settings: Mapping[str, Any] = {}
        raw_stream = (settings or {}).get("stream") if isinstance(settings, Mapping) else None
        if isinstance(raw_stream, Mapping):
            user_stream_settings = raw_stream

        environment_base = self._STREAM_ENDPOINTS.get(environment)
        if environment_base:
            override_key = f"{environment.value}_base_url"
            env_override = stream_settings.get(override_key)
            if env_override:
                stream_settings["base_url"] = env_override
            elif "base_url" not in user_stream_settings:
                stream_settings["base_url"] = environment_base
        super().__init__(
            credentials,
            exchange_id=combined_settings.pop("exchange_id", "deribit"),
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
            raise PermissionError("Adapter Deribit futures wymaga secret do kanałów prywatnych.")
        return self._build_long_poll_stream("private", channels)

    @staticmethod
    def _decode_error_payload(payload: object) -> Mapping[str, Any] | None:
        if isinstance(payload, Mapping):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = payload.decode("utf-8")
            except Exception:  # pragma: no cover - diagnostyka pomocnicza
                return None
        if isinstance(payload, str):
            text = payload.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    return None
                if isinstance(parsed, Mapping):
                    return parsed
        return None

    def _call_client(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        operation = f"{self.name}.{method_name}"
        try:
            return self._watchdog.execute(  # type: ignore[attr-defined]
                operation,
                lambda: super(WatchdogCCXTAdapter, self)._call_client(method_name, *args, **kwargs),
            )
        except ExchangeAPIError as exc:
            payload = self._decode_error_payload(exc.payload)
            if payload is not None:
                try:
                    raise_for_deribit_error(
                        status_code=exc.status_code or 500,
                        payload=payload,
                        default_message=str(exc),
                    )
                except ExchangeAPIError as mapped:
                    raise mapped from exc
            raise


__all__ = ["DeribitFuturesAdapter"]

