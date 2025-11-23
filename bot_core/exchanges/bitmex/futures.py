"""Adapter CCXT futures dla BitMEX."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bot_core.exchanges._long_poll_ccxt import CCXTLongPollMixin
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import WatchdogCCXTAdapter, merge_adapter_settings
from bot_core.exchanges.hypercare import HypercareChecklistExporter
from bot_core.exchanges.error_mapping import raise_for_bitmex_error
from bot_core.exchanges.errors import ExchangeAPIError
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.rate_limiter import RateLimitRule
from bot_core.exchanges.signal_quality import SignalQualityReporter
from bot_core.exchanges.streaming import LocalLongPollStream


class BitmexFuturesAdapter(CCXTLongPollMixin, WatchdogCCXTAdapter):
    """Adapter futures BitMEX zintegrowany z watchdogiem i long-pollem."""

    name = "bitmex_futures"
    hypercare_checklist_id = "stage6-bitmex-futures-2024q4"

    _STREAM_DEFAULTS: Mapping[str, Any] = {
        "base_url": "https://stream.hyperion.dudzian.ai/exchanges",
        "public_path": "/bitmex/futures/public",
        "private_path": "/bitmex/futures/private",
        "poll_interval": 0.35,
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
                RateLimitRule(rate=600, per=60.0),
            ),
            "retry_policy": {
                "max_attempts": 5,
                "base_delay": 0.3,
                "max_delay": 2.5,
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

    def stream_order_book(self, symbol: str, *, depth: int = 50) -> LocalLongPollStream:
        return self.stream_public_data(channels=[f"order_book:{symbol}:{depth}"])

    def stream_ticker(self, symbol: str) -> LocalLongPollStream:
        return self.stream_public_data(channels=[f"ticker:{symbol}"])

    def stream_fills(self, symbol: str) -> LocalLongPollStream:
        return self.stream_private_data(channels=[f"fills:{symbol}"])

    def fetch_order_book(
        self,
        symbol: str,
        *,
        limit: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
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
                    raise_for_bitmex_error(
                        status_code=exc.status_code or 500,
                        payload=payload,
                        default_message=str(exc),
                    )
                except ExchangeAPIError as mapped:
                    raise mapped from exc
            raise

    @classmethod
    def export_hypercare_assets(
        cls,
        *,
        report_dir: str | None = None,
        signal_quality_dir: str | None = None,
        daily_csv_dir: str | None = None,
        reporter: SignalQualityReporter | None = None,
        load_existing_snapshot: bool = True,
    ) -> tuple[str, str | None]:
        """Publikuje checklistę HyperCare i snapshot jakości sygnałów."""
        signal_root = Path(signal_quality_dir) if signal_quality_dir else Path("reports/exchanges/signal_quality")
        signal_root.mkdir(parents=True, exist_ok=True)

        quality_reporter = reporter
        quality_snapshot: Path | None = None

        if quality_reporter is None and load_existing_snapshot:
            existing_snapshot = signal_root / f"{cls.name}.json"
            if existing_snapshot.exists():
                quality_snapshot = existing_snapshot

        if quality_reporter is None and quality_snapshot is None:
            quality_reporter = SignalQualityReporter(
                exchange_id=cls.name,
                report_dir=signal_root,
                enable_csv_export=True,
                csv_dir=signal_root,
            )

        if quality_reporter is not None:
            quality_snapshot = quality_reporter.write_snapshot()

        checklist = HypercareChecklistExporter(
            exchange=cls.name,
            checklist_id=cls.hypercare_checklist_id,
            signed_by="exchange_ops",
        )
        checklist_json, checklist_csv = checklist.export(
            report_dir=report_dir or "reports/exchanges/hypercare",
            signal_quality_snapshot=quality_snapshot,
            daily_csv_dir=daily_csv_dir or "reports/exchanges",
        )
        return str(checklist_json), str(checklist_csv) if checklist_csv else None


__all__ = ["BitmexFuturesAdapter"]

