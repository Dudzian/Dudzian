"""Adapter REST dla Kraken Futures zgodny z interfejsem ``ExchangeAdapter``."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.error_mapping import raise_for_kraken_error
from bot_core.exchanges.errors import ExchangeAPIError
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.exchanges.http_client import urlopen

_API_PREFIX = "/derivatives/api/v3"
_BASE_ORIGINS: Mapping[Environment, str] = {
    Environment.LIVE: "https://futures.kraken.com",
    Environment.PAPER: "https://futures.kraken.com",
    Environment.TESTNET: "https://demo-futures.kraken.com",
}

_INTERVAL_MAPPING: Mapping[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


@dataclass(slots=True)
class _RequestContext:
    path: str
    method: str
    params: Mapping[str, Any]
    body: Mapping[str, Any] | None = None


class KrakenFuturesAdapter(ExchangeAdapter):
    """Obsługuje publiczne i prywatne endpointy Kraken Futures."""

    name = "kraken_futures"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment,
        settings: Mapping[str, object] | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment
        try:
            self._origin = _BASE_ORIGINS[environment]
        except KeyError as exc:  # pragma: no cover - brak konfiguracji w testach
            raise ValueError(f"Nieobsługiwane środowisko Kraken Futures: {environment}") from exc
        self._permission_set = frozenset(str(perm).lower() for perm in credentials.permissions)
        self._ip_allowlist: Sequence[str] | None = None
        self._http_timeout = 20
        self._settings = dict(settings or {})
        self._watchdog = watchdog or Watchdog()

    # ------------------------------------------------------------------
    # Konfiguracja streamingu long-pollowego
    # ------------------------------------------------------------------
    def _stream_settings(self) -> Mapping[str, object]:
        raw = self._settings.get("stream")
        if isinstance(raw, Mapping):
            return raw
        return {}

    def _build_stream(self, scope: str, channels: Sequence[str]) -> LocalLongPollStream:
        stream_settings = dict(self._stream_settings())
        base_url = str(
            stream_settings.get("base_url", self._settings.get("stream_base_url", "http://127.0.0.1:8765"))
        )
        default_path = f"/stream/{self.name}/{scope}"
        path = str(
            stream_settings.get(
                f"{scope}_path",
                self._settings.get(f"stream_{scope}_path", default_path),
            )
            or default_path
        )
        poll_interval = float(
            stream_settings.get(
                "poll_interval",
                self._settings.get("stream_poll_interval", 0.5),
            )
        )
        timeout = float(stream_settings.get("timeout", self._settings.get("stream_timeout", 10.0)))
        max_retries = int(stream_settings.get("max_retries", self._settings.get("stream_max_retries", 3)))
        backoff_base = float(
            stream_settings.get("backoff_base", self._settings.get("stream_backoff_base", 0.25))
        )
        backoff_cap = float(
            stream_settings.get("backoff_cap", self._settings.get("stream_backoff_cap", 2.0))
        )
        jitter = stream_settings.get("jitter", self._settings.get("stream_jitter", (0.05, 0.30)))
        channel_param = stream_settings.get(f"{scope}_channel_param")
        if channel_param is None:
            channel_param = stream_settings.get(
                "channel_param", self._settings.get("stream_channel_param", "channels")
            )
        cursor_param = stream_settings.get(f"{scope}_cursor_param")
        if cursor_param is None:
            cursor_param = stream_settings.get(
                "cursor_param", self._settings.get("stream_cursor_param", "cursor")
            )
        initial_cursor = stream_settings.get(f"{scope}_initial_cursor")
        if initial_cursor is None:
            initial_cursor = stream_settings.get("initial_cursor")
        channel_serializer = None
        serializer_candidate = stream_settings.get(f"{scope}_channel_serializer")
        if not callable(serializer_candidate):
            serializer_candidate = stream_settings.get("channel_serializer")
        if callable(serializer_candidate):
            channel_serializer = serializer_candidate
        else:
            separator = stream_settings.get(f"{scope}_channel_separator")
            if separator is None:
                separator = stream_settings.get(
                    "channel_separator", self._settings.get("stream_channel_separator", ",")
                )
            if isinstance(separator, str):
                channel_serializer = lambda values, sep=separator: sep.join(values)  # noqa: E731
        headers_raw = stream_settings.get("headers")
        header_map = dict(headers_raw) if isinstance(headers_raw, Mapping) else None
        params: dict[str, object] = {}
        base_params = stream_settings.get("params")
        if isinstance(base_params, Mapping):
            params.update(base_params)
        scope_params = stream_settings.get(f"{scope}_params")
        if isinstance(scope_params, Mapping):
            params.update(scope_params)
        token_key = f"{scope}_token"
        if isinstance(stream_settings.get(token_key), str):
            params.setdefault("token", stream_settings[token_key])
        elif isinstance(stream_settings.get("auth_token"), str):
            params.setdefault("token", stream_settings["auth_token"])
        http_method = stream_settings.get(f"{scope}_method")
        if http_method is None:
            http_method = stream_settings.get("method", "GET")
        params_in_body = stream_settings.get(f"{scope}_params_in_body")
        if params_in_body is None:
            params_in_body = stream_settings.get("params_in_body", False)
        channels_in_body = stream_settings.get(f"{scope}_channels_in_body")
        if channels_in_body is None:
            channels_in_body = stream_settings.get("channels_in_body", False)
        cursor_in_body = stream_settings.get(f"{scope}_cursor_in_body")
        if cursor_in_body is None:
            cursor_in_body = stream_settings.get("cursor_in_body", False)
        body_params: dict[str, object] = {}
        base_body = stream_settings.get("body_params")
        if isinstance(base_body, Mapping):
            body_params.update(base_body)
        scope_body = stream_settings.get(f"{scope}_body_params")
        if isinstance(scope_body, Mapping):
            body_params.update(scope_body)
        body_encoder = stream_settings.get(f"{scope}_body_encoder")
        if body_encoder is None:
            body_encoder = stream_settings.get("body_encoder")

        buffer_size_raw = stream_settings.get(f"{scope}_buffer_size")
        if buffer_size_raw is None:
            buffer_size_raw = stream_settings.get("buffer_size", 64)
        try:
            buffer_size = int(buffer_size_raw)
        except (TypeError, ValueError):
            buffer_size = 64
        if buffer_size < 1:
            buffer_size = 1

        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=self.name,
            scope=scope,
            environment=self._environment.value,
            params=params,
            headers=header_map,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter if isinstance(jitter, Sequence) else (0.05, 0.30),
            channel_param=str(channel_param).strip() if channel_param not in (None, "") else "",
            cursor_param=str(cursor_param).strip() if cursor_param not in (None, "") else "",
            initial_cursor=initial_cursor,
            channel_serializer=channel_serializer,
            http_method=str(http_method or "GET"),
            params_in_body=bool(params_in_body),
            channels_in_body=bool(channels_in_body),
            cursor_in_body=bool(cursor_in_body),
            body_params=body_params or None,
            body_encoder=body_encoder,
            buffer_size=buffer_size,
            metrics_registry=self._metrics,
        )

    # ------------------------------------------------------------------
    # Konfiguracja sieci
    # ------------------------------------------------------------------
    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # type: ignore[override]
        self._ip_allowlist = tuple(ip_allowlist) if ip_allowlist else None

    # ------------------------------------------------------------------
    # Dane konta i publiczne API
    # ------------------------------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        if "read" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken Futures nie mają uprawnień do odczytu.")

        context = _RequestContext(path="/accounts", method="GET", params={})
        payload = self._watchdog.execute(
            "kraken_futures_private_request",
            lambda: self._private_request(context),
        )
        accounts = payload.get("accounts", {}) if isinstance(payload, Mapping) else {}
        futures = accounts.get("futures", {}) if isinstance(accounts, Mapping) else {}

        balances: MutableMapping[str, float] = {}
        raw_balances = futures.get("balances") if isinstance(futures, Mapping) else None
        if isinstance(raw_balances, Mapping):
            for asset, info in raw_balances.items():
                if isinstance(info, Mapping):
                    amount = info.get("balance") or info.get("available")
                else:
                    amount = info
                try:
                    balances[str(asset)] = float(amount) if amount is not None else 0.0
                except (TypeError, ValueError):
                    continue

        total_equity = _to_float(futures.get("accountValue"))
        available_margin = _to_float(futures.get("availableMargin"))
        maintenance_margin = _to_float(futures.get("maintenanceMargin"))
        if maintenance_margin == 0.0:
            maintenance_margin = _to_float(futures.get("initialMargin"))

        return AccountSnapshot(
            balances=dict(balances),
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=maintenance_margin,
        )

    def fetch_symbols(self) -> Sequence[str]:  # type: ignore[override]
        def _call() -> Mapping[str, Any]:
            return self._public_request("/instruments", params={})

        payload = self._watchdog.execute("kraken_futures_fetch_symbols", _call)
        instruments = payload.get("instruments") if isinstance(payload, Mapping) else None
        symbols: list[str] = []
        if isinstance(instruments, Sequence):
            for item in instruments:
                symbol = item.get("symbol") if isinstance(item, Mapping) else None
                if isinstance(symbol, str):
                    symbols.append(symbol)
        return sorted(set(symbols))

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # type: ignore[override]
        resolution = _INTERVAL_MAPPING.get(interval)
        if resolution is None:
            raise ValueError(f"Nieobsługiwany interwał {interval!r} dla Kraken Futures")

        params: MutableMapping[str, Any] = {"symbol": symbol, "resolution": resolution}
        if start is not None:
            params["from"] = int(start)
        if end is not None:
            params["to"] = int(end)

        def _call() -> Mapping[str, Any]:
            return self._public_request("/ohlc", params=params)

        payload = self._watchdog.execute("kraken_futures_fetch_ohlcv", _call)
        candles: list[Sequence[float]] = []
        series = payload.get("series") if isinstance(payload, Mapping) else None
        if isinstance(series, Sequence):
            for entry in series:
                if not isinstance(entry, Mapping):
                    continue
                data = entry.get("data")
                if isinstance(data, Sequence):
                    for candle in data:
                        if isinstance(candle, Sequence) and len(candle) >= 6:
                            ts, o, h, l, c, v = candle[:6]
                            candles.append([float(ts), float(o), float(h), float(l), float(c), float(v)])
                break
        if limit is not None:
            candles = candles[:limit]
        return candles

    # ------------------------------------------------------------------
    # Operacje tradingowe
    # ------------------------------------------------------------------
    def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken Futures nie mają uprawnień tradingowych.")

        order_type = request.order_type.lower()
        if order_type == "market":
            api_order_type = "mkt"
        elif order_type == "limit":
            api_order_type = "lmt"
        else:
            raise ValueError(f"Nieobsługiwany typ zlecenia Kraken Futures: {request.order_type}")

        body: MutableMapping[str, Any] = {
            "orderType": api_order_type,
            "symbol": request.symbol,
            "side": request.side.lower(),
            "size": f"{request.quantity:.8f}",
        }
        if api_order_type == "lmt":
            if request.price is None:
                raise ValueError("Zlecenie limit na Kraken Futures wymaga ceny.")
            body["limitPrice"] = f"{request.price:.2f}"
        if request.time_in_force:
            tif = request.time_in_force.upper()
            if tif not in {"GTC", "IOC", "FOK"}:
                raise ValueError(f"Nieobsługiwane time in force '{tif}' dla Kraken Futures")
            body["timeInForce"] = tif
        if request.client_order_id:
            body["cliOrdId"] = request.client_order_id

        context = _RequestContext(path="/orders", method="POST", params={}, body=body)
        payload = self._watchdog.execute(
            "kraken_futures_place_order",
            lambda: self._private_request(context),
        )

        send_status = payload.get("sendStatus") if isinstance(payload, Mapping) else None
        order_id = ""
        if isinstance(send_status, Mapping):
            order_id = str(send_status.get("order_id") or send_status.get("orderId") or "")
            events = send_status.get("orderEvents")
        else:
            events = None
        filled_quantity = 0.0
        avg_price: float | None = None
        if isinstance(events, Sequence):
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                if event.get("type") == "fill":
                    filled_quantity += _to_float(event.get("fill_size"))
                    avg_price = _to_float(event.get("price"))
        status = "NEW"
        if isinstance(send_status, Mapping):
            status = str(send_status.get("status") or "NEW").upper()

        return OrderResult(
            order_id=order_id,
            status=status,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            raw_response=send_status if isinstance(send_status, Mapping) else {},
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken Futures nie mają uprawnień tradingowych.")

        context = _RequestContext(path=f"/orders/{order_id}", method="DELETE", params={})
        payload = self._watchdog.execute(
            "kraken_futures_cancel_order",
            lambda: self._private_request(context),
        )
        result = payload.get("cancelStatus") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping) or str(result.get("status")).lower() != "cancelled":
            raise RuntimeError(f"Kraken Futures nie potwierdził anulowania zlecenia: {payload}")

    # ------------------------------------------------------------------
    # Streaming (do implementacji w dalszym etapie)
    # ------------------------------------------------------------------
    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia Kraken Futures nie pozwalają na prywatny stream danych.")
        return self._build_stream("private", channels)

    # ------------------------------------------------------------------
    # Wewnętrzne narzędzia HTTP/podpisy
    # ------------------------------------------------------------------
    def _public_request(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self._origin}{_API_PREFIX}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "bot-core/kraken-futures"})
        with urlopen(request, timeout=self._http_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        _ensure_success(payload)
        return payload

    def _private_request(self, context: _RequestContext) -> Mapping[str, Any]:
        if not self.credentials.secret:
            raise PermissionError("Poświadczenia Kraken Futures wymagają sekretu do wywołań prywatnych.")

        path = _normalize_path(context.path)
        query = f"?{urlencode(context.params)}" if context.params else ""
        url = f"{self._origin}{_API_PREFIX}{path}{query}"
        body_bytes = b""
        if context.body is not None:
            body_bytes = json.dumps(
                context.body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        nonce = str(int(time.time() * 1000))
        message = nonce.encode("utf-8") + path.encode("utf-8") + query.encode("utf-8") + body_bytes
        sha_digest = hashlib.sha256(message).digest()
        decoded_secret = base64.b64decode(self.credentials.secret)
        mac = hmac.new(decoded_secret, sha_digest, hashlib.sha256)
        signature = base64.b64encode(mac.digest()).decode("utf-8")

        headers = {
            "User-Agent": "bot-core/kraken-futures",
            "Content-Type": "application/json",
            "APIKey": self.credentials.key_id,
            "Authent": signature,
            "Nonce": nonce,
        }
        request = Request(url, data=body_bytes if body_bytes else None, headers=headers, method=context.method)
        with urlopen(request, timeout=self._http_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        _ensure_success(payload)
        return payload


def _normalize_path(path: str) -> str:
    path = path.strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _to_float(value: Any) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _ensure_success(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ExchangeAPIError(
            message="Kraken Futures API zwróciło nieoczekiwaną odpowiedź.",
            status_code=400,
            payload=payload,
        )

    errors = payload.get("error")
    if isinstance(errors, Sequence):
        filtered = [item for item in errors if item]
        if filtered:
            raise_for_kraken_error(
                payload=payload,
                default_message="Kraken Futures API zwróciło błąd",
            )

    result = payload.get("result")
    if isinstance(result, str) and result.lower() == "success":
        return
    if isinstance(result, Mapping):
        status = result.get("status") or result.get("result")
        if isinstance(status, str) and status.lower() in {"success", "ok"}:
            return

    send_status = payload.get("sendStatus")
    if isinstance(send_status, str) and send_status.lower() in {"acknowledged", "ok", "success"}:
        return

    if isinstance(errors, Sequence) and not errors:
        return

    raise ExchangeAPIError(
        message="Kraken Futures API zwróciło błąd.",
        status_code=400,
        payload=payload,
    )


__all__ = ["KrakenFuturesAdapter"]
