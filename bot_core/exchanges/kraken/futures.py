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
from urllib.request import Request, urlopen

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)

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

    def __init__(self, credentials: ExchangeCredentials, *, environment: Environment) -> None:
        super().__init__(credentials)
        self._environment = environment
        try:
            self._origin = _BASE_ORIGINS[environment]
        except KeyError as exc:  # pragma: no cover - brak konfiguracji w testach
            raise ValueError(f"Nieobsługiwane środowisko Kraken Futures: {environment}") from exc
        self._permission_set = frozenset(str(perm).lower() for perm in credentials.permissions)
        self._ip_allowlist: Sequence[str] | None = None
        self._http_timeout = 20

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
        payload = self._private_request(context)
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
        payload = self._public_request("/instruments", params={})
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

        payload = self._public_request("/ohlc", params=params)
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
        payload = self._private_request(context)

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
        payload = self._private_request(context)
        result = payload.get("cancelStatus") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping) or str(result.get("status")).lower() != "cancelled":
            raise RuntimeError(f"Kraken Futures nie potwierdził anulowania zlecenia: {payload}")

    # ------------------------------------------------------------------
    # Streaming (do implementacji w dalszym etapie)
    # ------------------------------------------------------------------
    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming Kraken Futures zostanie dodany w przyszłym etapie.")

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming prywatny Kraken Futures zostanie dodany w przyszłym etapie.")

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
    result = payload.get("result") if isinstance(payload, Mapping) else None
    if isinstance(result, str) and result.lower() == "success":
        return
    if "error" in payload and not payload.get("error"):
        return
    if result is None and "sendStatus" in payload:
        return
    raise RuntimeError(f"Kraken Futures API zwróciło błąd: {payload}")


__all__ = ["KrakenFuturesAdapter"]
