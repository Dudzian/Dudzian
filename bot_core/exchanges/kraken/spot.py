"""Adapter REST dla Kraken Spot zgodny z interfejsem `ExchangeAdapter`."""
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


_BASE_URLS: Mapping[Environment, str] = {
    Environment.LIVE: "https://api.kraken.com",
    Environment.PAPER: "https://api.kraken.com",
    Environment.TESTNET: "https://api.kraken.com",
}


_INTERVAL_MAPPING: Mapping[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


@dataclass(slots=True)
class _RequestContext:
    path: str
    params: Mapping[str, Any]


class KrakenSpotAdapter(ExchangeAdapter):
    """Implementacja publicznych i prywatnych endpointów Kraken Spot."""

    name = "kraken_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment
        try:
            self._base_url = _BASE_URLS[environment]
        except KeyError as exc:  # pragma: no cover - brak konfiguracji
            raise ValueError(f"Nieobsługiwane środowisko Kraken: {environment}") from exc
        self._http_timeout = 15
        self._permission_set = set(credentials.permissions)
        self._ip_allowlist: Sequence[str] | None = None
        self._last_nonce = 0
        self._settings = dict(settings or {})
        asset = str(self._settings.get("valuation_asset", "ZUSD") or "ZUSD").strip().upper()
        if asset and not asset.startswith("Z") and len(asset) <= 4:
            asset = f"Z{asset}"
        self._valuation_asset = asset or "ZUSD"

    # ------------------------------------------------------------------
    # Konfiguracja sieciowa
    # ------------------------------------------------------------------
    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # type: ignore[override]
        self._ip_allowlist = tuple(ip_allowlist) if ip_allowlist else None

    # ------------------------------------------------------------------
    # Dane konta i publiczne API
    # ------------------------------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        if "read" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień do odczytu.")
        balances_payload = self._private_request(_RequestContext(path="/0/private/Balance", params={}))
        trade_balance_payload = self._private_request(
            _RequestContext(path="/0/private/TradeBalance", params={"asset": self._valuation_asset})
        )

        balances_data = balances_payload.get("result", {}) if isinstance(balances_payload, Mapping) else {}
        balances: MutableMapping[str, float] = {}
        for asset, amount in balances_data.items():
            try:
                balances[asset] = float(amount)
            except (TypeError, ValueError):
                continue

        trade_data = trade_balance_payload.get("result", {}) if isinstance(trade_balance_payload, Mapping) else {}
        total_equity = float(trade_data.get("eb", trade_data.get("e", 0.0)) or 0.0)
        available_margin = float(trade_data.get("mf", 0.0) or 0.0)
        maintenance_margin = float(trade_data.get("m", 0.0) or 0.0)

        return AccountSnapshot(
            balances=dict(balances),
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=maintenance_margin,
        )

    def fetch_symbols(self) -> Sequence[str]:  # type: ignore[override]
        payload = self._public_request("/0/public/AssetPairs", params={})
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        symbols: list[str] = []
        for value in result.values():
            if isinstance(value, Mapping):
                altname = value.get("altname") or value.get("wsname")
                if isinstance(altname, str):
                    symbols.append(altname)
        return sorted(set(symbols))

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # type: ignore[override]
        minutes = _INTERVAL_MAPPING.get(interval)
        if minutes is None:
            raise ValueError(f"Nieobsługiwany interwał {interval!r} dla Kraken Spot")

        params: MutableMapping[str, Any] = {"pair": symbol, "interval": minutes}
        if start:
            params["since"] = int(start)
        payload = self._public_request("/0/public/OHLC", params=params)
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        candles = []
        for values in result.values():
            if isinstance(values, Sequence):
                for candle in values:
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
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień tradingowych.")

        params: MutableMapping[str, Any] = {
            "pair": request.symbol,
            "type": request.side.lower(),
            "volume": f"{request.quantity:.10f}",
        }

        order_type = request.order_type.lower()
        if order_type == "market":
            params["ordertype"] = "market"
        elif order_type == "limit":
            if request.price is None:
                raise ValueError("Zlecenie limit na Kraken wymaga ceny.")
            params["ordertype"] = "limit"
            params["price"] = f"{request.price:.10f}"
        else:
            raise ValueError(f"Nieobsługiwany typ zlecenia dla Kraken Spot: {request.order_type}")

        if request.time_in_force:
            tif = request.time_in_force.upper()
            if tif not in {"GTC", "IOC", "GTD"}:
                raise ValueError(f"Nieobsługiwane time in force '{tif}' dla Kraken Spot")
            params["timeinforce"] = tif
        if request.client_order_id:
            params["userref"] = request.client_order_id

        payload = self._private_request(_RequestContext(path="/0/private/AddOrder", params=params))
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        txid_seq = result.get("txid") if isinstance(result, Mapping) else None
        txid: str | None = None
        if isinstance(txid_seq, Sequence) and txid_seq:
            first = txid_seq[0]
            if isinstance(first, str):
                txid = first
        return OrderResult(
            order_id=txid or "",
            status="NEW",
            filled_quantity=0.0,
            avg_price=None,
            raw_response=result if isinstance(result, Mapping) else {},
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień tradingowych.")
        params: Mapping[str, Any] = {"txid": order_id}
        payload = self._private_request(_RequestContext(path="/0/private/CancelOrder", params=params))
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(result, Mapping) or int(result.get("count", 0)) < 1:
            raise RuntimeError(f"Kraken nie potwierdził anulowania zlecenia: {payload}")

    # ------------------------------------------------------------------
    # Streaming (do implementacji w dalszych etapach)
    # ------------------------------------------------------------------
    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming publiczny Kraken zostanie dodany w przyszłym etapie.")

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming prywatny Kraken zostanie dodany w przyszłym etapie.")

    # ------------------------------------------------------------------
    # Wewnętrzne narzędzia HTTP/podpisy
    # ------------------------------------------------------------------
    def _public_request(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self._base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "bot-core/kraken-spot"})
        with urlopen(request, timeout=self._http_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._ensure_no_error(payload)
        return payload

    def _private_request(self, context: _RequestContext) -> Mapping[str, Any]:
        if not self.credentials.secret:
            raise PermissionError("Poświadczenia Kraken wymagają sekretu do wywołań prywatnych.")

        nonce = self._generate_nonce()
        sorted_items = [(key, context.params[key]) for key in sorted(context.params.keys())]
        post_items = [("nonce", nonce)] + [(k, v) for k, v in sorted_items]
        encoded = urlencode(post_items)
        data = encoded.encode("utf-8")

        encoded_params = urlencode(sorted_items)
        message = (nonce + encoded_params).encode("utf-8")
        sha_digest = hashlib.sha256(message).digest()
        decoded_secret = base64.b64decode(self.credentials.secret)
        mac = hmac.new(decoded_secret, (context.path.encode("utf-8") + sha_digest), hashlib.sha512)
        signature = base64.b64encode(mac.digest()).decode("utf-8")

        headers = {
            "User-Agent": "bot-core/kraken-spot",
            "API-Key": self.credentials.key_id,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        request = Request(f"{self._base_url}{context.path}", data=data, headers=headers)
        with urlopen(request, timeout=self._http_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._ensure_no_error(payload)
        return payload

    def _ensure_no_error(self, payload: Mapping[str, Any]) -> None:
        errors = payload.get("error") if isinstance(payload, Mapping) else None
        if errors:
            raise RuntimeError(f"Kraken API zwróciło błąd: {errors}")

    def _generate_nonce(self) -> str:
        candidate = int(time.time() * 1000)
        if candidate <= self._last_nonce:
            candidate = self._last_nonce + 1
        self._last_nonce = candidate
        return str(candidate)


__all__ = ["KrakenSpotAdapter"]
