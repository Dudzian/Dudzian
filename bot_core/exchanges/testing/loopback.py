"""Prosty adapter giełdowy używany w scenariuszach E2E i testach hermetycznych.

Adapter komunikuje się z lokalnym serwerem HTTP, który udostępnia minimalny
zestaw endpointów pozwalających na symulację środowiska paper/testnet. Dzięki
temu scenariusze end-to-end mogą uruchamiać pełny `LiveExecutionRouter`
bez łączenia się z zewnętrznymi giełdami.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import httpx

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeNetworkError


@dataclass(slots=True)
class _LoopbackStream:
    """Obiekt reprezentujący połączenie streamingowe z serwerem loopback."""

    client: httpx.Client
    endpoint: str
    payload: MutableMapping[str, object]

    def close(self) -> None:
        try:
            self.client.post(self.endpoint, json={"action": "close", **self.payload}, timeout=5.0)
        except httpx.HTTPError:
            # Zamknięcie streamu jest operacją typu best-effort — brak wyjątków.
            pass


class LoopbackExchangeAdapter(ExchangeAdapter):
    """Adapter komunikujący się z lokalnym serwerem HTTP sterującym testnetem."""

    name = "loopback_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        live_settings = self._extract_live_settings(settings)
        self._base_url = str(live_settings.get("base_url", "http://127.0.0.1:39765"))
        self._endpoints = {
            "account": str(live_settings.get("account_endpoint", "/account")),
            "symbols": str(live_settings.get("symbols_endpoint", "/symbols")),
            "ohlcv": str(live_settings.get("ohlcv_endpoint", "/ohlcv")),
            "orders": str(live_settings.get("orders_endpoint", "/orders")),
            "stream_public": str(live_settings.get("stream_public_endpoint", "/stream/public")),
            "stream_private": str(live_settings.get("stream_private_endpoint", "/stream/private")),
        }
        self._client = httpx.Client(base_url=self._base_url, timeout=httpx.Timeout(5.0))

    @staticmethod
    def _extract_live_settings(settings: Mapping[str, object] | None) -> Mapping[str, object]:
        if not settings:
            return {}
        if "live_trading" in settings:
            candidate = settings["live_trading"]
            if isinstance(candidate, Mapping):
                return candidate
        return settings

    # --- Metody ExchangeAdapter -------------------------------------------------

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401
        del ip_allowlist
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        response = self._request("GET", self._endpoints["account"])
        payload = response.json()
        balances = {str(asset): float(value) for asset, value in payload.get("balances", {}).items()}
        return AccountSnapshot(
            balances=balances,
            total_equity=float(payload.get("total_equity", 0.0)),
            available_margin=float(payload.get("available_margin", 0.0)),
            maintenance_margin=float(payload.get("maintenance_margin", 0.0)),
        )

    def fetch_symbols(self) -> Sequence[str]:
        response = self._request("GET", self._endpoints["symbols"])
        payload = response.json()
        entries = payload.get("symbols", [])
        return tuple(str(symbol) for symbol in entries)

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        params = {
            "symbol": symbol,
            "interval": interval,
        }
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
        if limit is not None:
            params["limit"] = int(limit)
        response = self._request("GET", self._endpoints["ohlcv"], params=params)
        payload = response.json()
        rows = payload.get("rows", [])
        return [list(map(float, row)) for row in rows]

    def place_order(self, request: OrderRequest) -> OrderResult:
        response = self._request(
            "POST",
            self._endpoints["orders"],
            json={
                "symbol": request.symbol,
                "side": request.side,
                "type": request.order_type,
                "quantity": request.quantity,
                "price": request.price,
                "time_in_force": getattr(request, "time_in_force", None),
                "client_order_id": getattr(request, "client_order_id", None),
            },
        )
        payload = response.json()
        return OrderResult(
            order_id=str(payload.get("order_id", "loopback-order")),
            status=str(payload.get("status", "filled")),
            filled_quantity=float(payload.get("filled_quantity", request.quantity)),
            avg_price=float(payload.get("avg_price", request.price or 0.0)),
            raw_response=payload,
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        endpoint = f"{self._endpoints['orders'].rstrip('/')}/{order_id}"
        payload: MutableMapping[str, object] = {}
        if symbol is not None:
            payload["symbol"] = symbol
        self._request("DELETE", endpoint, json=payload or None)

    def stream_public_data(self, *, channels: Sequence[str]) -> _LoopbackStream:
        return self._open_stream(self._endpoints["stream_public"], channels)

    def stream_private_data(self, *, channels: Sequence[str]) -> _LoopbackStream:
        return self._open_stream(self._endpoints["stream_private"], channels)

    # --- Pomocnicze ------------------------------------------------------------

    def _open_stream(self, endpoint: str, channels: Sequence[str]) -> _LoopbackStream:
        payload = {"channels": list(channels)}
        self._request("POST", endpoint, json={"action": "open", **payload})
        return _LoopbackStream(client=self._client, endpoint=endpoint, payload=payload)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, object] | None = None,
        json: Mapping[str, object] | None = None,
    ) -> httpx.Response:
        url = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        try:
            response = self._client.request(method, url, params=params, json=json)
        except httpx.TimeoutException as exc:
            raise ExchangeNetworkError("timeout", str(exc)) from exc
        except httpx.HTTPError as exc:  # noqa: BLE001
            raise ExchangeNetworkError("http_error", str(exc)) from exc
        if response.status_code >= 500:
            raise ExchangeNetworkError("server_error", response.text)
        if response.status_code >= 400:
            raise ExchangeAPIError(response.status_code, response.text)
        return response

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensywne sprzątanie
            pass


__all__ = ["LoopbackExchangeAdapter"]
