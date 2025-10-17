"""Minimalistyczny adapter REST dla rynku spot nowa_gielda."""
from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.nowa_gielda import symbols

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """Opis pojedynczego limitu wagowego endpointu REST."""

    method: str
    path: str
    weight: int
    window_seconds: float
    max_requests: int

    @property
    def key(self) -> str:
        return f"{self.method.upper()} {self.path}".strip()


_RATE_LIMITS: Mapping[str, RateLimitRule] = {
    rule.key: rule
    for rule in (
        RateLimitRule("GET", "/public/ticker", weight=1, window_seconds=1.0, max_requests=20),
        RateLimitRule("GET", "/public/orderbook", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("POST", "/private/orders", weight=5, window_seconds=1.0, max_requests=5),
        RateLimitRule("DELETE", "/private/orders", weight=1, window_seconds=1.0, max_requests=10),
    )
}


def _canonical_payload(data: Mapping[str, Any] | None) -> str:
    if not data:
        return ""
    items: list[tuple[str, str]] = []
    for key, value in data.items():
        if value is None:
            continue
        items.append((str(key), str(value)))
    items.sort()
    return "&".join(f"{key}={value}" for key, value in items)


class NowaGieldaSpotAdapter(ExchangeAdapter):
    """Adapter implementujący podstawowe operacje dla nowa_gielda."""

    __slots__ = ("_environment", "_ip_allowlist", "_metrics")

    name: str = "nowa_gielda_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._ip_allowlist: tuple[str, ...] = ()
        self._metrics: Mapping[str, Any] | None = None

    # --- Utilities ---------------------------------------------------------
    @staticmethod
    def _timestamp() -> int:
        return int(time.time() * 1000)

    def _secret(self) -> bytes:
        secret = self.credentials.secret or ""
        return secret.encode("utf-8")

    # --- Configuration -----------------------------------------------------
    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        self._ip_allowlist = tuple(ip_allowlist or ())
        if self._ip_allowlist:
            _LOGGER.debug("Skonfigurowano allowlistę IP: %s", self._ip_allowlist)

    # --- ExchangeAdapter API -----------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 0.0},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        return symbols.supported_internal_symbols()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        payload = {
            "symbol": symbols.to_exchange_symbol(request.symbol),
            "side": request.side,
            "type": request.order_type,
            "quantity": request.quantity,
            "price": request.price,
        }
        timestamp = self._timestamp()
        signature = self.sign_request(timestamp, "POST", "/private/orders", body=payload)
        headers = self.build_auth_headers(timestamp, signature)
        _LOGGER.debug("Składanie zlecenia %s z nagłówkami %s", payload, headers)
        return OrderResult(
            order_id="simulated-order",
            status="accepted",
            filled_quantity=0.0,
            avg_price=None,
            raw_response={"payload": payload, "headers": headers},
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        _LOGGER.debug("Anulowanie zlecenia %s dla symbolu %s", order_id, symbol)

    def stream_public_data(self, *, channels: Sequence[str]) -> Protocol:
        raise NotImplementedError("Strumień publiczny nie jest wspierany w testowym adapterze")

    def stream_private_data(self, *, channels: Sequence[str]) -> Protocol:
        raise NotImplementedError("Strumień prywatny nie jest wspierany w testowym adapterze")

    # --- Custom helpers ----------------------------------------------------
    def sign_request(
        self,
        timestamp: int,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> str:
        canonical = _canonical_payload(body)
        message = f"{timestamp}{method.upper()}{path}{canonical}".encode("utf-8")
        return hmac.new(self._secret(), message, sha256).hexdigest()

    def build_auth_headers(self, timestamp: int, signature: str) -> Mapping[str, str]:
        return {
            "X-API-KEY": self.credentials.key_id,
            "X-API-SIGN": signature,
            "X-API-TIMESTAMP": str(timestamp),
        }

    def rate_limit_rule(self, method: str, path: str) -> RateLimitRule | None:
        key = f"{method.upper()} {path}".strip()
        return _RATE_LIMITS.get(key)

    def request_weight(self, method: str, path: str) -> int:
        rule = self.rate_limit_rule(method, path)
        return rule.weight if rule else 1


__all__ = ["NowaGieldaSpotAdapter", "RateLimitRule"]
