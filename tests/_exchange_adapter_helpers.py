from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Sequence

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)


class StubExchangeAdapter(ExchangeAdapter):
    """Wielokrotnego użytku adapter giełdowy wykorzystywany w testach."""

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        name: str | None = None,
        responses: Sequence[OrderResult | Exception] | None = None,
    ) -> None:
        super().__init__(credentials)
        self.name = name or credentials.key_id
        self._responses: Deque[OrderResult | Exception] = deque(responses or ())
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.ip_allowlist: Sequence[str] | None = None

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        environment: Environment = Environment.PAPER,
        responses: Sequence[OrderResult | Exception] | None = None,
    ) -> "StubExchangeAdapter":
        credentials = ExchangeCredentials(key_id=name, environment=environment)
        return cls(credentials, name=name, responses=responses)

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401
        self.ip_allowlist = ip_allowlist

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # noqa: ARG002
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.placed.append(request)
        if not self._responses:
            return OrderResult(
                order_id="1",
                status="filled",
                filled_quantity=0.0,
                avg_price=None,
                raw_response={},
            )
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # noqa: ARG002
        self.cancelled.append(order_id)

    def stream_public_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        return self

    def stream_private_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        return self
