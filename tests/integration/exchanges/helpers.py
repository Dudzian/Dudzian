from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from bot_core.exchanges.base import AccountSnapshot, OrderRequest


@dataclass
class FakeOrderResponse:
    order_id: str = "order-1"
    status: str = "closed"
    filled: float = 1.0
    price: float = 100.0

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "id": self.order_id,
            "status": self.status,
            "filled": self.filled,
            "price": self.price,
        }


def make_order_request() -> OrderRequest:
    return OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        order_type="market",
    )


def build_account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": 1000.0},
        total_equity=1000.0,
        available_margin=900.0,
        maintenance_margin=50.0,
    )


class CCXTFakeClient:
    def __init__(
        self,
        *,
        fail_first_order: bool = True,
        exception: type[Exception] = RuntimeError,
    ) -> None:
        self.fail_first_order = fail_first_order
        self.create_attempts = 0
        self.exception = exception

    def load_markets(self):
        return {"BTC/USDT": {}}

    def fetch_balance(self):
        return {
            "free": {"USDT": 900.0},
            "total": {"USDT": 1000.0},
            "used": {"USDT": 100.0},
        }

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        return [[1_700_000_000.0, 100.0, 110.0, 95.0, 105.0, 5.0]]

    def create_order(self, symbol, order_type, side, amount, price, params=None):
        self.create_attempts += 1
        if self.fail_first_order and self.create_attempts == 1:
            raise self.exception("temporary failure")
        return {
            "id": "order-1",
            "status": "closed",
            "filled": amount,
            "price": price,
        }

    def cancel_order(self, order_id, symbol, params=None):
        return {"id": order_id, "status": "canceled"}
