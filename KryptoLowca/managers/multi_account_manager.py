"""Manager rozdzielający zlecenia pomiędzy wiele giełd."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from KryptoLowca.exchanges.interfaces import (
    ExchangeAdapter,
    ExchangeCredentials,
    MarketPayload,
    MarketSubscription,
    OrderRequest,
    OrderStatus,
    WebSocketSubscription,
)


@dataclass(slots=True)
class ManagedAccount:
    exchange: str
    account: str
    adapter: ExchangeAdapter
    weight: int = 1
    tags: set[str] = field(default_factory=set)


class MultiExchangeAccountManager:
    """Zarządza wieloma kontami, balansując obciążenie i monitorując zlecenia."""

    def __init__(self) -> None:
        self._accounts: Dict[Tuple[str, str], ManagedAccount] = {}
        self._round_robin: Deque[Tuple[str, str]] = deque()
        self._order_index: Dict[str, Tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    def register_account(
        self,
        *,
        exchange: str,
        account: str,
        adapter: ExchangeAdapter,
        weight: int = 1,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        key = (exchange, account)
        self._accounts[key] = ManagedAccount(
            exchange=exchange,
            account=account,
            adapter=adapter,
            weight=max(1, weight),
            tags=set(tags or []),
        )
        for _ in range(max(1, weight)):
            self._round_robin.append(key)

    async def connect_all(self, credentials: Dict[Tuple[str, str], ExchangeCredentials]) -> None:
        for key, account in self._accounts.items():
            await account.adapter.connect()
            creds = credentials.get(key)
            if creds:
                await account.adapter.authenticate(creds)

    async def stream_market_data(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[str, str, MarketPayload], Awaitable[None]],
    ) -> List[WebSocketSubscription]:
        tasks: List[WebSocketSubscription] = []
        for (exchange, account), managed in self._accounts.items():
            async def _wrap(event: MarketPayload, ex=exchange, acc=account) -> None:
                await callback(ex, acc, event)

            tasks.append(managed.adapter.stream_market_data(subscriptions, _wrap))
        return tasks

    async def dispatch_order(self, order: OrderRequest) -> OrderStatus:
        async with self._lock:
            if not self._round_robin:
                raise RuntimeError("Brak zarejestrowanych kont giełdowych")
            key = self._round_robin[0]
            self._round_robin.rotate(-1)
            managed = self._accounts[key]
        status = await managed.adapter.submit_order(order)
        self._order_index[status.order_id] = key
        return status

    async def fetch_order_status(self, order_id: str) -> OrderStatus:
        key = self._order_index.get(order_id)
        if not key:
            raise KeyError(f"Nieznane zlecenie {order_id}")
        managed = self._accounts[key]
        return await managed.adapter.fetch_order_status(order_id)

    async def cancel_order(self, order_id: str) -> OrderStatus:
        key = self._order_index.get(order_id)
        if not key:
            raise KeyError(f"Nieznane zlecenie {order_id}")
        managed = self._accounts[key]
        status = await managed.adapter.cancel_order(order_id)
        self._order_index.pop(order_id, None)
        return status

    async def monitor_open_orders(
        self,
        *,
        poll_interval: float = 1.0,
        timeout: float = 60.0,
    ) -> Dict[str, OrderStatus]:
        results: Dict[str, OrderStatus] = {}
        tasks = []
        for order_id, key in list(self._order_index.items()):
            adapter = self._accounts[key].adapter
            tasks.append(
                asyncio.create_task(
                    adapter.monitor_order(order_id, poll_interval=poll_interval, timeout=timeout)
                )
            )
        for task in asyncio.as_completed(tasks):
            status = await task
            results[status.order_id] = status
            if status.status.upper() in {"FILLED", "CANCELED", "REJECTED"}:
                self._order_index.pop(status.order_id, None)
        return results

    def list_accounts(self) -> List[ManagedAccount]:
        return list(self._accounts.values())


__all__ = ["MultiExchangeAccountManager", "ManagedAccount"]
