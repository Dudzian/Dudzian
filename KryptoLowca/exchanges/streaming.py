"""Long-poll based streaming helpers used by legacy exchange adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from .interfaces import MarketPayload, MarketStreamHandle, MarketSubscription
from .polling import MarketDataPoller


CallbackT = Callable[[MarketPayload], Awaitable[None]]
Symbol = str


def _extract_interval(params: Mapping[str, Any] | None, default: float) -> float:
    if not params:
        return default
    for key in ("poll_interval", "interval", "polling_interval"):
        value = params.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return default


@dataclass(slots=True)
class _SymbolMetadata:
    channels: List[str]
    params: Mapping[str, Any]
    interval: float


class LongPollSubscription:
    """Uchwyt strumienia bazujący na :class:`MarketDataPoller`.

    Subskrypcja deduplikuje symbole dla wszystkich kanałów i cyklicznie odpyta
    adapter metodą ``fetch_market_data``. Każda paczka przekazywana do
    ``callback`` zawiera nazwę symbolu oraz metadane kanału, co pozwala
    zachować kompatybilność z konsumentami poprzedniego interfejsu.
    """

    def __init__(
        self,
        adapter: Any,
        subscriptions: Iterable[MarketSubscription],
        callback: CallbackT,
        *,
        default_interval: float = 1.0,
    ) -> None:
        self._adapter = adapter
        self._subscriptions = list(subscriptions)
        if not self._subscriptions:
            raise ValueError("LongPollSubscription wymaga co najmniej jednej subskrypcji")
        self._callback = callback
        self._metadata: Dict[Symbol, _SymbolMetadata] = {}
        self._interval = float(default_interval) if default_interval > 0 else 1.0
        self._poller: Optional[MarketDataPoller] = None
        self._build_symbol_metadata()

    async def __aenter__(self) -> "LongPollSubscription":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        await self.stop()
        return None

    async def start(self) -> None:
        if self._poller is not None:
            return
        symbols = list(self._metadata.keys())
        self._poller = MarketDataPoller(
            adapter=self._adapter,
            symbols=symbols,
            interval=self._interval,
            callback=self._dispatch,
        )
        await self._poller.start()

    async def stop(self) -> None:
        if self._poller is None:
            return
        await self._poller.stop()
        self._poller = None

    def _build_symbol_metadata(self) -> None:
        for subscription in self._subscriptions:
            symbols = [symbol for symbol in subscription.symbols if symbol]
            if not symbols:
                raise ValueError(
                    "Subskrypcje long-poll wymagają jawnej listy symboli (kanał %s)"
                    % subscription.channel
                )
            interval = _extract_interval(subscription.params, self._interval)
            self._interval = min(self._interval, interval)
            for symbol in symbols:
                metadata = self._metadata.setdefault(
                    symbol,
                    _SymbolMetadata(channels=[], params=subscription.params, interval=interval),
                )
                if subscription.channel not in metadata.channels:
                    metadata.channels.append(subscription.channel)
                metadata.interval = min(metadata.interval, interval)

    async def _dispatch(self, symbol: Symbol, payload: MarketPayload) -> None:
        metadata = self._metadata.get(symbol)
        enriched: Dict[str, Any] = dict(payload)
        enriched.setdefault("symbol", symbol)
        if metadata is not None:
            enriched.setdefault("_channels", tuple(metadata.channels))
            if metadata.params:
                enriched.setdefault("_subscription_params", dict(metadata.params))
        result = self._callback(enriched)
        if asyncio.iscoroutine(result):
            await result


if TYPE_CHECKING:
    def _ensure_protocol(subscription: LongPollSubscription) -> MarketStreamHandle:
        return subscription


__all__ = ["LongPollSubscription"]
