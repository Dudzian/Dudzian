"""Deterministic local paper portfolio reducer for preview mode.

The reducer is in-memory only. It consumes ``PaperOrderEvent`` objects from the
local paper event spine and never talks to exchanges, reads accounts, reads
secrets, starts workers, or exports telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Iterable, Mapping

from bot_core.runtime.paper_event_spine import PaperOrderEvent, PaperOrderEventType
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    PreviewModePolicy,
    RuntimeCapability,
    build_preview_mode_policy,
)

_FILL_EVENT_TYPES = frozenset(
    {
        PaperOrderEventType.PAPER_ORDER_PARTIALLY_FILLED,
        PaperOrderEventType.PAPER_ORDER_FILLED,
    }
)
_NON_MUTATING_EVENT_TYPES = frozenset(
    {
        PaperOrderEventType.ORDER_INTENT_RECEIVED,
        PaperOrderEventType.PAPER_ORDER_ACCEPTED,
        PaperOrderEventType.PAPER_ORDER_REJECTED,
        PaperOrderEventType.PAPER_ORDER_CANCELLED,
    }
)
_SECRET_METADATA_TOKENS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passphrase",
    "credential",
    "credentials",
    "token",
    "private_key",
)
_SAFE_TRADE_METADATA_KEYS = frozenset({"strategy", "source", "tag", "note", "scenario"})


class PaperPortfolioReducerError(ValueError):
    """Raised when a paper portfolio reduction is invalid."""


@dataclass(frozen=True, slots=True)
class PaperTrade:
    """One deterministic local trade derived from a paper fill event."""

    trade_id: str
    event_id: str
    order_id: str
    sequence: int
    symbol: str
    side: str
    quantity: float
    price: float
    realized_pnl: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """Current deterministic local long/flat paper position."""

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class PaperPortfolioSnapshot:
    """Immutable deterministic paper portfolio snapshot."""

    positions: tuple[PaperPosition, ...]
    trades: tuple[PaperTrade, ...]
    applied_event_ids: tuple[str, ...]


class PaperPortfolioReducer:
    """In-memory long/flat portfolio reducer for local paper fill events."""

    def __init__(
        self,
        *,
        mode: str | PreviewMode = PreviewMode.PAPER,
        capabilities: tuple[str | RuntimeCapability, ...] = (
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        ),
    ) -> None:
        self.policy = build_preview_mode_policy(mode, capabilities)
        if self.policy.mode is not PreviewMode.PAPER:
            raise PreviewModeContractError("PaperPortfolioReducer requires preview mode 'paper'")
        if RuntimeCapability.PAPER_ORDER_LIFECYCLE not in self.policy.capabilities:
            raise PreviewModeContractError(
                "PaperPortfolioReducer requires paper_order_lifecycle capability"
            )
        self._positions: dict[str, PaperPosition] = {}
        self._trades: list[PaperTrade] = []
        self._applied_event_ids: set[str] = set()
        self._applied_events: list[tuple[int, str]] = []
        self._order_filled_quantities: dict[str, float] = {}

    def apply_events(
        self, events_with_prices: Iterable[tuple[PaperOrderEvent, float | None]]
    ) -> PaperPortfolioSnapshot:
        """Apply paper events with explicit deterministic fill prices."""

        for event, fill_price in events_with_prices:
            self.apply_event(event, fill_price=fill_price)
        return self.snapshot()

    def apply_event(
        self, event: PaperOrderEvent, *, fill_price: float | None = None
    ) -> PaperPortfolioSnapshot:
        """Apply one paper event and return a deterministic snapshot."""

        if event.event_id in self._applied_event_ids:
            raise PaperPortfolioReducerError(f"Paper event already applied: {event.event_id}")
        event_type = self._coerce_event_type(event.event_type)

        if event_type in _NON_MUTATING_EVENT_TYPES:
            self._mark_applied(event)
            self._order_filled_quantities.setdefault(event.order_id, event.filled_quantity)
            return self.snapshot()
        if event_type not in _FILL_EVENT_TYPES:
            raise PaperPortfolioReducerError(
                f"Unsupported paper order event type: {event.event_type!r}"
            )

        price = self._resolve_fill_price(event, fill_price)
        fill_quantity = self._newly_filled_quantity(event)
        if fill_quantity <= 0:
            raise PaperPortfolioReducerError("fill event must increase filled_quantity")
        side = event.side.strip().lower()
        if side == "buy":
            next_position, realized_pnl = self._prepare_buy(event.symbol, fill_quantity, price)
        elif side == "sell":
            next_position, realized_pnl = self._prepare_sell(event.symbol, fill_quantity, price)
        else:
            raise PaperPortfolioReducerError(f"Unsupported paper order side: {event.side!r}")

        self._mark_applied(event)
        self._positions[event.symbol] = next_position
        self._order_filled_quantities[event.order_id] = event.filled_quantity
        self._trades.append(
            PaperTrade(
                trade_id=f"trade-{event.sequence:06d}",
                event_id=event.event_id,
                order_id=event.order_id,
                sequence=event.sequence,
                symbol=event.symbol,
                side=side,
                quantity=fill_quantity,
                price=price,
                realized_pnl=realized_pnl,
                metadata=self._safe_trade_metadata(event.metadata),
            )
        )
        return self.snapshot()

    def snapshot(self) -> PaperPortfolioSnapshot:
        """Return a deterministic immutable snapshot."""

        return PaperPortfolioSnapshot(
            positions=tuple(self._positions[symbol] for symbol in sorted(self._positions)),
            trades=tuple(sorted(self._trades, key=lambda trade: (trade.sequence, trade.event_id))),
            applied_event_ids=tuple(
                event_id for _, event_id in sorted(self._applied_events, key=lambda item: item)
            ),
        )

    @staticmethod
    def _coerce_event_type(event_type: object) -> PaperOrderEventType:
        if isinstance(event_type, PaperOrderEventType):
            return event_type
        try:
            return PaperOrderEventType(str(event_type))
        except ValueError as exc:
            raise PaperPortfolioReducerError(
                f"Unsupported paper order event type: {event_type!r}"
            ) from exc

    @staticmethod
    def _resolve_fill_price(event: PaperOrderEvent, fill_price: float | None) -> float:
        price = fill_price
        if price is None:
            candidate = event.metadata.get("price")
            if isinstance(candidate, int | float):
                price = float(candidate)
        if price is None or price <= 0:
            raise PaperPortfolioReducerError("fill_price must be > 0")
        return float(price)

    def _newly_filled_quantity(self, event: PaperOrderEvent) -> float:
        previous = self._order_filled_quantities.get(event.order_id, 0.0)
        return event.filled_quantity - previous

    def _mark_applied(self, event: PaperOrderEvent) -> None:
        self._applied_event_ids.add(event.event_id)
        self._applied_events.append((event.sequence, event.event_id))

    def _prepare_buy(
        self, symbol: str, quantity: float, price: float
    ) -> tuple[PaperPosition, float]:
        position = self._positions.get(symbol, PaperPosition(symbol=symbol))
        total_quantity = position.quantity + quantity
        avg_entry_price = (
            (position.quantity * position.avg_entry_price) + (quantity * price)
        ) / total_quantity
        return (
            replace(
                position,
                quantity=total_quantity,
                avg_entry_price=avg_entry_price,
            ),
            0.0,
        )

    def _prepare_sell(
        self, symbol: str, quantity: float, price: float
    ) -> tuple[PaperPosition, float]:
        position = self._positions.get(symbol, PaperPosition(symbol=symbol))
        if quantity > position.quantity:
            raise PaperPortfolioReducerError("sell quantity exceeds current long position")
        realized_pnl = quantity * (price - position.avg_entry_price)
        remaining_quantity = position.quantity - quantity
        avg_entry_price = 0.0 if remaining_quantity == 0 else position.avg_entry_price
        return (
            replace(
                position,
                quantity=remaining_quantity,
                avg_entry_price=avg_entry_price,
                realized_pnl=position.realized_pnl + realized_pnl,
            ),
            realized_pnl,
        )

    @staticmethod
    def _safe_trade_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
        safe = {
            str(key): value
            for key, value in metadata.items()
            if str(key) in _SAFE_TRADE_METADATA_KEYS and not _metadata_key_is_secret(key)
        }
        return MappingProxyType(safe)


def _metadata_key_is_secret(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _SECRET_METADATA_TOKENS)


__all__ = [
    "PaperPortfolioReducer",
    "PaperPortfolioReducerError",
    "PaperPortfolioSnapshot",
    "PaperPosition",
    "PaperTrade",
    "PreviewModePolicy",
]
