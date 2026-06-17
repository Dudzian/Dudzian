"""Deterministic local paper order event spine for preview mode.

The spine is in-memory only. It does not talk to exchanges, read accounts,
read secrets, start workers, or export telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    PreviewModePolicy,
    RuntimeCapability,
    build_preview_mode_policy,
)


class PaperOrderEventType(StrEnum):
    """Paper order lifecycle event names emitted by the local spine."""

    ORDER_INTENT_RECEIVED = "order_intent_received"
    PAPER_ORDER_ACCEPTED = "paper_order_accepted"
    PAPER_ORDER_REJECTED = "paper_order_rejected"
    PAPER_ORDER_PARTIALLY_FILLED = "paper_order_partially_filled"
    PAPER_ORDER_FILLED = "paper_order_filled"
    PAPER_ORDER_CANCELLED = "paper_order_cancelled"


class PaperOrderStatus(StrEnum):
    """Small local paper order states."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


TERMINAL_PAPER_ORDER_STATUSES = frozenset(
    {
        PaperOrderStatus.REJECTED,
        PaperOrderStatus.FILLED,
        PaperOrderStatus.CANCELLED,
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


class PaperOrderTransitionError(ValueError):
    """Raised when a deterministic paper order transition is invalid."""


@dataclass(frozen=True, slots=True)
class PaperOrderEvent:
    """One immutable event in the local paper order lifecycle journal."""

    event_id: str
    order_id: str
    client_order_id: str | None
    event_type: PaperOrderEventType
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    remaining_quantity: float
    status: PaperOrderStatus
    reason: str | None
    sequence: int
    created_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PaperOrderSnapshot:
    """Current local state for one paper order."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    remaining_quantity: float
    status: PaperOrderStatus
    metadata: Mapping[str, object] = field(default_factory=dict)


class PaperEventSpine:
    """In-memory deterministic paper lifecycle reducer/journal."""

    def __init__(
        self,
        *,
        mode: str | PreviewMode = PreviewMode.PAPER,
        capabilities: tuple[str | RuntimeCapability, ...] = (
            RuntimeCapability.PAPER_ORDER_SUBMIT,
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        ),
        created_at: str | None = None,
    ) -> None:
        self.policy = build_preview_mode_policy(mode, capabilities)
        if self.policy.mode is not PreviewMode.PAPER:
            raise PreviewModeContractError("PaperEventSpine requires preview mode 'paper'")
        required = {
            RuntimeCapability.PAPER_ORDER_SUBMIT,
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        }
        if not required.issubset(set(self.policy.capabilities)):
            raise PreviewModeContractError(
                "PaperEventSpine requires paper_order_submit and paper_order_lifecycle"
            )
        self._created_at = created_at
        self._sequence = 0
        self._events: list[PaperOrderEvent] = []
        self._orders: dict[str, PaperOrderSnapshot] = {}

    def submit_order(
        self,
        *,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PaperOrderEvent:
        """Accept a local paper order intent and emit the accepted event."""

        self._validate_new_order(order_id=order_id, symbol=symbol, side=side, quantity=quantity)
        safe_metadata = self._safe_metadata(metadata)
        return self._append_event(
            order_id=order_id,
            client_order_id=client_order_id,
            event_type=PaperOrderEventType.PAPER_ORDER_ACCEPTED,
            symbol=symbol,
            side=side,
            quantity=quantity,
            filled_quantity=0.0,
            remaining_quantity=quantity,
            status=PaperOrderStatus.ACCEPTED,
            reason=None,
            metadata=safe_metadata,
        )

    def reject_order(self, order_id: str, *, reason: str) -> PaperOrderEvent:
        snapshot = self._active_order(order_id)
        return self._append_event(
            order_id=order_id,
            client_order_id=snapshot.client_order_id,
            event_type=PaperOrderEventType.PAPER_ORDER_REJECTED,
            symbol=snapshot.symbol,
            side=snapshot.side,
            quantity=snapshot.quantity,
            filled_quantity=snapshot.filled_quantity,
            remaining_quantity=snapshot.remaining_quantity,
            status=PaperOrderStatus.REJECTED,
            reason=reason,
            metadata=snapshot.metadata,
        )

    def partial_fill(self, order_id: str, *, fill_quantity: float) -> PaperOrderEvent:
        snapshot = self._active_order(order_id)
        self._validate_fill_quantity(fill_quantity, snapshot=snapshot, allow_full=False)
        filled = snapshot.filled_quantity + fill_quantity
        remaining = snapshot.quantity - filled
        return self._append_event(
            order_id=order_id,
            client_order_id=snapshot.client_order_id,
            event_type=PaperOrderEventType.PAPER_ORDER_PARTIALLY_FILLED,
            symbol=snapshot.symbol,
            side=snapshot.side,
            quantity=snapshot.quantity,
            filled_quantity=filled,
            remaining_quantity=remaining,
            status=PaperOrderStatus.PARTIALLY_FILLED,
            reason=None,
            metadata=snapshot.metadata,
        )

    def fill_order(self, order_id: str, *, fill_quantity: float | None = None) -> PaperOrderEvent:
        snapshot = self._active_order(order_id)
        quantity = snapshot.remaining_quantity if fill_quantity is None else fill_quantity
        self._validate_fill_quantity(quantity, snapshot=snapshot, allow_full=True)
        filled = snapshot.filled_quantity + quantity
        remaining = snapshot.quantity - filled
        status = PaperOrderStatus.FILLED if remaining == 0 else PaperOrderStatus.PARTIALLY_FILLED
        event_type = (
            PaperOrderEventType.PAPER_ORDER_FILLED
            if status is PaperOrderStatus.FILLED
            else PaperOrderEventType.PAPER_ORDER_PARTIALLY_FILLED
        )
        return self._append_event(
            order_id=order_id,
            client_order_id=snapshot.client_order_id,
            event_type=event_type,
            symbol=snapshot.symbol,
            side=snapshot.side,
            quantity=snapshot.quantity,
            filled_quantity=filled,
            remaining_quantity=remaining,
            status=status,
            reason=None,
            metadata=snapshot.metadata,
        )

    def cancel_order(self, order_id: str, *, reason: str | None = None) -> PaperOrderEvent:
        snapshot = self._active_order(order_id)
        return self._append_event(
            order_id=order_id,
            client_order_id=snapshot.client_order_id,
            event_type=PaperOrderEventType.PAPER_ORDER_CANCELLED,
            symbol=snapshot.symbol,
            side=snapshot.side,
            quantity=snapshot.quantity,
            filled_quantity=snapshot.filled_quantity,
            remaining_quantity=snapshot.remaining_quantity,
            status=PaperOrderStatus.CANCELLED,
            reason=reason,
            metadata=snapshot.metadata,
        )

    def events_for_order(self, order_id: str) -> tuple[PaperOrderEvent, ...]:
        return tuple(event for event in self._events if event.order_id == order_id)

    def all_events(self) -> tuple[PaperOrderEvent, ...]:
        return tuple(self._events)

    def order_state(self, order_id: str) -> PaperOrderSnapshot:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise PaperOrderTransitionError(f"Unknown paper order: {order_id}") from exc

    def _append_event(
        self,
        *,
        order_id: str,
        client_order_id: str | None,
        event_type: PaperOrderEventType,
        symbol: str,
        side: str,
        quantity: float,
        filled_quantity: float,
        remaining_quantity: float,
        status: PaperOrderStatus,
        reason: str | None,
        metadata: Mapping[str, object],
    ) -> PaperOrderEvent:
        self._sequence += 1
        event = PaperOrderEvent(
            event_id=f"paper-{self._sequence:06d}",
            order_id=order_id,
            client_order_id=client_order_id,
            event_type=event_type,
            symbol=symbol,
            side=side,
            quantity=quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            status=status,
            reason=reason,
            sequence=self._sequence,
            created_at=self._created_at,
            metadata=metadata,
        )
        self._events.append(event)
        self._orders[order_id] = PaperOrderSnapshot(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            status=status,
            metadata=metadata,
        )
        return event

    def _validate_new_order(
        self, *, order_id: str, symbol: str, side: str, quantity: float
    ) -> None:
        if not order_id.strip():
            raise PaperOrderTransitionError("order_id is required")
        if order_id in self._orders:
            raise PaperOrderTransitionError(f"Paper order already exists: {order_id}")
        if not symbol.strip():
            raise PaperOrderTransitionError("symbol is required")
        if not side.strip():
            raise PaperOrderTransitionError("side is required")
        if quantity <= 0:
            raise PaperOrderTransitionError("quantity must be > 0")

    def _active_order(self, order_id: str) -> PaperOrderSnapshot:
        snapshot = self.order_state(order_id)
        if snapshot.status in TERMINAL_PAPER_ORDER_STATUSES:
            raise PaperOrderTransitionError(
                f"Paper order {order_id} is terminal: {snapshot.status.value}"
            )
        return snapshot

    @staticmethod
    def _validate_fill_quantity(
        fill_quantity: float, *, snapshot: PaperOrderSnapshot, allow_full: bool
    ) -> None:
        if fill_quantity <= 0:
            raise PaperOrderTransitionError("fill_quantity must be > 0")
        if fill_quantity > snapshot.remaining_quantity:
            raise PaperOrderTransitionError("fill_quantity exceeds remaining quantity")
        if not allow_full and fill_quantity == snapshot.remaining_quantity:
            raise PaperOrderTransitionError("partial fill must leave remaining quantity")

    @staticmethod
    def _safe_metadata(metadata: Mapping[str, object] | None) -> Mapping[str, object]:
        if metadata is None:
            return MappingProxyType({})
        unsafe = [key for key in metadata if _metadata_key_is_secret(key)]
        if unsafe:
            raise PaperOrderTransitionError(
                f"metadata contains forbidden credential-like keys: {', '.join(sorted(unsafe))}"
            )
        return MappingProxyType(dict(metadata))


def _metadata_key_is_secret(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _SECRET_METADATA_TOKENS)


__all__ = [
    "PaperEventSpine",
    "PaperOrderEvent",
    "PaperOrderEventType",
    "PaperOrderSnapshot",
    "PaperOrderStatus",
    "PaperOrderTransitionError",
    "PreviewModePolicy",
    "TERMINAL_PAPER_ORDER_STATUSES",
]
