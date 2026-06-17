"""Deterministic local paper audit journal for preview mode.

The journal is in-memory only. It consumes local paper order/trade events and
never talks to exchanges, reads accounts, reads secrets, starts workers, or
exports telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from bot_core.runtime.paper_event_spine import PaperOrderEvent, PaperOrderEventType
from bot_core.runtime.paper_portfolio_reducer import PaperTrade
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    PreviewModePolicy,
    RuntimeCapability,
    build_preview_mode_policy,
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
_ORDER_EVENT_TYPES: dict[PaperOrderEventType, str] = {
    PaperOrderEventType.PAPER_ORDER_ACCEPTED: "order_accepted",
    PaperOrderEventType.PAPER_ORDER_REJECTED: "order_rejected",
    PaperOrderEventType.PAPER_ORDER_CANCELLED: "order_cancelled",
    PaperOrderEventType.PAPER_ORDER_PARTIALLY_FILLED: "order_partially_filled",
    PaperOrderEventType.PAPER_ORDER_FILLED: "order_filled",
}


class PaperAuditSeverity(StrEnum):
    """Local severity for deterministic paper audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PaperAuditJournalError(ValueError):
    """Raised when a paper audit event cannot be recorded safely."""


@dataclass(frozen=True, slots=True)
class PaperAuditEvent:
    """One immutable local paper audit/alert event."""

    audit_id: str
    sequence: int
    source_event_id: str | None
    event_type: str
    severity: PaperAuditSeverity
    symbol: str | None
    order_id: str | None
    trade_id: str | None
    message: str
    created_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class PaperAuditJournal:
    """In-memory local-only consumer for paper order and trade audit events."""

    def __init__(
        self,
        *,
        mode: str | PreviewMode = PreviewMode.PAPER,
        capabilities: tuple[str | RuntimeCapability, ...] = (
            RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        ),
        created_at: str | None = None,
    ) -> None:
        self.policy = build_preview_mode_policy(mode, capabilities)
        if self.policy.mode is not PreviewMode.PAPER:
            raise PreviewModeContractError("PaperAuditJournal requires preview mode 'paper'")
        required = {
            RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        }
        if not required.issubset(set(self.policy.capabilities)):
            raise PreviewModeContractError(
                "PaperAuditJournal requires local_telemetry_audit and paper_order_lifecycle"
            )
        self._created_at = created_at
        self._sequence = 0
        self._events: list[PaperAuditEvent] = []

    def record_order_event(self, event: PaperOrderEvent) -> PaperAuditEvent:
        """Consume one local paper order event without mutating order state."""

        event_type = self._coerce_order_event_type(event.event_type)
        audit_event_type = _ORDER_EVENT_TYPES.get(event_type)
        if audit_event_type is None:
            raise PaperAuditJournalError(
                f"Unsupported paper order event type: {event.event_type!r}"
            )
        metadata: dict[str, object] = {
            "paper_event_type": event_type.value,
            "status": event.status.value,
            "filled_quantity": event.filled_quantity,
            "remaining_quantity": event.remaining_quantity,
        }
        if event.reason is not None:
            metadata["reason"] = event.reason
        metadata.update(self._safe_metadata(event.metadata))
        severity = (
            PaperAuditSeverity.WARNING
            if event_type is PaperOrderEventType.PAPER_ORDER_REJECTED
            else PaperAuditSeverity.INFO
        )
        return self._append_event(
            source_event_id=event.event_id,
            event_type=audit_event_type,
            severity=severity,
            symbol=event.symbol,
            order_id=event.order_id,
            trade_id=None,
            message=self._order_message(event_type, event),
            metadata=metadata,
        )

    def record_trade(self, trade: PaperTrade) -> tuple[PaperAuditEvent, ...]:
        """Consume one local paper trade and optional realized-PnL audit event."""

        events = [
            self._append_event(
                source_event_id=trade.event_id,
                event_type="trade_recorded",
                severity=PaperAuditSeverity.INFO,
                symbol=trade.symbol,
                order_id=trade.order_id,
                trade_id=trade.trade_id,
                message=f"Paper trade {trade.trade_id} recorded for {trade.symbol}",
                metadata={
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "realized_pnl": trade.realized_pnl,
                    **self._safe_metadata(trade.metadata),
                },
            )
        ]
        if trade.realized_pnl != 0:
            severity = (
                PaperAuditSeverity.WARNING if trade.realized_pnl < 0 else PaperAuditSeverity.INFO
            )
            events.append(
                self._append_event(
                    source_event_id=trade.event_id,
                    event_type="realized_pnl_recorded",
                    severity=severity,
                    symbol=trade.symbol,
                    order_id=trade.order_id,
                    trade_id=trade.trade_id,
                    message=f"Paper trade {trade.trade_id} realized PnL {trade.realized_pnl}",
                    metadata={"realized_pnl": trade.realized_pnl},
                )
            )
        return tuple(events)

    def all_events(self) -> tuple[PaperAuditEvent, ...]:
        """Return an immutable deterministic sequence-ordered audit snapshot."""

        return tuple(sorted(self._events, key=lambda event: event.sequence))

    def events_for_order(self, order_id: str) -> tuple[PaperAuditEvent, ...]:
        return tuple(event for event in self.all_events() if event.order_id == order_id)

    def events_for_trade(self, trade_id: str) -> tuple[PaperAuditEvent, ...]:
        return tuple(event for event in self.all_events() if event.trade_id == trade_id)

    def events_for_source(self, source_event_id: str) -> tuple[PaperAuditEvent, ...]:
        return tuple(
            event for event in self.all_events() if event.source_event_id == source_event_id
        )

    def _append_event(
        self,
        *,
        source_event_id: str | None,
        event_type: str,
        severity: PaperAuditSeverity,
        symbol: str | None,
        order_id: str | None,
        trade_id: str | None,
        message: str,
        metadata: Mapping[str, object],
    ) -> PaperAuditEvent:
        self._sequence += 1
        event = PaperAuditEvent(
            audit_id=f"audit-{self._sequence:06d}",
            sequence=self._sequence,
            source_event_id=source_event_id,
            event_type=event_type,
            severity=severity,
            symbol=symbol,
            order_id=order_id,
            trade_id=trade_id,
            message=message,
            created_at=self._created_at,
            metadata=MappingProxyType(dict(metadata)),
        )
        self._events.append(event)
        return event

    @staticmethod
    def _coerce_order_event_type(event_type: object) -> PaperOrderEventType:
        if isinstance(event_type, PaperOrderEventType):
            return event_type
        try:
            return PaperOrderEventType(str(event_type))
        except ValueError as exc:
            raise PaperAuditJournalError(
                f"Unsupported paper order event type: {event_type!r}"
            ) from exc

    @staticmethod
    def _order_message(event_type: PaperOrderEventType, event: PaperOrderEvent) -> str:
        if event_type is PaperOrderEventType.PAPER_ORDER_REJECTED:
            return f"Paper order {event.order_id} rejected: {event.reason}"
        if event_type is PaperOrderEventType.PAPER_ORDER_CANCELLED:
            return f"Paper order {event.order_id} cancelled"
        if event_type is PaperOrderEventType.PAPER_ORDER_PARTIALLY_FILLED:
            return f"Paper order {event.order_id} partially filled"
        if event_type is PaperOrderEventType.PAPER_ORDER_FILLED:
            return f"Paper order {event.order_id} filled"
        return f"Paper order {event.order_id} accepted"

    @staticmethod
    def _safe_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
        unsafe = [str(key) for key in metadata if _metadata_key_is_secret(key)]
        if unsafe:
            raise PaperAuditJournalError(
                f"metadata contains forbidden credential-like keys: {', '.join(sorted(unsafe))}"
            )
        return MappingProxyType(dict(metadata))


def _metadata_key_is_secret(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _SECRET_METADATA_TOKENS)


__all__ = [
    "PaperAuditEvent",
    "PaperAuditJournal",
    "PaperAuditJournalError",
    "PaperAuditSeverity",
    "PreviewModePolicy",
]
