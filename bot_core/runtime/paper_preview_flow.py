"""Local deterministic paper preview composition flow.

This module wires the in-memory paper event spine, portfolio reducer, and audit
journal for unit-level preview proof only. It does not start runtime loops, talk
to exchanges, fetch accounts, read secrets, open sockets, or export telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from bot_core.runtime.paper_audit_journal import PaperAuditEvent, PaperAuditJournal
from bot_core.runtime.paper_event_spine import PaperEventSpine, PaperOrderEvent
from bot_core.runtime.paper_portfolio_reducer import (
    PaperPortfolioReducer,
    PaperPortfolioSnapshot,
)
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    PreviewModePolicy,
    RuntimeCapability,
    build_preview_mode_policy,
)

_REQUIRED_CAPABILITIES = (
    RuntimeCapability.PAPER_ORDER_SUBMIT,
    RuntimeCapability.PAPER_ORDER_LIFECYCLE,
    RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
)


@dataclass(frozen=True, slots=True)
class PaperPreviewFlowResult:
    """Result of one local paper preview composition operation."""

    order_event: PaperOrderEvent
    portfolio_snapshot: PaperPortfolioSnapshot
    audit_events: tuple[PaperAuditEvent, ...]


@dataclass(frozen=True, slots=True)
class PaperPreviewSnapshot:
    """Immutable snapshot of the composed local paper preview state."""

    order_events: tuple[PaperOrderEvent, ...]
    portfolio: PaperPortfolioSnapshot
    audit_events: tuple[PaperAuditEvent, ...]


class PaperPreviewFlow:
    """Small in-memory composition service for paper preview unit evidence."""

    def __init__(self, *, created_at: str | None = None) -> None:
        self.policy = build_preview_mode_policy(PreviewMode.PAPER, _REQUIRED_CAPABILITIES)
        if set(self.policy.capabilities) != set(_REQUIRED_CAPABILITIES):
            raise PreviewModeContractError("PaperPreviewFlow requires paper preview capabilities")
        self.spine = PaperEventSpine(created_at=created_at)
        self.reducer = PaperPortfolioReducer()
        self.audit_journal = PaperAuditJournal(created_at=created_at)
        self._audited_trade_ids: set[str] = set()

    def submit_order(
        self,
        *,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PaperPreviewFlowResult:
        """Submit one local paper order and audit the accepted event."""

        event = self.spine.submit_order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            client_order_id=client_order_id,
            metadata=metadata,
        )
        audit_event = self.audit_journal.record_order_event(event)
        return self._result(event, (audit_event,))

    def reject_order(self, order_id: str, *, reason: str) -> PaperPreviewFlowResult:
        """Reject an active local paper order and audit the terminal event."""

        event = self.spine.reject_order(order_id, reason=reason)
        audit_event = self.audit_journal.record_order_event(event)
        return self._result(event, (audit_event,))

    def cancel_order(self, order_id: str, *, reason: str | None = None) -> PaperPreviewFlowResult:
        """Cancel an active local paper order and audit the terminal event."""

        event = self.spine.cancel_order(order_id, reason=reason)
        audit_event = self.audit_journal.record_order_event(event)
        return self._result(event, (audit_event,))

    def partial_fill_order(
        self, order_id: str, *, fill_quantity: float, fill_price: float
    ) -> PaperPreviewFlowResult:
        """Partially fill an order, then apply portfolio and audit changes."""

        self._validate_fill_price(fill_price)
        event = self.spine.partial_fill(order_id, fill_quantity=fill_quantity)
        return self._apply_fill_event(event, fill_price=fill_price)

    def fill_order(
        self,
        order_id: str,
        *,
        fill_price: float,
        fill_quantity: float | None = None,
    ) -> PaperPreviewFlowResult:
        """Fill an order, then apply portfolio and audit changes."""

        self._validate_fill_price(fill_price)
        event = self.spine.fill_order(order_id, fill_quantity=fill_quantity)
        return self._apply_fill_event(event, fill_price=fill_price)

    def snapshot(self) -> PaperPreviewSnapshot:
        """Return deterministic immutable composed state."""

        return PaperPreviewSnapshot(
            order_events=self.all_order_events(),
            portfolio=self.portfolio_snapshot(),
            audit_events=self.all_audit_events(),
        )

    def all_order_events(self) -> tuple[PaperOrderEvent, ...]:
        """Return order events in deterministic spine sequence order."""

        return self.spine.all_events()

    def portfolio_snapshot(self) -> PaperPortfolioSnapshot:
        """Return reducer snapshot with deterministic trades and positions."""

        return self.reducer.snapshot()

    def all_audit_events(self) -> tuple[PaperAuditEvent, ...]:
        """Return audit events in deterministic audit sequence order."""

        return self.audit_journal.all_events()

    def _apply_fill_event(
        self, event: PaperOrderEvent, *, fill_price: float
    ) -> PaperPreviewFlowResult:
        before_trade_ids = {trade.trade_id for trade in self.reducer.snapshot().trades}
        portfolio = self.reducer.apply_event(event, fill_price=fill_price)
        new_trades = tuple(
            trade
            for trade in portfolio.trades
            if trade.trade_id not in before_trade_ids
            and trade.trade_id not in self._audited_trade_ids
        )
        order_audit = self.audit_journal.record_order_event(event)
        trade_audits: list[PaperAuditEvent] = []
        for trade in new_trades:
            trade_audits.extend(self.audit_journal.record_trade(trade))
            self._audited_trade_ids.add(trade.trade_id)
        return PaperPreviewFlowResult(
            order_event=event,
            portfolio_snapshot=portfolio,
            audit_events=(order_audit, *trade_audits),
        )

    def _result(
        self, event: PaperOrderEvent, audit_events: tuple[PaperAuditEvent, ...]
    ) -> PaperPreviewFlowResult:
        return PaperPreviewFlowResult(
            order_event=event,
            portfolio_snapshot=self.portfolio_snapshot(),
            audit_events=audit_events,
        )

    @staticmethod
    def _validate_fill_price(fill_price: float) -> None:
        if fill_price <= 0:
            raise ValueError("fill_price must be > 0")


__all__ = [
    "PaperPreviewFlow",
    "PaperPreviewFlowResult",
    "PaperPreviewSnapshot",
    "PreviewModePolicy",
]
