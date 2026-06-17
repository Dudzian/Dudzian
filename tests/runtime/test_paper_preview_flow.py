"""Unit tests for the local deterministic paper preview composition flow."""

from __future__ import annotations

import builtins
import os
import socket

import pytest

from bot_core.runtime.paper_audit_journal import PaperAuditSeverity
from bot_core.runtime.paper_event_spine import PaperOrderTransitionError
from bot_core.runtime.paper_preview_flow import PaperPreviewFlow
from bot_core.runtime.preview_modes import (
    PreviewMode,
    RuntimeCapability,
    is_capability_allowed_in_preview,
)


def _audit_types(flow: PaperPreviewFlow) -> list[str]:
    return [event.event_type for event in flow.all_audit_events()]


def test_composition_uses_preview_mode_contract() -> None:
    flow = PaperPreviewFlow()

    assert flow.policy.mode is PreviewMode.PAPER
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in flow.policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in flow.policy.capabilities
    assert RuntimeCapability.LOCAL_TELEMETRY_AUDIT in flow.policy.capabilities
    assert is_capability_allowed_in_preview(RuntimeCapability.PAPER_ORDER_SUBMIT)
    assert is_capability_allowed_in_preview(RuntimeCapability.PAPER_ORDER_LIFECYCLE)
    assert is_capability_allowed_in_preview(RuntimeCapability.LOCAL_TELEMETRY_AUDIT)

    for blocked in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        assert not is_capability_allowed_in_preview(blocked)


def test_happy_path_full_fill_updates_portfolio_and_audit_deterministically() -> None:
    flow = PaperPreviewFlow(created_at="2026-06-17T00:00:00Z")

    submit = flow.submit_order(order_id="buy-1", symbol="BTC/USDT", side="buy", quantity=2.0)
    filled = flow.fill_order("buy-1", fill_price=100.0)
    snapshot = flow.snapshot()

    assert submit.order_event.event_id == "paper-000001"
    assert submit.audit_events[0].event_type == "order_accepted"
    assert filled.order_event.event_id == "paper-000002"
    assert filled.order_event.sequence == 2
    assert len(snapshot.portfolio.trades) == 1
    assert snapshot.portfolio.trades[0].trade_id == "trade-000002"
    assert snapshot.portfolio.positions[0].quantity == 2.0
    assert _audit_types(flow) == ["order_accepted", "order_filled", "trade_recorded"]
    assert [event.audit_id for event in flow.all_audit_events()] == [
        "audit-000001",
        "audit-000002",
        "audit-000003",
    ]


def test_partial_fill_then_final_fill_updates_weighted_average_and_audit_order() -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(order_id="buy-2", symbol="BTC/USDT", side="buy", quantity=2.0)

    partial = flow.partial_fill_order("buy-2", fill_quantity=1.0, fill_price=100.0)
    final = flow.fill_order("buy-2", fill_price=120.0)
    snapshot = flow.snapshot()

    assert partial.order_event.event_id == "paper-000002"
    assert final.order_event.event_id == "paper-000003"
    assert [trade.trade_id for trade in snapshot.portfolio.trades] == [
        "trade-000002",
        "trade-000003",
    ]
    assert snapshot.portfolio.positions[0].avg_entry_price == pytest.approx(110.0)
    assert _audit_types(flow) == [
        "order_accepted",
        "order_partially_filled",
        "trade_recorded",
        "order_filled",
        "trade_recorded",
    ]


def test_sell_realized_pnl_flow_is_audited() -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(order_id="buy-3", symbol="BTC/USDT", side="buy", quantity=1.0)
    flow.fill_order("buy-3", fill_price=100.0)
    flow.submit_order(order_id="sell-3", symbol="BTC/USDT", side="sell", quantity=1.0)

    flow.fill_order("sell-3", fill_price=130.0)
    snapshot = flow.snapshot()

    assert snapshot.portfolio.positions[0].realized_pnl == pytest.approx(30.0)
    assert snapshot.portfolio.trades[-1].realized_pnl == pytest.approx(30.0)
    assert _audit_types(flow)[-2:] == ["trade_recorded", "realized_pnl_recorded"]


def test_reject_path_creates_warning_audit_without_trade_or_portfolio_change() -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(order_id="reject-1", symbol="ETH/USDT", side="buy", quantity=1.0)

    flow.reject_order("reject-1", reason="risk_block")
    snapshot = flow.snapshot()

    assert snapshot.portfolio.trades == ()
    assert snapshot.portfolio.positions == ()
    rejected = flow.all_audit_events()[-1]
    assert rejected.event_type == "order_rejected"
    assert rejected.severity is PaperAuditSeverity.WARNING
    assert rejected.metadata["reason"] == "risk_block"


def test_cancel_path_creates_audit_without_trade_or_portfolio_change() -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(order_id="cancel-1", symbol="SOL/USDT", side="buy", quantity=1.0)

    flow.cancel_order("cancel-1", reason="user_cancel")
    snapshot = flow.snapshot()

    assert snapshot.portfolio.trades == ()
    assert snapshot.portfolio.positions == ()
    assert flow.all_audit_events()[-1].event_type == "order_cancelled"


@pytest.mark.parametrize("fill_price", [0.0, -1.0])
def test_invalid_fill_price_does_not_emit_fill_trade_or_audit(fill_price: float) -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(
        order_id=f"bad-price-{fill_price}", symbol="BTC/USDT", side="buy", quantity=1.0
    )
    before_events = flow.all_order_events()
    before_audit = flow.all_audit_events()

    with pytest.raises(ValueError, match="fill_price must be > 0"):
        flow.fill_order(f"bad-price-{fill_price}", fill_price=fill_price)

    assert flow.all_order_events() == before_events
    assert flow.portfolio_snapshot().trades == ()
    assert flow.all_audit_events() == before_audit


def test_terminal_transition_raises_without_extra_audit_or_trade() -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(order_id="terminal-1", symbol="BTC/USDT", side="buy", quantity=1.0)
    flow.fill_order("terminal-1", fill_price=100.0)
    before_audit = flow.all_audit_events()
    before_trades = flow.portfolio_snapshot().trades

    with pytest.raises(PaperOrderTransitionError, match="terminal"):
        flow.cancel_order("terminal-1")

    assert flow.all_audit_events() == before_audit
    assert flow.portfolio_snapshot().trades == before_trades


def test_local_only_no_network_env_file_write_or_runtime_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not be used")

    def guarded_open(file: object, mode: str = "r", *_args: object, **_kwargs: object):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("file writes must not be used")
        return original_open(file, mode, *_args, **_kwargs)

    original_open = builtins.open
    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(
        os,
        "getenv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("env must not be read")),
    )
    monkeypatch.setattr(builtins, "open", guarded_open)

    flow = PaperPreviewFlow()
    flow.submit_order(order_id="safe", symbol="BTC/USDT", side="buy", quantity=1.0)
    flow.fill_order("safe", fill_price=100.0)

    assert flow.snapshot().portfolio.positions[0].quantity == 1.0


def test_snapshot_is_immutable_ordered_and_does_not_duplicate_trade_audits() -> None:
    flow = PaperPreviewFlow()
    flow.submit_order(order_id="buy-4", symbol="BTC/USDT", side="buy", quantity=2.0)
    flow.partial_fill_order("buy-4", fill_quantity=1.0, fill_price=100.0)
    flow.fill_order("buy-4", fill_price=120.0)

    first = flow.snapshot()
    second = flow.snapshot()

    assert isinstance(first.order_events, tuple)
    assert isinstance(first.portfolio.trades, tuple)
    assert isinstance(first.audit_events, tuple)
    assert first.audit_events == second.audit_events
    assert [event.sequence for event in first.order_events] == [1, 2, 3]
    assert [trade.sequence for trade in first.portfolio.trades] == [2, 3]
    assert [event.sequence for event in first.audit_events] == [1, 2, 3, 4, 5]
