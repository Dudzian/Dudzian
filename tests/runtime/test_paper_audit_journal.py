"""Unit tests for the deterministic local paper audit journal."""

from __future__ import annotations

import builtins
import os
import socket
from dataclasses import replace

import pytest

from bot_core.runtime.paper_audit_journal import (
    PaperAuditJournal,
    PaperAuditJournalError,
    PaperAuditSeverity,
)
from bot_core.runtime.paper_event_spine import PaperEventSpine
from bot_core.runtime.paper_portfolio_reducer import PaperPortfolioReducer
from bot_core.runtime.preview_modes import PreviewModeContractError, RuntimeCapability


def _filled_trade(*, realized: str = "positive"):
    spine = PaperEventSpine()
    reducer = PaperPortfolioReducer()
    spine.submit_order(order_id="buy-1", symbol="BTC/USDT", side="buy", quantity=1.0)
    reducer.apply_event(spine.fill_order("buy-1"), fill_price=100.0)
    spine.submit_order(order_id=f"sell-{realized}", symbol="BTC/USDT", side="sell", quantity=1.0)
    sell = spine.fill_order(f"sell-{realized}")
    price = 130.0 if realized == "positive" else 90.0
    snapshot = reducer.apply_event(sell, fill_price=price)
    return reducer, snapshot.trades[-1]


def test_journal_uses_preview_mode_contract() -> None:
    journal = PaperAuditJournal(mode="paper")

    assert RuntimeCapability.LOCAL_TELEMETRY_AUDIT in journal.policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in journal.policy.capabilities

    for blocked in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            PaperAuditJournal(
                capabilities=(
                    RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
                    RuntimeCapability.PAPER_ORDER_LIFECYCLE,
                    blocked,
                )
            )

    with pytest.raises(PreviewModeContractError):
        PaperAuditJournal(capabilities=(RuntimeCapability.LOCAL_TELEMETRY_AUDIT,))


def test_order_lifecycle_audit_events_are_deterministic() -> None:
    spine = PaperEventSpine()
    journal = PaperAuditJournal(created_at="2026-06-17T00:00:00Z")

    accepted = spine.submit_order(
        order_id="ord-1",
        symbol="BTC/USDT",
        side="buy",
        quantity=2.0,
        metadata={"strategy": "unit"},
    )
    partial = spine.partial_fill("ord-1", fill_quantity=1.0)
    filled = spine.fill_order("ord-1")
    spine.submit_order(order_id="ord-2", symbol="ETH/USDT", side="sell", quantity=1.0)
    rejected = spine.reject_order("ord-2", reason="risk_block")
    spine.submit_order(order_id="ord-3", symbol="SOL/USDT", side="buy", quantity=1.0)
    cancelled = spine.cancel_order("ord-3", reason="user_cancel")

    for event in (accepted, rejected, cancelled, partial, filled):
        journal.record_order_event(event)

    events = journal.all_events()
    assert isinstance(events, tuple)
    assert [event.audit_id for event in events] == [
        "audit-000001",
        "audit-000002",
        "audit-000003",
        "audit-000004",
        "audit-000005",
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
    assert [event.event_type for event in events] == [
        "order_accepted",
        "order_rejected",
        "order_cancelled",
        "order_partially_filled",
        "order_filled",
    ]
    assert [event.severity for event in events] == [
        PaperAuditSeverity.INFO,
        PaperAuditSeverity.WARNING,
        PaperAuditSeverity.INFO,
        PaperAuditSeverity.INFO,
        PaperAuditSeverity.INFO,
    ]
    assert events[0].created_at == "2026-06-17T00:00:00Z"
    assert events[0].metadata["strategy"] == "unit"
    assert events[1].metadata["reason"] == "risk_block"
    assert events[1].source_event_id == rejected.event_id


def test_trade_audit_records_trade_and_positive_realized_pnl() -> None:
    _reducer, trade = _filled_trade(realized="positive")
    journal = PaperAuditJournal()

    events = journal.record_trade(trade)

    assert [event.event_type for event in events] == ["trade_recorded", "realized_pnl_recorded"]
    assert [event.severity for event in events] == [
        PaperAuditSeverity.INFO,
        PaperAuditSeverity.INFO,
    ]
    assert events[0].trade_id == trade.trade_id
    assert events[0].source_event_id == trade.event_id
    assert events[1].metadata["realized_pnl"] == pytest.approx(30.0)


def test_trade_audit_records_negative_realized_pnl_as_warning() -> None:
    _reducer, trade = _filled_trade(realized="negative")
    journal = PaperAuditJournal()

    events = journal.record_trade(trade)

    assert events[1].event_type == "realized_pnl_recorded"
    assert events[1].severity is PaperAuditSeverity.WARNING
    assert events[1].metadata["realized_pnl"] == pytest.approx(-10.0)


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

    spine = PaperEventSpine()
    event = spine.submit_order(order_id="safe", symbol="BTC/USDT", side="buy", quantity=1.0)
    audit = PaperAuditJournal().record_order_event(event)

    assert audit.event_type == "order_accepted"


def test_secret_like_metadata_is_rejected_and_safe_metadata_is_preserved() -> None:
    spine = PaperEventSpine()
    event = spine.submit_order(
        order_id="safe-metadata",
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        metadata={"strategy": "unit"},
    )
    journal = PaperAuditJournal()

    audit = journal.record_order_event(event)

    assert audit.metadata["strategy"] == "unit"
    unsafe = replace(event, metadata={"private_key": "blocked"})
    with pytest.raises(PaperAuditJournalError, match="forbidden credential-like"):
        journal.record_order_event(unsafe)


def test_snapshot_and_filters_are_deterministic() -> None:
    spine = PaperEventSpine()
    journal = PaperAuditJournal()
    accepted = spine.submit_order(
        order_id="ord-filter", symbol="BTC/USDT", side="buy", quantity=1.0
    )
    filled = spine.fill_order("ord-filter")
    reducer = PaperPortfolioReducer()
    trade = reducer.apply_event(filled, fill_price=100.0).trades[0]

    journal.record_order_event(accepted)
    journal.record_order_event(filled)
    journal.record_trade(trade)

    assert [event.sequence for event in journal.all_events()] == [1, 2, 3]
    assert [event.audit_id for event in journal.events_for_order("ord-filter")] == [
        "audit-000001",
        "audit-000002",
        "audit-000003",
    ]
    assert [event.audit_id for event in journal.events_for_trade(trade.trade_id)] == [
        "audit-000003"
    ]
    assert [event.audit_id for event in journal.events_for_source(filled.event_id)] == [
        "audit-000002",
        "audit-000003",
    ]


def test_recording_audit_does_not_mutate_portfolio_reducer_state() -> None:
    reducer, trade = _filled_trade(realized="positive")
    before = reducer.snapshot()

    journal = PaperAuditJournal()
    journal.record_trade(trade)

    after = reducer.snapshot()
    assert after == before
