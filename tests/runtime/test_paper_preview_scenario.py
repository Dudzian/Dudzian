from __future__ import annotations

import builtins
import os
import socket
from pathlib import Path

import pytest

from bot_core.runtime.paper_audit_journal import PaperAuditSeverity
from bot_core.runtime.paper_preview_scenario import (
    PaperPreviewScenario,
    PaperPreviewScenarioAction,
    PaperPreviewScenarioError,
    PaperPreviewScenarioRunner,
    PaperPreviewScenarioStep,
)
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    RuntimeCapability,
    build_preview_mode_policy,
)


def _scenario(*steps: PaperPreviewScenarioStep, name: str = "scenario") -> PaperPreviewScenario:
    return PaperPreviewScenario(name=name, steps=steps)


def _snapshot_counts(runner: PaperPreviewScenarioRunner) -> tuple[int, int, int]:
    snapshot = runner.flow.snapshot()
    return (len(snapshot.order_events), len(snapshot.portfolio.trades), len(snapshot.audit_events))


def test_scenario_runner_uses_preview_mode_contract() -> None:
    runner = PaperPreviewScenarioRunner(created_at="2026-06-17T00:00:00Z")

    assert runner.policy.mode is PreviewMode.PAPER
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in runner.policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in runner.policy.capabilities
    assert RuntimeCapability.LOCAL_TELEMETRY_AUDIT in runner.policy.capabilities

    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


def test_happy_path_scenario_submit_and_fill_updates_snapshot_and_summary() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="buy-1",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
                metadata={"scenario": "happy"},
            ),
            PaperPreviewScenarioStep(action="fill", order_id="buy-1", fill_price=100),
            name="happy",
        )
    )

    assert isinstance(result.step_results, tuple)
    assert len(result.final_snapshot.portfolio.trades) == 1
    position = result.final_snapshot.portfolio.positions[0]
    assert position.symbol == "BTCUSDT"
    assert position.quantity == 1
    assert position.avg_entry_price == 100
    audit_types = [event.event_type for event in result.final_snapshot.audit_events]
    assert audit_types == ["order_accepted", "order_filled", "trade_recorded"]
    assert result.summary.scenario_name == "happy"
    assert result.summary.step_count == 2
    assert result.summary.order_event_count == 2
    assert result.summary.trade_count == 1
    assert result.summary.position_count == 1
    assert result.summary.audit_event_count == 3
    assert result.summary.realized_pnl_total == 0
    assert result.summary.symbols == ("BTCUSDT",)
    assert result.summary.terminal_order_count == 1
    assert result.summary.failed_step_count == 0


def test_partial_then_final_fill_scenario_is_deterministic() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="buy-2", symbol="ETHUSDT", side="buy", quantity=2
            ),
            PaperPreviewScenarioStep(
                action="partial_fill", order_id="buy-2", fill_quantity=1, fill_price=100
            ),
            PaperPreviewScenarioStep(action="fill", order_id="buy-2", fill_price=120),
            name="partial-final",
        )
    )

    position = result.final_snapshot.portfolio.positions[0]
    assert position.quantity == 2
    assert position.avg_entry_price == 110
    assert [trade.price for trade in result.final_snapshot.portfolio.trades] == [100, 120]
    assert result.summary.order_event_count == 3
    assert result.summary.trade_count == 2
    assert result.summary.audit_event_count == 5
    assert result.summary.symbols == ("ETHUSDT",)


def test_realized_pnl_scenario_records_summary_and_audit() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=100),
            PaperPreviewScenarioStep(
                action="submit", order_id="sell", symbol="BTCUSDT", side="sell", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="sell", fill_price=130),
            name="realized-pnl",
        )
    )

    assert result.summary.realized_pnl_total == 30
    assert result.final_snapshot.portfolio.positions[0].quantity == 0
    assert "realized_pnl_recorded" in [
        event.event_type for event in result.final_snapshot.audit_events
    ]


def test_reject_and_cancel_scenario_has_no_trades_and_warning_audit() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="reject-me", symbol="SOLUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="reject", order_id="reject-me", reason="risk denied"),
            PaperPreviewScenarioStep(
                action="submit", order_id="cancel-me", symbol="ADAUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="cancel", order_id="cancel-me", reason="user cancel"),
            name="reject-cancel",
        )
    )

    assert result.final_snapshot.portfolio.trades == ()
    rejected_audits = [
        event
        for event in result.final_snapshot.audit_events
        if event.event_type == "order_rejected"
    ]
    assert rejected_audits[0].severity is PaperAuditSeverity.WARNING
    assert result.summary.order_event_count == 4
    assert result.summary.trade_count == 0
    assert result.summary.position_count == 0
    assert result.summary.audit_event_count == 4
    assert result.summary.symbols == ("ADAUSDT", "SOLUSDT")
    assert result.summary.terminal_order_count == 2


def test_invalid_action_raises_before_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="Unsupported scenario action"):
        runner.run(_scenario(PaperPreviewScenarioStep(action="explode", order_id="bad")))

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_invalid_fill_price_raises_before_fill_mutation() -> None:
    runner = PaperPreviewScenarioRunner()
    runner.flow.submit_order(order_id="buy", symbol="BTCUSDT", side="buy", quantity=1)

    with pytest.raises(PaperPreviewScenarioError, match="fill_price must be > 0"):
        runner.run(_scenario(PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=0)))

    snapshot = runner.flow.snapshot()
    assert [event.event_type.value for event in snapshot.order_events] == ["paper_order_accepted"]
    assert snapshot.portfolio.trades == ()
    assert [event.event_type for event in snapshot.audit_events] == ["order_accepted"]


def test_missing_required_field_raises_before_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="symbol is required"):
        runner.run(
            _scenario(
                PaperPreviewScenarioStep(
                    action="submit", order_id="missing", side="buy", quantity=1
                )
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_secret_metadata_key_is_rejected_before_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="credential-like"):
        runner.run(
            _scenario(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="secret",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=1,
                    metadata={"api_key": "not-allowed"},
                )
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_invalid_second_action_raises_before_any_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="Unsupported scenario action"):
        runner.run(
            _scenario(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="valid-first",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=1,
                ),
                PaperPreviewScenarioStep(action="explode", order_id="bad-second"),
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_second_step_missing_required_field_raises_before_any_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="symbol is required"):
        runner.run(
            _scenario(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="valid-first",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=1,
                ),
                PaperPreviewScenarioStep(
                    action="submit", order_id="bad-second", side="buy", quantity=1
                ),
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_second_step_secret_metadata_raises_before_any_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="credential-like"):
        runner.run(
            _scenario(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="valid-first",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=1,
                ),
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="bad-second",
                    symbol="ETHUSDT",
                    side="buy",
                    quantity=1,
                    metadata={"secret_token": "not-allowed"},
                ),
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_second_step_invalid_fill_price_raises_before_any_flow_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="fill_price must be > 0"):
        runner.run(
            _scenario(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="valid-first",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=1,
                ),
                PaperPreviewScenarioStep(action="fill", order_id="valid-first", fill_price=0),
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_empty_scenario_raises_without_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="at least one step"):
        runner.run(PaperPreviewScenario(name="empty", steps=()))

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_local_only_safety_no_network_env_reads_or_file_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network must not be used")

    def forbidden_getenv(*args: object, **kwargs: object) -> str | None:
        raise AssertionError("env must not be read")

    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("file writes must not be used")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    monkeypatch.setattr(os, "getenv", forbidden_getenv)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("file writes must not be used")
        ),
    )
    monkeypatch.setattr(builtins, "open", guarded_open)

    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action=PaperPreviewScenarioAction.SUBMIT,
                order_id="local",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
            ),
            PaperPreviewScenarioStep(action="fill", order_id="local", fill_price=100),
            name="local-only",
        )
    )

    assert result.summary.trade_count == 1
    assert result.summary.failed_step_count == 0


def test_snapshot_determinism_ordering_and_sorted_symbols() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="b", symbol="ZZZUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="b", fill_price=10),
            PaperPreviewScenarioStep(
                action="submit", order_id="a", symbol="AAAUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="a", fill_price=20),
            name="determinism",
        )
    )

    assert isinstance(result.step_results, tuple)
    assert [event.event_id for event in result.final_snapshot.order_events] == [
        "paper-000001",
        "paper-000002",
        "paper-000003",
        "paper-000004",
    ]
    assert [trade.trade_id for trade in result.final_snapshot.portfolio.trades] == [
        "trade-000002",
        "trade-000004",
    ]
    assert [event.audit_id for event in result.final_snapshot.audit_events] == [
        "audit-000001",
        "audit-000002",
        "audit-000003",
        "audit-000004",
        "audit-000005",
        "audit-000006",
    ]
    assert result.summary.symbols == ("AAAUSDT", "ZZZUSDT")
