from __future__ import annotations

import builtins
import os
import socket
from pathlib import Path

import pytest

import dataclasses

from bot_core.runtime.paper_audit_journal import PaperAuditSeverity
from bot_core.runtime.paper_preview_bundle_boundary import (
    PaperPreviewBundleBoundary,
    PaperPreviewBundleBoundaryError,
    build_local_bundle_boundary_matrix,
    build_local_bundle_refusal,
    refuse_local_bundle_boundary,
)
from bot_core.runtime.paper_preview_bundle_read_model import (
    PaperPreviewBundleReadModelError,
    PaperPreviewReadModelBoundary,
    build_paper_preview_bundle_read_model,
    build_paper_preview_read_model_boundary_matrix,
)
from bot_core.runtime.paper_preview_ui_runtime_preflight import (
    PaperPreviewUiRuntimePreflightError,
    build_paper_preview_ui_runtime_preflight,
)
from bot_core.runtime.paper_preview_integration_gate import (
    PaperPreviewIntegrationReadinessGateError,
    build_paper_preview_integration_readiness_gate,
)
from bot_core.runtime.paper_preview_scenario import (
    PaperPreviewDecisionContext,
    PaperPreviewDecisionDryRunAuditEntry,
    PaperPreviewDecisionDryRunAuditTrail,
    PaperPreviewLocalDecisionBundle,
    PaperPreviewRiskPlaceholder,
    PaperPreviewScenario,
    PaperPreviewScenarioAction,
    PaperPreviewScenarioError,
    PaperPreviewScenarioRunner,
    PaperPreviewScenarioStep,
)
from bot_core.runtime.read_only_market_data import (
    InMemoryReadOnlyMarketDataProvider,
    MarketCandle,
    MarketQuote,
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


def test_decision_context_exists_after_successful_scenario() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="buy-ctx", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="buy-ctx", fill_price=100),
            name="decision-context",
        )
    )

    context = result.decision_context
    assert context is not None
    assert context.scenario_name == "decision-context"
    assert context.step_count == 2
    assert context.trade_count == result.summary.trade_count == 1
    assert context.position_count == result.summary.position_count == 1
    assert context.audit_event_count == result.summary.audit_event_count == 3
    assert context.realized_pnl_total == result.summary.realized_pnl_total == 0
    assert context.decision_status == "context_only"
    assert context.generated_order_count == 0
    assert context.generated_decision_count == 0


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


def _market_provider() -> InMemoryReadOnlyMarketDataProvider:
    return InMemoryReadOnlyMarketDataProvider(
        quotes={
            "BTCUSDT": MarketQuote(
                "BTCUSDT", bid=100.0, ask=101.0, last=100.5, timestamp="2026-06-18T00:00:00Z"
            ),
            "ETHUSDT": MarketQuote(
                "ETHUSDT", bid=10.0, ask=11.0, last=10.5, timestamp="2026-06-18T00:00:01Z"
            ),
        },
        candles={
            ("BTCUSDT", "1m"): (
                MarketCandle("BTCUSDT", "1m", "2026-06-18T00:00:00Z", 100, 102, 99, 101, 5),
                MarketCandle("BTCUSDT", "1m", "2026-06-18T00:01:00Z", 101, 103, 100, 102, 6),
            ),
            ("ETHUSDT", "1m"): (
                MarketCandle("ETHUSDT", "1m", "2026-06-18T00:00:00Z", 10, 12, 9, 11, 7),
            ),
        },
    )


def test_scenario_with_read_only_market_context_keeps_paper_flow() -> None:
    result = PaperPreviewScenarioRunner(market_data_provider=_market_provider()).run(
        PaperPreviewScenario(
            name="market-context",
            market_symbols=("ETHUSDT", "BTCUSDT"),
            steps=(
                PaperPreviewScenarioStep(
                    action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                ),
                PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=100),
            ),
        )
    )

    assert result.market_context is not None
    assert result.market_context.symbols == ("BTCUSDT", "ETHUSDT")
    assert tuple(quote.symbol for quote in result.market_context.quotes) == ("BTCUSDT", "ETHUSDT")
    assert tuple(quote.last for quote in result.market_context.quotes) == (100.5, 10.5)
    assert result.market_context.candle_sets == ()
    assert result.summary.trade_count == 1
    assert result.summary.order_event_count == 2
    assert result.summary.audit_event_count == 3
    assert result.decision_context is not None
    assert result.decision_context.has_market_context is True
    assert result.decision_context.market_symbols == ("BTCUSDT", "ETHUSDT")
    assert result.decision_context.trade_count == result.summary.trade_count
    assert result.decision_context.audit_event_count == result.summary.audit_event_count
    assert result.decision_context.generated_order_count == 0
    assert result.decision_context.generated_decision_count == 0


def test_scenario_with_read_only_market_candles_context_is_deterministic() -> None:
    result = PaperPreviewScenarioRunner(market_data_provider=_market_provider()).run(
        PaperPreviewScenario(
            name="candles-context",
            market_symbols=("ETHUSDT", "BTCUSDT"),
            market_timeframe="1m",
            market_candle_limit=99,
            steps=(
                PaperPreviewScenarioStep(
                    action="submit", order_id="buy", symbol="ETHUSDT", side="buy", quantity=1
                ),
                PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=10),
            ),
        )
    )

    assert result.market_context is not None
    assert result.market_context.symbols == ("BTCUSDT", "ETHUSDT")
    assert tuple((item.symbol, item.timeframe) for item in result.market_context.candle_sets) == (
        ("BTCUSDT", "1m"),
        ("ETHUSDT", "1m"),
    )
    assert tuple(len(item.candles) for item in result.market_context.candle_sets) == (2, 1)
    assert isinstance(result.market_context.candle_sets, tuple)
    with pytest.raises(AttributeError):
        result.market_context.candle_sets.append("x")  # type: ignore[attr-defined]


def test_scenario_without_market_context_keeps_existing_path() -> None:
    result = PaperPreviewScenarioRunner(market_data_provider=_market_provider()).run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=100),
        )
    )

    assert result.market_context is None
    assert result.summary.trade_count == 1
    assert result.decision_context is not None
    assert result.decision_context.has_market_context is False
    assert result.decision_context.market_symbols == ()
    assert result.decision_context.trade_count == result.summary.trade_count


def test_decision_context_default_risk_placeholder_has_no_enforcement() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="risk-default",
                symbol="BTCUSDT",
                side="buy",
                quantity=10,
            ),
            PaperPreviewScenarioStep(action="fill", order_id="risk-default", fill_price=100),
        )
    )

    assert result.decision_context is not None
    assert result.decision_context.risk.source == "placeholder"
    assert result.decision_context.risk.risk_checks_enabled is False
    assert result.summary.trade_count == 1
    assert result.decision_context.generated_order_count == 0


def test_custom_risk_placeholder_is_context_only_and_does_not_block_flow() -> None:
    risk = PaperPreviewRiskPlaceholder(
        max_position_notional=1.0,
        max_order_quantity=0.1,
        max_daily_loss=2.0,
        risk_checks_enabled=False,
        source="unit-test-placeholder",
    )
    result = PaperPreviewScenarioRunner().run(
        PaperPreviewScenario(
            name="custom-risk",
            risk=risk,
            steps=(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="risk-custom",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=10,
                ),
                PaperPreviewScenarioStep(action="fill", order_id="risk-custom", fill_price=100),
            ),
        )
    )

    assert result.decision_context is not None
    assert result.decision_context.risk == risk
    assert result.summary.trade_count == 1
    assert result.decision_context.generated_order_count == 0
    assert result.decision_context.generated_decision_count == 0


def test_decision_context_and_risk_placeholder_are_immutable_and_deterministic() -> None:
    scenario = _scenario(
        PaperPreviewScenarioStep(
            action="submit", order_id="det", symbol="BTCUSDT", side="buy", quantity=1
        ),
        PaperPreviewScenarioStep(action="fill", order_id="det", fill_price=100),
        name="det-context",
    )
    first = PaperPreviewScenarioRunner(created_at="fixed").run(scenario).decision_context
    second = PaperPreviewScenarioRunner(created_at="fixed").run(scenario).decision_context

    assert first == second
    assert first is not None
    with pytest.raises(AttributeError):
        first.generated_order_count = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first.risk.source = "changed"  # type: ignore[misc]


def test_decision_context_has_no_decision_engine_account_or_secret_surface() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="surface", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="surface", fill_price=100),
        )
    )
    assert result.decision_context is not None
    forbidden_methods = {
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "generate_order",
        "execute",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "export",
        "cloud_sink",
    }
    forbidden_fields = {
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
    }

    assert forbidden_methods.isdisjoint(set(dir(result.decision_context)))
    assert forbidden_methods.isdisjoint(set(dir(PaperPreviewScenarioRunner())))
    assert forbidden_fields.isdisjoint(set(dir(result.decision_context)))
    assert forbidden_fields.isdisjoint(set(dir(result.decision_context.risk)))
    assert isinstance(result.decision_context, PaperPreviewDecisionContext)


def test_missing_market_provider_raises_before_paper_mutation() -> None:
    runner = PaperPreviewScenarioRunner()

    with pytest.raises(PaperPreviewScenarioError, match="market_data_provider is required"):
        runner.run(
            PaperPreviewScenario(
                name="missing-provider",
                market_symbols=("BTCUSDT",),
                steps=(
                    PaperPreviewScenarioStep(
                        action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                    ),
                ),
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_unknown_market_symbol_raises_before_paper_mutation() -> None:
    runner = PaperPreviewScenarioRunner(market_data_provider=_market_provider())

    with pytest.raises(PaperPreviewScenarioError, match="unknown market data symbol"):
        runner.run(
            PaperPreviewScenario(
                name="unknown-symbol",
                market_symbols=("DOGEUSDT",),
                steps=(
                    PaperPreviewScenarioStep(
                        action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                    ),
                ),
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"market_timeframe": "1m"}, "provided together"),
        ({"market_candle_limit": 1}, "provided together"),
        ({"market_timeframe": "1m", "market_candle_limit": 0}, "must be > 0"),
        ({"market_timeframe": " ", "market_candle_limit": 1}, "non-empty"),
    ],
)
def test_invalid_market_request_raises_before_paper_mutation(
    kwargs: dict[str, object], message: str
) -> None:
    runner = PaperPreviewScenarioRunner(market_data_provider=_market_provider())

    with pytest.raises(PaperPreviewScenarioError, match=message):
        runner.run(
            PaperPreviewScenario(
                name="invalid-market",
                market_symbols=("BTCUSDT",),
                steps=(
                    PaperPreviewScenarioStep(
                        action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                    ),
                ),
                **kwargs,
            )
        )

    assert _snapshot_counts(runner) == (0, 0, 0)


def test_read_only_market_policy_gate_for_scenario_context() -> None:
    policy = build_preview_mode_policy(
        PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    )
    assert policy.capabilities == (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.READ_ONLY_MARKET, (capability,))


def test_scenario_market_context_keeps_account_order_method_separation() -> None:
    runner = PaperPreviewScenarioRunner(market_data_provider=_market_provider())
    forbidden = {
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions",
        "get_open_orders",
        "create_order",
        "submit_order",
        "cancel_order",
        "read_credentials",
    }

    assert forbidden.isdisjoint(set(dir(runner.market_data_provider)))
    assert forbidden.isdisjoint(set(dir(runner)))


def test_market_context_has_no_secret_metadata_surface() -> None:
    result = PaperPreviewScenarioRunner(market_data_provider=_market_provider()).run(
        PaperPreviewScenario(
            name="secret-surface",
            market_symbols=("BTCUSDT",),
            steps=(
                PaperPreviewScenarioStep(
                    action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                ),
            ),
        )
    )

    assert result.market_context is not None
    for field in (
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
    ):
        assert not hasattr(result.market_context, field)


def test_dry_run_artifact_exists_after_successful_scenario() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="dry", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="dry", fill_price=100),
            name="dry-run",
        )
    )

    artifact = result.dry_run_artifact
    assert artifact is not None
    assert artifact.artifact_kind == "context_only_dry_run"
    assert artifact.scenario_name == "dry-run"
    assert artifact.step_count == 2
    assert artifact.generated_order_count == 0
    assert artifact.generated_decision_count == 0
    assert artifact.no_action_reason == "dry_run_context_only"
    assert result.dry_run_artifact_audit.entries
    assert len(result.dry_run_artifact_audit.entries) == 1
    entry = result.dry_run_artifact_audit.entries[0]
    assert entry.event_type == "decision_dry_run_artifact_created"
    assert entry.artifact_kind == "context_only_dry_run"
    assert entry.no_action_reason == "dry_run_context_only"


def test_dry_run_artifact_mirrors_decision_context_and_risk() -> None:
    risk = PaperPreviewRiskPlaceholder(risk_checks_enabled=True, source="unit-risk")
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        PaperPreviewScenario(
            name="mirror",
            risk=risk,
            steps=(
                PaperPreviewScenarioStep(
                    action="submit", order_id="mirror", symbol="BTCUSDT", side="buy", quantity=1
                ),
                PaperPreviewScenarioStep(action="fill", order_id="mirror", fill_price=100),
            ),
        )
    )

    context = result.decision_context
    artifact = result.dry_run_artifact
    assert context is not None
    assert artifact is not None
    assert artifact.decision_status == context.decision_status
    assert artifact.market_symbols == context.market_symbols
    assert artifact.trade_count == context.trade_count == result.summary.trade_count
    assert artifact.position_count == context.position_count == result.summary.position_count
    assert (
        artifact.audit_event_count == context.audit_event_count == result.summary.audit_event_count
    )
    assert artifact.realized_pnl_total == context.realized_pnl_total
    assert artifact.risk_source == context.risk.source == "unit-risk"
    assert artifact.risk_checks_enabled is context.risk.risk_checks_enabled is True
    entry = result.dry_run_artifact_audit.entries[0]
    assert entry.scenario_name == artifact.scenario_name == result.summary.scenario_name
    assert entry.step_count == artifact.step_count == result.summary.step_count
    assert entry.decision_status == artifact.decision_status
    assert entry.generated_order_count == artifact.generated_order_count == 0
    assert entry.generated_decision_count == artifact.generated_decision_count == 0
    assert entry.trade_count == artifact.trade_count == result.summary.trade_count
    assert entry.position_count == artifact.position_count == result.summary.position_count
    assert entry.audit_event_count == artifact.audit_event_count == result.summary.audit_event_count
    assert entry.risk_source == artifact.risk_source == "unit-risk"
    assert entry.risk_checks_enabled is artifact.risk_checks_enabled is True


def test_dry_run_artifact_includes_market_context_summary() -> None:
    result = PaperPreviewScenarioRunner(market_data_provider=_market_provider()).run(
        PaperPreviewScenario(
            name="dry-market",
            market_symbols=("ETHUSDT", "BTCUSDT"),
            market_timeframe="1m",
            market_candle_limit=2,
            steps=(
                PaperPreviewScenarioStep(
                    action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                ),
                PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=100),
            ),
        )
    )

    artifact = result.dry_run_artifact
    assert artifact is not None
    assert artifact.has_market_context is True
    assert artifact.market_symbols == ("BTCUSDT", "ETHUSDT")
    assert artifact.quote_count == 2
    assert artifact.candle_set_count == 2
    assert artifact.paper_symbols == ("BTCUSDT",)
    entry = result.dry_run_artifact_audit.entries[0]
    assert entry.has_market_context is True
    assert entry.market_symbols == ("BTCUSDT", "ETHUSDT")
    assert entry.quote_count == 2
    assert entry.candle_set_count == 2


def test_dry_run_artifact_without_market_context_summary_is_empty() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="no-market", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="no-market", fill_price=100),
            name="no-market",
        )
    )

    artifact = result.dry_run_artifact
    assert artifact is not None
    assert artifact.has_market_context is False
    assert artifact.market_symbols == ()
    assert artifact.quote_count == 0
    assert artifact.candle_set_count == 0
    entry = result.dry_run_artifact_audit.entries[0]
    assert entry.has_market_context is False
    assert entry.market_symbols == ()
    assert entry.quote_count == 0
    assert entry.candle_set_count == 0


def test_dry_run_artifact_does_not_affect_paper_flow_counts_or_audit() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="flow", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="flow", fill_price=100),
            name="flow-unchanged",
        )
    )

    artifact = result.dry_run_artifact
    assert artifact is not None
    assert artifact.order_event_count == result.summary.order_event_count == 2
    assert artifact.trade_count == result.summary.trade_count == 1
    assert artifact.audit_event_count == result.summary.audit_event_count == 3
    assert [event.event_type for event in result.final_snapshot.audit_events] == [
        "order_accepted",
        "order_filled",
        "trade_recorded",
    ]
    assert len(result.dry_run_artifact_audit.entries) == 1


def test_dry_run_artifact_is_immutable_and_deterministic() -> None:
    scenario = _scenario(
        PaperPreviewScenarioStep(
            action="submit", order_id="det-dry", symbol="BTCUSDT", side="buy", quantity=1
        ),
        PaperPreviewScenarioStep(action="fill", order_id="det-dry", fill_price=100),
        name="det-dry",
    )
    first_result = PaperPreviewScenarioRunner(created_at="fixed").run(scenario)
    second_result = PaperPreviewScenarioRunner(created_at="fixed").run(scenario)
    first = first_result.dry_run_artifact
    second = second_result.dry_run_artifact

    assert first == second
    assert first_result.dry_run_artifact_audit == second_result.dry_run_artifact_audit
    assert first is not None
    with pytest.raises(AttributeError):
        first.generated_order_count = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first_result.dry_run_artifact_audit.entries = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first_result.dry_run_artifact_audit.entries.append("x")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        first_result.dry_run_artifact_audit.entries[0].event_type = "changed"  # type: ignore[misc]


def test_dry_run_artifact_has_no_decision_scoring_account_secret_or_export_surface() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="surface-dry", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="surface-dry", fill_price=100),
        )
    )
    artifact = result.dry_run_artifact
    assert artifact is not None
    audit = result.dry_run_artifact_audit
    entry = audit.entries[0]
    forbidden_decision_surface = {
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "confidence",
    }
    forbidden_account_secret_surface = {
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "external_export_sink",
    }
    forbidden_artifact_only = {
        "export",
        "cloud_sink",
    }

    assert forbidden_decision_surface.isdisjoint(set(dir(artifact)))
    assert forbidden_decision_surface.isdisjoint(set(dir(audit)))
    assert forbidden_decision_surface.isdisjoint(set(dir(entry)))
    assert forbidden_decision_surface.isdisjoint(set(dir(PaperPreviewScenarioRunner())))
    assert forbidden_account_secret_surface.isdisjoint(set(dir(artifact)))
    assert forbidden_account_secret_surface.isdisjoint(set(dir(audit)))
    assert forbidden_account_secret_surface.isdisjoint(set(dir(entry)))
    assert forbidden_account_secret_surface.isdisjoint(set(dir(PaperPreviewScenarioRunner())))
    assert forbidden_artifact_only.isdisjoint(set(dir(artifact)))
    assert entry.export_sink == "none"
    assert entry.cloud_sink == "none"
    assert entry.external_export is False
    assert not hasattr(entry, "export_path")
    assert not hasattr(entry, "file_path")
    assert not hasattr(entry, "cloud_url")
    assert isinstance(entry, PaperPreviewDecisionDryRunAuditEntry)
    assert isinstance(audit, PaperPreviewDecisionDryRunAuditTrail)


def test_dry_run_artifact_lists_blocked_integrations() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="blocked", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="blocked", fill_price=100),
        )
    )

    artifact = result.dry_run_artifact
    assert artifact is not None
    assert set(artifact.blocked_engine_integrations) >= {
        "strategy_engine",
        "ai_model_inference",
        "decision_envelope",
        "trading_controller",
        "order_generation",
    }


def test_dry_run_preview_policy_allows_read_only_and_paper_but_blocks_live() -> None:
    read_only = build_preview_mode_policy(
        PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    )
    paper = PaperPreviewScenarioRunner(created_at="fixed").policy

    assert read_only.capabilities == (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in paper.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in paper.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


def test_local_bundle_exists_after_successful_scenario() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="bundle", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="bundle", fill_price=100),
            name="bundle",
        )
    )

    assert result.dry_run_artifact is not None
    assert len(result.dry_run_artifact_audit.entries) == 1
    assert result.local_bundle is not None
    assert result.local_bundle.bundle_kind == "local_context_artifact_audit_bundle"


def test_local_bundle_mirrors_decision_context_artifact_and_audit() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="mirror-bundle", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="mirror-bundle", fill_price=100),
            name="mirror-bundle",
        )
    )

    bundle = result.local_bundle
    context = result.decision_context
    artifact = result.dry_run_artifact
    assert bundle is not None
    assert context is not None
    assert artifact is not None
    assert bundle.scenario_name == context.scenario_name == result.summary.scenario_name
    assert bundle.step_count == context.step_count == result.summary.step_count
    assert bundle.decision_status == context.decision_status
    assert bundle.artifact_kind == artifact.artifact_kind
    assert bundle.artifact_no_action_reason == artifact.no_action_reason
    assert bundle.audit_entry_count == 1
    assert bundle.audit_event_types == ("decision_dry_run_artifact_created",)
    assert bundle.generated_order_count == 0
    assert bundle.generated_decision_count == 0


def test_local_bundle_includes_market_context_summary() -> None:
    result = PaperPreviewScenarioRunner(market_data_provider=_market_provider()).run(
        PaperPreviewScenario(
            name="bundle-market",
            market_symbols=("ETHUSDT", "BTCUSDT"),
            market_timeframe="1m",
            market_candle_limit=2,
            steps=(
                PaperPreviewScenarioStep(
                    action="submit", order_id="buy", symbol="BTCUSDT", side="buy", quantity=1
                ),
                PaperPreviewScenarioStep(action="fill", order_id="buy", fill_price=100),
            ),
        )
    )

    bundle = result.local_bundle
    assert bundle is not None
    assert bundle.has_market_context is True
    assert bundle.market_symbols == ("BTCUSDT", "ETHUSDT")
    assert bundle.quote_count == 2
    assert bundle.candle_set_count == 2


def test_local_bundle_without_market_context_summary_is_empty() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="bundle-no-market",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
            ),
            PaperPreviewScenarioStep(action="fill", order_id="bundle-no-market", fill_price=100),
            name="bundle-no-market",
        )
    )

    bundle = result.local_bundle
    assert bundle is not None
    assert bundle.has_market_context is False
    assert bundle.market_symbols == ()
    assert bundle.quote_count == 0
    assert bundle.candle_set_count == 0


def test_local_bundle_no_export_no_cloud_and_no_file_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("file writes must not be used")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="no-export", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="no-export", fill_price=100),
            name="no-export",
        )
    )

    bundle = result.local_bundle
    assert bundle is not None
    assert bundle.export_sink == "none"
    assert bundle.cloud_sink == "none"
    assert bundle.external_export is False
    assert not hasattr(bundle, "export_path")
    assert not hasattr(bundle, "file_path")
    assert not hasattr(bundle, "cloud_url")


def test_local_bundle_does_not_affect_paper_flow_counts_or_audit() -> None:
    result = PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="bundle-flow", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="bundle-flow", fill_price=100),
            name="bundle-flow",
        )
    )

    bundle = result.local_bundle
    artifact = result.dry_run_artifact
    assert bundle is not None
    assert artifact is not None
    assert (
        bundle.order_event_count == artifact.order_event_count == result.summary.order_event_count
    )
    assert bundle.trade_count == artifact.trade_count == result.summary.trade_count
    assert (
        bundle.audit_event_count == artifact.audit_event_count == result.summary.audit_event_count
    )
    assert [event.event_type for event in result.final_snapshot.audit_events] == [
        "order_accepted",
        "order_filled",
        "trade_recorded",
    ]
    assert len(result.final_snapshot.order_events) == 2
    assert len(result.final_snapshot.portfolio.trades) == 1


def test_local_bundle_is_immutable_and_deterministic() -> None:
    scenario = _scenario(
        PaperPreviewScenarioStep(
            action="submit", order_id="det-bundle", symbol="BTCUSDT", side="buy", quantity=1
        ),
        PaperPreviewScenarioStep(action="fill", order_id="det-bundle", fill_price=100),
        name="det-bundle",
    )
    first = PaperPreviewScenarioRunner(created_at="fixed").run(scenario).local_bundle
    second = PaperPreviewScenarioRunner(created_at="fixed").run(scenario).local_bundle

    assert first == second
    assert first is not None
    assert isinstance(first, PaperPreviewLocalDecisionBundle)
    with pytest.raises(AttributeError):
        first.generated_order_count = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first.market_symbols.append("X")  # type: ignore[attr-defined]


def test_local_bundle_has_no_decision_scoring_recommendation_surface() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="bundle-surface", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="bundle-surface", fill_price=100),
        )
    )
    bundle = result.local_bundle
    assert bundle is not None
    forbidden = {
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
    }
    assert forbidden.isdisjoint(set(dir(bundle)))
    assert forbidden.isdisjoint(set(dir(PaperPreviewScenarioRunner())))


def test_local_bundle_has_no_account_live_secret_or_export_surface() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="bundle-secret", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="bundle-secret", fill_price=100),
        )
    )
    bundle = result.local_bundle
    assert bundle is not None
    forbidden = {
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export",
        "external_export_sink",
        "export_path",
        "file_path",
        "cloud_url",
    }
    assert forbidden.isdisjoint(set(dir(bundle)))
    assert forbidden.isdisjoint(set(dir(PaperPreviewScenarioRunner())))


def test_local_bundle_preserves_blocked_engine_integrations() -> None:
    result = PaperPreviewScenarioRunner().run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="blocked", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="blocked", fill_price=100),
        )
    )

    bundle = result.local_bundle
    assert bundle is not None
    assert {
        "strategy_engine",
        "ai_model_inference",
        "decision_envelope",
        "trading_controller",
        "order_generation",
    }.issubset(set(bundle.blocked_engine_integrations))


def test_local_bundle_preview_policy_keeps_live_capabilities_blocked() -> None:
    read_only_policy = build_preview_mode_policy(
        PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    )
    paper_policy = build_preview_mode_policy(
        PreviewMode.PAPER,
        (RuntimeCapability.PAPER_ORDER_SUBMIT, RuntimeCapability.PAPER_ORDER_LIFECYCLE),
    )
    assert read_only_policy.capabilities == (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    assert paper_policy.capabilities == (
        RuntimeCapability.PAPER_ORDER_SUBMIT,
        RuntimeCapability.PAPER_ORDER_LIFECYCLE,
    )
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


def _local_bundle_result():
    return PaperPreviewScenarioRunner(created_at="fixed").run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="boundary-order",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
            ),
            PaperPreviewScenarioStep(action="fill", order_id="boundary-order", fill_price=100),
            name="bundle-boundary",
        )
    )


def test_bundle_boundary_matrix_exists_for_local_bundle() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    report = build_local_bundle_boundary_matrix(result.local_bundle)

    assert report.report_kind == "local_bundle_boundary_refusal_matrix"
    assert report.bundle_kind == result.local_bundle.bundle_kind
    assert report.scenario_name == result.local_bundle.scenario_name
    assert report.row_count == len(PaperPreviewBundleBoundary)
    assert len(report.rows) == len(PaperPreviewBundleBoundary)
    assert report.all_refused is True
    assert report.export_sink == "none"
    assert report.cloud_sink == "none"
    assert report.external_export is False
    assert report.generated_order_count == 0
    assert report.generated_decision_count == 0


def test_bundle_boundary_matrix_contains_every_boundary_once_in_enum_order() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    report = build_local_bundle_boundary_matrix(result.local_bundle)
    expected = tuple(boundary.value for boundary in PaperPreviewBundleBoundary)

    assert tuple(row.boundary_kind for row in report.rows) == expected
    assert {row.boundary_kind for row in report.rows} == set(expected)
    assert len({row.boundary_kind for row in report.rows}) == len(PaperPreviewBundleBoundary)
    assert all(row.refused is True for row in report.rows)


def test_bundle_boundary_matrix_rows_mirror_refusal_data() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    report = build_local_bundle_boundary_matrix(result.local_bundle)

    for row, boundary in zip(report.rows, PaperPreviewBundleBoundary, strict=True):
        refusal = build_local_bundle_refusal(result.local_bundle, boundary)
        assert row.boundary_kind == refusal.boundary_kind
        assert row.reason == refusal.reason
        assert row.bundle_kind == result.local_bundle.bundle_kind
        assert row.scenario_name == result.local_bundle.scenario_name
        assert row.generated_order_count == 0
        assert row.generated_decision_count == 0
        assert row.export_sink == "none"
        assert row.cloud_sink == "none"
        assert row.external_export is False


def test_bundle_boundary_matrix_covers_export_serialization_cloud_external() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    rows = {
        row.boundary_kind: row
        for row in build_local_bundle_boundary_matrix(result.local_bundle).rows
    }

    for boundary in ("file_export", "serialized_export", "cloud_sink", "external_export"):
        assert rows[boundary].refused is True
        assert (
            rows[boundary].reason
            == build_local_bundle_refusal(result.local_bundle, boundary).reason
        )


def test_bundle_boundary_matrix_covers_engine_and_order_handoffs() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    rows = {
        row.boundary_kind: row
        for row in build_local_bundle_boundary_matrix(result.local_bundle).rows
    }

    for boundary in (
        "strategy_engine_handoff",
        "ai_model_inference_handoff",
        "scoring_handoff",
        "recommendation_handoff",
        "decision_envelope_handoff",
        "trading_controller_handoff",
        "order_generation_handoff",
    ):
        assert rows[boundary].refused is True
        assert rows[boundary].generated_order_count == 0
        assert rows[boundary].generated_decision_count == 0


def test_bundle_boundary_matrix_has_no_file_network_or_serialization_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("matrix must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket used")),
    )

    report = build_local_bundle_boundary_matrix(result.local_bundle)

    for forbidden in (
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "to_json",
        "to_yaml",
        "to_csv",
    ):
        assert not hasattr(report, forbidden)
        assert all(not hasattr(row, forbidden) for row in report.rows)


def test_bundle_boundary_matrix_is_deterministic_and_immutable() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    first = build_local_bundle_boundary_matrix(result.local_bundle)
    second = build_local_bundle_boundary_matrix(result.local_bundle)

    assert first == second
    assert isinstance(first.rows, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.export_sink = "file"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.rows[0] = first.rows[0]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.rows[0].export_sink = "file"  # type: ignore[misc]


def test_bundle_boundary_matrix_does_not_affect_paper_flow() -> None:
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    result = runner.run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="matrix-stable", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="matrix-stable", fill_price=100),
            name="matrix-stable-flow",
        )
    )
    assert result.local_bundle is not None
    before = _snapshot_counts(runner)

    report = build_local_bundle_boundary_matrix(result.local_bundle)

    assert report.row_count == len(PaperPreviewBundleBoundary)
    assert _snapshot_counts(runner) == before
    assert before == (
        result.summary.order_event_count,
        result.summary.trade_count,
        result.summary.audit_event_count,
    )


def test_bundle_boundary_matrix_has_no_decision_account_live_secret_export_surface() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    report = build_local_bundle_boundary_matrix(result.local_bundle)
    runner = PaperPreviewScenarioRunner(created_at="fixed")

    forbidden = {
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "order_payload",
    }
    for item in (result.local_bundle, report, *report.rows, runner):
        for name in forbidden:
            assert not hasattr(item, name)


def test_bundle_boundary_refuses_file_export_without_file_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("file writes must not be attempted")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    with pytest.raises(
        PaperPreviewBundleBoundaryError,
        match="local bundle boundary refuses file_export",
    ):
        refuse_local_bundle_boundary(result.local_bundle, PaperPreviewBundleBoundary.FILE_EXPORT)

    refusal = build_local_bundle_refusal(result.local_bundle, "file_export")
    assert refusal.boundary_kind == "file_export"
    assert refusal.refused is True
    assert refusal.export_sink == "none"
    assert refusal.cloud_sink == "none"
    assert refusal.external_export is False
    assert refusal.generated_order_count == 0
    assert refusal.generated_decision_count == 0
    assert refusal.bundle_kind == result.local_bundle.bundle_kind


def test_bundle_boundary_refuses_serialized_export_without_payload_surface() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    with pytest.raises(PaperPreviewBundleBoundaryError, match="serialized_export"):
        refuse_local_bundle_boundary(result.local_bundle, "serialized_export")

    for name in ("to_json", "to_yaml", "to_csv", "serialize", "serialized_payload"):
        assert not hasattr(result.local_bundle, name)


def test_bundle_boundary_refuses_cloud_and_external_export_without_network_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket used")),
    )

    for boundary in ("cloud_sink", "external_export"):
        with pytest.raises(PaperPreviewBundleBoundaryError, match=boundary):
            refuse_local_bundle_boundary(result.local_bundle, boundary)
        refusal = build_local_bundle_refusal(result.local_bundle, boundary)
        assert refusal.cloud_sink == "none"
        assert refusal.export_sink == "none"
        assert refusal.external_export is False

    assert not hasattr(result.local_bundle, "cloud_url")


@pytest.mark.parametrize(
    "boundary",
    [
        "strategy_engine_handoff",
        "ai_model_inference_handoff",
        "scoring_handoff",
        "recommendation_handoff",
        "decision_envelope_handoff",
        "trading_controller_handoff",
        "order_generation_handoff",
    ],
)
def test_bundle_boundary_refuses_engine_and_order_handoffs(boundary: str) -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    with pytest.raises(PaperPreviewBundleBoundaryError, match=boundary):
        refuse_local_bundle_boundary(result.local_bundle, boundary)

    refusal = build_local_bundle_refusal(result.local_bundle, boundary)
    assert refusal.boundary_kind == boundary
    assert refusal.generated_order_count == 0
    assert refusal.generated_decision_count == 0
    assert refusal.export_sink == "none"
    assert refusal.cloud_sink == "none"
    assert refusal.external_export is False


def test_bundle_refusal_is_deterministic_and_immutable() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    first = build_local_bundle_refusal(result.local_bundle, "file_export")
    second = build_local_bundle_refusal(result.local_bundle, PaperPreviewBundleBoundary.FILE_EXPORT)

    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.export_sink = "file"  # type: ignore[misc]

    with pytest.raises(PaperPreviewBundleBoundaryError) as first_error:
        refuse_local_bundle_boundary(result.local_bundle, "file_export")
    with pytest.raises(PaperPreviewBundleBoundaryError) as second_error:
        refuse_local_bundle_boundary(result.local_bundle, "file_export")
    assert str(first_error.value) == str(second_error.value)


def test_bundle_boundary_refusal_does_not_affect_paper_flow() -> None:
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    result = runner.run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="stable", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="stable", fill_price=100),
            name="stable-flow",
        )
    )
    assert result.local_bundle is not None
    before = _snapshot_counts(runner)

    for boundary in PaperPreviewBundleBoundary:
        with pytest.raises(PaperPreviewBundleBoundaryError):
            refuse_local_bundle_boundary(result.local_bundle, boundary)

    assert _snapshot_counts(runner) == before
    assert before == (
        result.summary.order_event_count,
        result.summary.trade_count,
        result.summary.audit_event_count,
    )


def test_bundle_boundary_has_no_decision_scoring_recommendation_account_live_secret_surface() -> (
    None
):
    result = _local_bundle_result()
    assert result.local_bundle is not None
    refusal = build_local_bundle_refusal(result.local_bundle, "file_export")
    runner = PaperPreviewScenarioRunner(created_at="fixed")

    forbidden = {
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }
    for item in (result.local_bundle, refusal, runner):
        for name in forbidden:
            assert not hasattr(item, name)


def test_bundle_boundary_unknown_kind_fails_closed() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    with pytest.raises(PaperPreviewBundleBoundaryError, match="unknown boundary"):
        build_local_bundle_refusal(result.local_bundle, "unlisted_boundary")


def test_preview_policy_boundary_live_capabilities_remain_blocked() -> None:
    assert (
        RuntimeCapability.READ_ONLY_MARKET_FETCH
        in build_preview_mode_policy(
            PreviewMode.READ_ONLY_MARKET,
            (RuntimeCapability.READ_ONLY_MARKET_FETCH,),
        ).capabilities
    )
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in PaperPreviewScenarioRunner().policy.capabilities
    assert (
        RuntimeCapability.PAPER_ORDER_LIFECYCLE in PaperPreviewScenarioRunner().policy.capabilities
    )

    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


def _local_bundle_with_market_result():
    return PaperPreviewScenarioRunner(
        created_at="fixed", market_data_provider=_market_provider()
    ).run(
        PaperPreviewScenario(
            name="bundle-read-model-market",
            market_symbols=("ETHUSDT", "BTCUSDT"),
            market_timeframe="1m",
            market_candle_limit=99,
            steps=(
                PaperPreviewScenarioStep(
                    action="submit",
                    order_id="read-model-market",
                    symbol="ETHUSDT",
                    side="buy",
                    quantity=1,
                ),
                PaperPreviewScenarioStep(
                    action="fill", order_id="read-model-market", fill_price=10
                ),
            ),
        )
    )


def _read_model_for_result(result):
    assert result.local_bundle is not None
    matrix = build_local_bundle_boundary_matrix(result.local_bundle)
    return build_paper_preview_bundle_read_model(result.local_bundle, matrix), matrix


def test_bundle_read_model_exists_for_local_bundle_and_matrix() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None

    model, matrix = _read_model_for_result(result)

    assert model.model_kind == "local_preview_bundle_read_model"
    assert model.scenario_name == result.local_bundle.scenario_name == matrix.scenario_name
    assert model.bundle_kind == result.local_bundle.bundle_kind == matrix.bundle_kind


def test_bundle_read_model_mirrors_bundle_and_boundary_matrix() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    model, matrix = _read_model_for_result(result)

    assert model.decision_status == result.local_bundle.decision_status
    assert model.market_symbols == result.local_bundle.market_symbols
    assert model.trade_count == result.local_bundle.trade_count
    assert model.position_count == result.local_bundle.position_count
    assert model.audit_event_count == result.local_bundle.audit_event_count
    assert model.generated_order_count == 0
    assert model.generated_decision_count == 0
    assert model.export_sink == "none"
    assert model.cloud_sink == "none"
    assert model.external_export is False
    assert model.boundary_row_count == len(matrix.rows)
    assert model.all_boundaries_refused is True
    assert model.boundary_kinds == tuple(row.boundary_kind for row in matrix.rows)
    assert tuple(row.boundary_kind for row in model.boundary_rows) == model.boundary_kinds


def test_bundle_read_model_summarizes_market_and_no_market_context() -> None:
    market_result = _local_bundle_with_market_result()
    assert market_result.local_bundle is not None
    market_model, _ = _read_model_for_result(market_result)

    assert market_model.has_market_context is True
    assert market_model.market_symbols == ("BTCUSDT", "ETHUSDT")
    assert market_model.quote_count == 2
    assert market_model.candle_set_count == 2

    no_market_result = _local_bundle_result()
    assert no_market_result.local_bundle is not None
    no_market_model, _ = _read_model_for_result(no_market_result)

    assert no_market_model.has_market_context is False
    assert no_market_model.market_symbols == ()
    assert no_market_model.quote_count == 0
    assert no_market_model.candle_set_count == 0


def test_bundle_read_model_static_flags_and_local_contract() -> None:
    model, _ = _read_model_for_result(_local_bundle_result())

    assert model.read_only is True
    assert model.runtime_backed is False
    assert model.ui_bound is False
    assert model.has_boundary_matrix is True
    assert model.has_dry_run_artifact is True
    assert model.has_audit_trail is True
    assert model.has_decision_context is True


def test_bundle_read_model_consistency_fails_closed() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    matrix = build_local_bundle_boundary_matrix(result.local_bundle)

    with pytest.raises(PaperPreviewBundleReadModelError, match="scenario_name"):
        build_paper_preview_bundle_read_model(
            result.local_bundle, dataclasses.replace(matrix, scenario_name="other")
        )
    with pytest.raises(PaperPreviewBundleReadModelError, match="bundle_kind"):
        build_paper_preview_bundle_read_model(
            result.local_bundle, dataclasses.replace(matrix, bundle_kind="other")
        )
    with pytest.raises(PaperPreviewBundleReadModelError, match="refuse all"):
        build_paper_preview_bundle_read_model(
            result.local_bundle, dataclasses.replace(matrix, all_refused=False)
        )
    with pytest.raises(PaperPreviewBundleReadModelError, match="row_count"):
        build_paper_preview_bundle_read_model(
            result.local_bundle, dataclasses.replace(matrix, row_count=matrix.row_count + 1)
        )


def test_bundle_read_model_has_no_file_network_serialization_or_ui_side_effects(
    monkeypatch,
) -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    matrix = build_local_bundle_boundary_matrix(result.local_bundle)

    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(token in mode for token in ("w", "a", "+", "x")):
            raise AssertionError("read model must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")),
    )
    monkeypatch.setattr(
        socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("socket"))
    )

    model = build_paper_preview_bundle_read_model(result.local_bundle, matrix)

    forbidden = {
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "qml",
        "QObject",
        "to_json",
        "to_yaml",
        "to_csv",
    }
    for name in forbidden:
        assert not hasattr(model, name)


def test_bundle_read_model_is_deterministic_and_immutable() -> None:
    model, matrix = _read_model_for_result(_local_bundle_result())
    result = _local_bundle_result()
    assert result.local_bundle is not None
    repeated = build_paper_preview_bundle_read_model(result.local_bundle, matrix)

    assert repeated == model
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.read_only = False  # type: ignore[misc]
    assert isinstance(model.market_symbols, tuple)
    with pytest.raises(AttributeError):
        model.market_symbols.append("X")  # type: ignore[attr-defined]


def test_bundle_read_model_does_not_affect_paper_flow_or_audit() -> None:
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    result = runner.run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="stable-read", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="stable-read", fill_price=100),
            name="stable-read-model",
        )
    )
    assert result.local_bundle is not None
    before = _snapshot_counts(runner)
    matrix = build_local_bundle_boundary_matrix(result.local_bundle)

    build_paper_preview_bundle_read_model(result.local_bundle, matrix)

    assert _snapshot_counts(runner) == before
    assert before == (
        result.summary.order_event_count,
        result.summary.trade_count,
        result.summary.audit_event_count,
    )


def test_bundle_read_model_has_no_decision_account_live_secret_export_ui_surface() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    matrix = build_local_bundle_boundary_matrix(result.local_bundle)
    model = build_paper_preview_bundle_read_model(result.local_bundle, matrix)
    runner = PaperPreviewScenarioRunner(created_at="fixed")

    forbidden = {
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
        "qml",
        "QObject",
    }
    for item in (model, result.local_bundle, matrix, runner):
        for name in forbidden:
            assert not hasattr(item, name)


def _read_model_boundary_report_for_result(result):
    model, _ = _read_model_for_result(result)
    return model, build_paper_preview_read_model_boundary_matrix(model)


def test_read_model_boundary_matrix_exists_for_local_read_model() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    bundle_matrix = build_local_bundle_boundary_matrix(result.local_bundle)
    read_model = build_paper_preview_bundle_read_model(result.local_bundle, bundle_matrix)

    report = build_paper_preview_read_model_boundary_matrix(read_model)

    assert report.report_kind == "local_read_model_boundary_refusal_matrix"
    assert report.row_count == len(PaperPreviewReadModelBoundary)
    assert len(report.rows) == len(PaperPreviewReadModelBoundary)
    assert report.all_refused is True


def test_read_model_boundary_matrix_contains_all_boundaries_once_in_order() -> None:
    _, report = _read_model_boundary_report_for_result(_local_bundle_result())
    expected = tuple(boundary.value for boundary in PaperPreviewReadModelBoundary)

    assert tuple(row.boundary_kind for row in report.rows) == expected
    assert {row.boundary_kind for row in report.rows} == set(expected)
    assert len({row.boundary_kind for row in report.rows}) == len(expected)
    assert expected == (
        "qml_binding",
        "pyside_bridge",
        "ui_runtime_binding",
        "app_runtime_loop",
        "controller_handoff",
        "trading_controller_handoff",
        "decision_envelope_handoff",
        "strategy_engine_handoff",
        "ai_model_inference_handoff",
        "scoring_handoff",
        "recommendation_handoff",
        "order_generation_handoff",
        "file_export",
        "serialized_export",
        "cloud_sink",
        "external_export",
    )


def test_read_model_boundary_matrix_rows_mirror_read_model_flags() -> None:
    model, report = _read_model_boundary_report_for_result(_local_bundle_result())

    assert report.model_kind == model.model_kind
    assert report.scenario_name == model.scenario_name
    assert report.read_only is True
    assert report.runtime_backed is False
    assert report.ui_bound is False
    assert report.generated_order_count == 0
    assert report.generated_decision_count == 0
    assert report.export_sink == "none"
    assert report.cloud_sink == "none"
    assert report.external_export is False
    for row in report.rows:
        assert row.model_kind == model.model_kind
        assert row.scenario_name == model.scenario_name
        assert row.read_only is True
        assert row.runtime_backed is False
        assert row.ui_bound is False
        assert row.generated_order_count == 0
        assert row.generated_decision_count == 0
        assert row.export_sink == "none"
        assert row.cloud_sink == "none"
        assert row.external_export is False


def test_read_model_boundary_matrix_refuses_ui_runtime_engine_and_export_boundaries() -> None:
    _, report = _read_model_boundary_report_for_result(_local_bundle_result())
    rows = {row.boundary_kind: row for row in report.rows}

    for boundary in (
        "qml_binding",
        "pyside_bridge",
        "ui_runtime_binding",
        "app_runtime_loop",
        "controller_handoff",
        "strategy_engine_handoff",
        "ai_model_inference_handoff",
        "scoring_handoff",
        "recommendation_handoff",
        "decision_envelope_handoff",
        "trading_controller_handoff",
        "order_generation_handoff",
        "file_export",
        "serialized_export",
        "cloud_sink",
        "external_export",
    ):
        assert rows[boundary].refused is True
        assert rows[boundary].reason == f"local read model boundary refuses {boundary}"


def test_read_model_boundary_matrix_has_no_file_network_serialization_ui_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _read_model_boundary_report_for_result(_local_bundle_result())
    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(token in mode for token in ("w", "a", "+", "x")):
            raise AssertionError("read model boundary matrix must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket used")),
    )

    report = build_paper_preview_read_model_boundary_matrix(model)

    for forbidden in (
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
        "runtime_handle",
    ):
        assert not hasattr(report, forbidden)
        assert all(not hasattr(row, forbidden) for row in report.rows)


def test_read_model_boundary_matrix_is_deterministic_and_immutable() -> None:
    model, first = _read_model_boundary_report_for_result(_local_bundle_result())
    second = build_paper_preview_read_model_boundary_matrix(model)

    assert first == second
    assert isinstance(first.rows, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.ui_bound = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.rows[0] = first.rows[0]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.rows[0].ui_bound = True  # type: ignore[misc]


def test_read_model_boundary_matrix_does_not_affect_paper_flow() -> None:
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    result = runner.run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="read-boundary-stable",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
            ),
            PaperPreviewScenarioStep(
                action="fill", order_id="read-boundary-stable", fill_price=100
            ),
            name="read-boundary-stable",
        )
    )
    assert result.local_bundle is not None
    before = _snapshot_counts(runner)
    model, _ = _read_model_for_result(result)

    report = build_paper_preview_read_model_boundary_matrix(model)

    assert report.row_count == len(PaperPreviewReadModelBoundary)
    assert _snapshot_counts(runner) == before
    assert before == (
        result.summary.order_event_count,
        result.summary.trade_count,
        result.summary.audit_event_count,
    )


def test_read_model_boundary_matrix_has_no_forbidden_surface() -> None:
    model, report = _read_model_boundary_report_for_result(_local_bundle_result())
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    forbidden = {
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "start_runtime",
        "run_loop",
        "connect_signal",
        "emit_signal",
        "create_controller",
        "serialize_for_ui",
        "qml",
        "qml_object",
        "QObject",
        "signal",
        "slot",
        "runtime_handle",
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }
    for item in (model, report, *report.rows, runner):
        for name in forbidden:
            assert not hasattr(item, name)


def _ui_runtime_preflight_for_result(result):
    model, _ = _read_model_for_result(result)
    matrix = build_paper_preview_read_model_boundary_matrix(model)
    return model, matrix, build_paper_preview_ui_runtime_preflight(model, matrix)


def test_ui_runtime_preflight_exists_for_read_model_and_boundary_matrix() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    bundle_matrix = build_local_bundle_boundary_matrix(result.local_bundle)
    read_model = build_paper_preview_bundle_read_model(result.local_bundle, bundle_matrix)
    read_model_matrix = build_paper_preview_read_model_boundary_matrix(read_model)

    report = build_paper_preview_ui_runtime_preflight(read_model, read_model_matrix)

    assert report.report_kind == "local_preview_ui_runtime_preflight"
    assert report.scenario_name == read_model.scenario_name == read_model_matrix.scenario_name
    assert report.model_kind == read_model.model_kind == read_model_matrix.model_kind


def test_ui_runtime_preflight_positive_local_static_checks() -> None:
    _, _, report = _ui_runtime_preflight_for_result(_local_bundle_result())
    checks = {check.check_name: check for check in report.checks}

    for name in (
        "local_read_model_present",
        "read_model_is_read_only",
        "read_model_not_runtime_backed",
        "read_model_not_ui_bound",
        "read_model_boundary_matrix_present",
        "all_read_model_boundaries_refused",
    ):
        assert checks[name].passed is True
        assert checks[name].blocking is False


def test_ui_runtime_preflight_blocking_missing_integration_checks() -> None:
    _, _, report = _ui_runtime_preflight_for_result(_local_bundle_result())
    checks = {check.check_name: check for check in report.checks}

    for name in (
        "qml_binding_missing",
        "pyside_bridge_missing",
        "app_runtime_loop_missing",
        "controller_handoff_missing",
        "trading_controller_handoff_missing",
        "decision_envelope_handoff_missing",
        "strategy_engine_missing",
        "ai_model_inference_missing",
        "scoring_missing",
        "recommendation_missing",
        "order_generation_missing",
        "real_market_adapter_missing",
        "testnet_sandbox_adapter_missing",
        "export_sink_missing",
        "cloud_sink_missing",
        "serialization_export_missing",
    ):
        assert checks[name].passed is False
        assert checks[name].blocking is True


def test_ui_runtime_preflight_readiness_flags_and_safety_mirror() -> None:
    model, _, report = _ui_runtime_preflight_for_result(_local_bundle_result())

    assert report.ready_for_ui_binding is False
    assert report.ready_for_runtime_loop is False
    assert report.ready_for_controller_handoff is False
    assert report.ready_for_decision_engine is False
    assert report.ready_for_export is False
    assert report.blocking_check_count > 0
    assert report.read_only is model.read_only is True
    assert report.runtime_backed is model.runtime_backed is False
    assert report.ui_bound is model.ui_bound is False
    assert report.export_sink == model.export_sink == "none"
    assert report.cloud_sink == model.cloud_sink == "none"
    assert report.external_export is model.external_export is False
    assert report.generated_order_count == model.generated_order_count == 0
    assert report.generated_decision_count == model.generated_decision_count == 0


def test_ui_runtime_preflight_consistency_fails_closed() -> None:
    model, matrix, _ = _ui_runtime_preflight_for_result(_local_bundle_result())

    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="scenario_name"):
        build_paper_preview_ui_runtime_preflight(
            model, dataclasses.replace(matrix, scenario_name="other")
        )
    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="model_kind"):
        build_paper_preview_ui_runtime_preflight(model, dataclasses.replace(matrix, model_kind="x"))
    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="refuse all"):
        build_paper_preview_ui_runtime_preflight(
            model, dataclasses.replace(matrix, all_refused=False)
        )
    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="row_count"):
        build_paper_preview_ui_runtime_preflight(
            model, dataclasses.replace(matrix, row_count=matrix.row_count + 1)
        )
    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="runtime-backed"):
        build_paper_preview_ui_runtime_preflight(
            dataclasses.replace(model, runtime_backed=True), matrix
        )
    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="unbound"):
        build_paper_preview_ui_runtime_preflight(dataclasses.replace(model, ui_bound=True), matrix)
    with pytest.raises(PaperPreviewUiRuntimePreflightError, match="read-only"):
        build_paper_preview_ui_runtime_preflight(
            dataclasses.replace(model, read_only=False), matrix
        )


def test_ui_runtime_preflight_has_no_file_network_serialization_or_ui_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, _ = _ui_runtime_preflight_for_result(_local_bundle_result())
    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(token in mode for token in ("w", "a", "+", "x")):
            raise AssertionError("preflight must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket used")),
    )

    report = build_paper_preview_ui_runtime_preflight(model, matrix)

    for forbidden in (
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
        "runtime_handle",
    ):
        assert not hasattr(report, forbidden)
        assert all(not hasattr(check, forbidden) for check in report.checks)


def test_ui_runtime_preflight_is_deterministic_and_immutable() -> None:
    model, matrix, first = _ui_runtime_preflight_for_result(_local_bundle_result())
    second = build_paper_preview_ui_runtime_preflight(model, matrix)

    assert first == second
    assert isinstance(first.checks, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.ready_for_ui_binding = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.checks[0] = first.checks[0]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.checks[0].passed = False  # type: ignore[misc]


def test_ui_runtime_preflight_does_not_affect_paper_flow() -> None:
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    result = runner.run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="preflight-stable",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
            ),
            PaperPreviewScenarioStep(action="fill", order_id="preflight-stable", fill_price=100),
            name="preflight-stable",
        )
    )
    assert result.local_bundle is not None
    before = _snapshot_counts(runner)
    model, _ = _read_model_for_result(result)
    matrix = build_paper_preview_read_model_boundary_matrix(model)

    report = build_paper_preview_ui_runtime_preflight(model, matrix)

    assert report.check_count == len(report.checks)
    assert _snapshot_counts(runner) == before
    assert before == (
        result.summary.order_event_count,
        result.summary.trade_count,
        result.summary.audit_event_count,
    )


def test_ui_runtime_preflight_has_no_forbidden_surface() -> None:
    model, matrix, report = _ui_runtime_preflight_for_result(_local_bundle_result())
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    forbidden = {
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "start_runtime",
        "run_loop",
        "connect_signal",
        "emit_signal",
        "create_controller",
        "serialize_for_ui",
        "qml",
        "qml_object",
        "QObject",
        "signal",
        "slot",
        "runtime_handle",
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }
    for item in (report, *report.checks, model, matrix, runner):
        for name in forbidden:
            assert not hasattr(item, name)


def test_ui_runtime_preflight_preview_policy_keeps_live_capabilities_blocked() -> None:
    policy = build_preview_mode_policy(
        PreviewMode.PAPER,
        (
            RuntimeCapability.READ_ONLY_MARKET_FETCH,
            RuntimeCapability.PAPER_ORDER_SUBMIT,
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        ),
    )

    assert RuntimeCapability.READ_ONLY_MARKET_FETCH in policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in policy.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


def _integration_gate_for_result(result):
    model, matrix, preflight = _ui_runtime_preflight_for_result(result)
    return model, matrix, preflight, build_paper_preview_integration_readiness_gate(preflight)


def test_integration_readiness_gate_exists_for_preflight_chain() -> None:
    result = _local_bundle_result()
    assert result.local_bundle is not None
    bundle_matrix = build_local_bundle_boundary_matrix(result.local_bundle)
    assert bundle_matrix.report_kind == "local_bundle_boundary_refusal_matrix"
    read_model = build_paper_preview_bundle_read_model(result.local_bundle, bundle_matrix)
    read_model_matrix = build_paper_preview_read_model_boundary_matrix(read_model)
    preflight = build_paper_preview_ui_runtime_preflight(read_model, read_model_matrix)

    gate = build_paper_preview_integration_readiness_gate(preflight)

    assert gate.gate_kind == "local_preview_integration_readiness_gate"
    assert gate.scenario_name == preflight.scenario_name == read_model.scenario_name
    assert gate.model_kind == preflight.model_kind == read_model.model_kind


def test_integration_readiness_gate_is_blocked_while_preflight_blocks() -> None:
    _, _, preflight, gate = _integration_gate_for_result(_local_bundle_result())

    assert preflight.blocking_check_count > 0
    assert gate.status == "blocked"
    assert gate.ready_for_next_block is False
    assert gate.ready_for_ui_runtime_integration is False
    assert gate.ready_for_ui_binding is False
    assert gate.ready_for_runtime_loop is False
    assert gate.ready_for_controller_handoff is False
    assert gate.ready_for_decision_engine is False
    assert gate.ready_for_export is False
    assert gate.blocking_check_count > 0
    for name in (
        "qml_binding_missing",
        "app_runtime_loop_missing",
        "controller_handoff_missing",
        "decision_envelope_handoff_missing",
        "real_market_adapter_missing",
        "testnet_sandbox_adapter_missing",
    ):
        assert name in gate.blocking_items


def test_integration_readiness_gate_checklist_mirrors_preflight_checks() -> None:
    _, _, preflight, gate = _integration_gate_for_result(_local_bundle_result())

    assert gate.check_count == preflight.check_count
    assert len(gate.checklist) == preflight.check_count
    for item, check in zip(gate.checklist, preflight.checks, strict=True):
        assert item.item_name == check.check_name
        assert item.source_check == check.check_name
        assert item.passed is check.passed
        assert item.blocking is check.blocking
        assert item.required_for == check.required_for
        assert item.reason == check.reason


def test_integration_readiness_gate_mirrors_safety_flags() -> None:
    _, _, _, gate = _integration_gate_for_result(_local_bundle_result())

    assert gate.read_only is True
    assert gate.runtime_backed is False
    assert gate.ui_bound is False
    assert gate.export_sink == "none"
    assert gate.cloud_sink == "none"
    assert gate.external_export is False
    assert gate.generated_order_count == 0
    assert gate.generated_decision_count == 0
    assert gate.all_boundaries_refused is True


def test_integration_readiness_gate_consistency_fails_closed() -> None:
    _, _, preflight, _ = _integration_gate_for_result(_local_bundle_result())

    replacements = (
        ("report_kind", "wrong", "report_kind"),
        ("check_count", preflight.check_count + 1, "check_count"),
        ("blocking_check_count", preflight.blocking_check_count + 1, "blocking_check_count"),
        ("read_only", False, "read-only"),
        ("runtime_backed", True, "runtime-backed"),
        ("ui_bound", True, "unbound"),
        ("ready_for_ui_binding", True, "UI binding"),
        ("ready_for_runtime_loop", True, "runtime-loop"),
        ("ready_for_controller_handoff", True, "controller-handoff"),
        ("ready_for_decision_engine", True, "decision-engine"),
        ("ready_for_export", True, "export readiness"),
        ("generated_order_count", 1, "generated_order_count"),
        ("generated_decision_count", 1, "generated_decision_count"),
        ("export_sink", "file", "export_sink"),
        ("cloud_sink", "cloud", "cloud_sink"),
        ("external_export", True, "external_export"),
    )
    for field_name, value, message in replacements:
        with pytest.raises(PaperPreviewIntegrationReadinessGateError, match=message):
            build_paper_preview_integration_readiness_gate(
                dataclasses.replace(preflight, **{field_name: value})
            )


def test_integration_readiness_gate_has_no_file_network_serialization_or_ui_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, preflight, _ = _integration_gate_for_result(_local_bundle_result())
    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(token in mode for token in ("w", "a", "+", "x")):
            raise AssertionError("integration gate must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket used")),
    )
    gate = build_paper_preview_integration_readiness_gate(preflight)

    for forbidden in (
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
        "runtime_handle",
    ):
        assert not hasattr(gate, forbidden)
        assert all(not hasattr(item, forbidden) for item in gate.checklist)


def test_integration_readiness_gate_is_deterministic_and_immutable() -> None:
    _, _, preflight, first = _integration_gate_for_result(_local_bundle_result())
    second = build_paper_preview_integration_readiness_gate(preflight)

    assert first == second
    assert isinstance(first.checklist, tuple)
    assert isinstance(first.blocking_items, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.ready_for_next_block = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.checklist[0] = first.checklist[0]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.checklist[0].passed = False  # type: ignore[misc]


def test_integration_readiness_gate_does_not_affect_paper_flow() -> None:
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    result = runner.run(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="gate-stable", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="gate-stable", fill_price=100),
            name="gate-stable",
        )
    )
    before = _snapshot_counts(runner)
    _, _, preflight = _ui_runtime_preflight_for_result(result)

    gate = build_paper_preview_integration_readiness_gate(preflight)

    assert gate.check_count == len(gate.checklist)
    assert _snapshot_counts(runner) == before
    assert before == (
        result.summary.order_event_count,
        result.summary.trade_count,
        result.summary.audit_event_count,
    )


def _assert_gate_surface_absent(forbidden: set[str]) -> None:
    model, _, preflight, gate = _integration_gate_for_result(_local_bundle_result())
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    for item in (gate, *gate.checklist, preflight, model, runner):
        for name in forbidden:
            assert not hasattr(item, name)


def test_integration_readiness_gate_has_no_ui_runtime_surface() -> None:
    _assert_gate_surface_absent(
        {
            "bind_qml",
            "bind_pyside",
            "attach_ui",
            "start_runtime",
            "run_loop",
            "connect_signal",
            "emit_signal",
            "create_controller",
            "serialize_for_ui",
            "qml",
            "qml_object",
            "QObject",
            "signal",
            "slot",
            "runtime_handle",
        }
    )


def test_integration_readiness_gate_has_no_decision_scoring_recommendation_surface() -> None:
    _assert_gate_surface_absent(
        {
            "decide",
            "evaluate_strategy",
            "score",
            "recommend",
            "recommendation",
            "confidence",
            "generate_order",
            "order_intent",
            "execute",
            "infer",
            "predict",
            "serialize_for_engine",
            "to_json",
            "to_yaml",
            "to_csv",
        }
    )


def test_integration_readiness_gate_has_no_account_live_secret_export_surface() -> None:
    _assert_gate_surface_absent(
        {
            "get_balance",
            "get_account",
            "get_account_snapshot",
            "get_positions_from_exchange",
            "get_open_orders",
            "read_credentials",
            "account_balance",
            "metadata",
            "api_key",
            "secret",
            "password",
            "passphrase",
            "credential",
            "credentials",
            "token",
            "private_key",
            "export_path",
            "file_path",
            "cloud_url",
        }
    )


def test_integration_readiness_gate_preview_policy_keeps_live_capabilities_blocked() -> None:
    policy = build_preview_mode_policy(
        PreviewMode.PAPER,
        (
            RuntimeCapability.READ_ONLY_MARKET_FETCH,
            RuntimeCapability.PAPER_ORDER_SUBMIT,
            RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        ),
    )

    assert RuntimeCapability.READ_ONLY_MARKET_FETCH in policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in policy.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


from bot_core.runtime.paper_preview_runtime_service import (
    PaperPreviewRuntimeService,
    PaperPreviewRuntimeServiceError,
    PaperPreviewRuntimeServiceSnapshot,
    run_paper_preview_runtime_service_once,
)


def _service_scenario(name: str = "runtime-service") -> PaperPreviewScenario:
    return _scenario(
        PaperPreviewScenarioStep(
            action="submit", order_id=f"{name}-buy", symbol="BTCUSDT", side="buy", quantity=1
        ),
        PaperPreviewScenarioStep(action="fill", order_id=f"{name}-buy", fill_price=100),
        name=name,
    )


def test_local_runtime_service_wrapper_exists_and_builds_full_contract_chain() -> None:
    result = run_paper_preview_runtime_service_once(_service_scenario(), created_at="fixed")

    assert isinstance(result, PaperPreviewRuntimeServiceSnapshot)
    assert result.service_kind == "local_paper_preview_runtime_service"
    assert result.scenario_name == "runtime-service"
    assert result.mode == "paper"
    assert result.single_shot is True
    assert result.runtime_loop_started is False
    assert result.ui_bound is False
    assert result.runtime_backed is False
    assert result.paper_only is True
    assert result.read_only is True
    assert result.scenario_result is not None
    assert result.local_bundle_present is True
    assert result.bundle_boundary_matrix_present is True
    assert result.read_model_present is True
    assert result.read_model_boundary_matrix_present is True
    assert result.preflight_present is True
    assert result.integration_gate_present is True
    assert result.integration_gate_status == "blocked"
    assert result.ready_for_ui_runtime_integration is False
    assert result.ready_for_decision_engine is False
    assert result.ready_for_export is False
    assert result.generated_order_count == 0
    assert result.generated_decision_count == 0
    assert result.export_sink == "none"
    assert result.cloud_sink == "none"
    assert result.external_export is False


def test_local_runtime_service_wrapper_preserves_paper_flow_summary() -> None:
    result = run_paper_preview_runtime_service_once(
        _service_scenario("summary"), created_at="fixed"
    )

    assert result.order_event_count == result.scenario_result.summary.order_event_count == 2
    assert result.trade_count == result.scenario_result.summary.trade_count == 1
    assert result.audit_event_count == result.scenario_result.summary.audit_event_count == 3
    assert result.position_count == result.scenario_result.summary.position_count == 1
    assert result.has_market_context is False
    assert result.market_symbols == ()
    assert result.read_model_kind == "local_preview_bundle_read_model"
    assert result.gate_kind == "local_preview_integration_readiness_gate"
    assert result.preflight_report_kind == "local_preview_ui_runtime_preflight"
    assert result.blocking_check_count == len(result.blocking_items) > 0


def test_local_runtime_service_wrapper_handles_market_and_no_market_context() -> None:
    market_result = PaperPreviewRuntimeService(
        market_data_provider=_market_provider(), created_at="fixed"
    ).run_once(
        PaperPreviewScenario(
            name="market-service",
            market_symbols=("ETHUSDT", "BTCUSDT"),
            market_timeframe="1m",
            market_candle_limit=1,
            steps=_service_scenario("market-service").steps,
        )
    )
    no_market_result = run_paper_preview_runtime_service_once(
        _service_scenario("no-market-service"), created_at="fixed"
    )

    assert market_result.has_market_context is True
    assert market_result.market_symbols == ("BTCUSDT", "ETHUSDT")
    assert no_market_result.has_market_context is False
    assert no_market_result.market_symbols == ()


class _NoBundleRunner(PaperPreviewScenarioRunner):
    def run(self, scenario: PaperPreviewScenario):  # type: ignore[override]
        result = super().run(scenario)
        return dataclasses.replace(result, local_bundle=None)


def test_local_runtime_service_wrapper_fails_closed_for_missing_local_bundle() -> None:
    service = PaperPreviewRuntimeService(
        created_at="fixed",
        scenario_runner_factory=lambda **kwargs: _NoBundleRunner(**kwargs),
    )

    with pytest.raises(PaperPreviewRuntimeServiceError, match="local bundle"):
        service.run_once(_service_scenario("missing-bundle"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "passed", "blocked"),
        ("ready_for_ui_runtime_integration", True, "UI/runtime"),
        ("ready_for_decision_engine", True, "decision-engine"),
        ("ready_for_export", True, "export"),
        ("generated_order_count", 1, "generated_order_count"),
        ("generated_decision_count", 1, "generated_decision_count"),
        ("export_sink", "file", "export_sink"),
        ("cloud_sink", "prod", "cloud_sink"),
        ("external_export", True, "external_export"),
    ],
)
def test_local_runtime_service_wrapper_fails_closed_for_unsafe_gate_markers(
    field: str, value: object, message: str
) -> None:
    def unsafe_gate(preflight):
        gate = build_paper_preview_integration_readiness_gate(preflight)
        return dataclasses.replace(gate, **{field: value})

    service = PaperPreviewRuntimeService(created_at="fixed", gate_builder=unsafe_gate)

    with pytest.raises(PaperPreviewRuntimeServiceError, match=message):
        service.run_once(_service_scenario(f"unsafe-{field}"))


def test_local_runtime_service_wrapper_has_no_runtime_loop_or_background_surface() -> None:
    result = run_paper_preview_runtime_service_once(
        _service_scenario("surface"), created_at="fixed"
    )
    forbidden = {
        "start",
        "start_loop",
        "run_loop",
        "stop_loop",
        "schedule",
        "worker",
        "thread",
        "timer",
        "async_task",
        "runtime_handle",
    }

    assert forbidden.isdisjoint(set(dir(PaperPreviewRuntimeService())))
    assert forbidden.isdisjoint(set(dir(result)))


def test_local_runtime_service_wrapper_has_no_file_network_serialization_or_ui_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(token in mode for token in ("w", "a", "+", "x")):
            raise AssertionError("unexpected write")
        return real_open(file, mode, *args, **kwargs)

    def fail_network(*args, **kwargs):
        raise AssertionError("unexpected network")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket, "socket", fail_network)

    result = run_paper_preview_runtime_service_once(_service_scenario("no-io"), created_at="fixed")
    forbidden = {
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
    }

    assert forbidden.isdisjoint(set(dir(result)))


def test_local_runtime_service_wrapper_is_deterministic_and_immutable() -> None:
    scenario = _service_scenario("det-service")
    first = run_paper_preview_runtime_service_once(scenario, created_at="fixed")
    second = run_paper_preview_runtime_service_once(scenario, created_at="fixed")

    assert dataclasses.is_dataclass(first)
    assert first.__dataclass_params__.frozen is True
    assert first.market_symbols == second.market_symbols
    assert first.blocking_items == second.blocking_items
    assert (
        first.service_kind,
        first.scenario_name,
        first.integration_gate_status,
        first.order_event_count,
        first.trade_count,
        first.audit_event_count,
        first.position_count,
        first.generated_order_count,
        first.generated_decision_count,
        first.export_sink,
        first.cloud_sink,
        first.external_export,
    ) == (
        second.service_kind,
        second.scenario_name,
        second.integration_gate_status,
        second.order_event_count,
        second.trade_count,
        second.audit_event_count,
        second.position_count,
        second.generated_order_count,
        second.generated_decision_count,
        second.export_sink,
        second.cloud_sink,
        second.external_export,
    )
    with pytest.raises(AttributeError):
        first.service_kind = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first.market_symbols.append("ETHUSDT")  # type: ignore[attr-defined]


def test_local_runtime_service_wrapper_keeps_forbidden_surfaces_absent() -> None:
    result = run_paper_preview_runtime_service_once(
        _service_scenario("forbidden"), created_at="fixed"
    )
    runner = PaperPreviewScenarioRunner(created_at="fixed")
    chain_objects = (
        PaperPreviewRuntimeService(),
        result,
        runner,
    )
    forbidden = {
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "start_runtime",
        "run_loop",
        "connect_signal",
        "emit_signal",
        "create_controller",
        "serialize_for_ui",
        "qml",
        "qml_object",
        "QObject",
        "signal",
        "slot",
        "runtime_handle",
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }

    for obj in chain_objects:
        assert forbidden.isdisjoint(set(dir(obj)))


def test_local_runtime_service_wrapper_preview_policy_blocks_live_capabilities() -> None:
    assert (
        RuntimeCapability.READ_ONLY_MARKET_FETCH
        in build_preview_mode_policy(
            PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
        ).capabilities
    )
    paper_policy = PaperPreviewScenarioRunner(created_at="fixed").policy
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in paper_policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in paper_policy.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


from bot_core.runtime.paper_preview_runtime_service import (
    PaperPreviewRuntimeService,
    run_paper_preview_runtime_service_once,
)
from bot_core.runtime.paper_preview_runtime_service_boundary import (
    PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES,
    PaperPreviewRuntimeServiceBoundaryError,
    build_paper_preview_runtime_service_boundary_matrix,
)


_RUNTIME_SERVICE_BOUNDARY_ORDER = (
    "app_runtime_loop",
    "scheduler_loop",
    "worker_loop",
    "background_thread",
    "background_timer",
    "async_task",
    "qml_binding",
    "pyside_bridge",
    "ui_runtime_binding",
    "controller_handoff",
    "trading_controller_handoff",
    "decision_envelope_handoff",
    "strategy_engine_handoff",
    "ai_model_inference_handoff",
    "scoring_handoff",
    "recommendation_handoff",
    "order_generation_handoff",
    "real_market_adapter",
    "testnet_sandbox_adapter",
    "live_exchange_io",
    "account_balance_fetch",
    "live_credentials_read",
    "file_export",
    "serialized_export",
    "cloud_sink",
    "external_export",
)


def _runtime_service_snapshot():
    return run_paper_preview_runtime_service_once(
        _scenario(
            PaperPreviewScenarioStep(
                action="submit", order_id="boundary-buy", symbol="BTCUSDT", side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id="boundary-buy", fill_price=100),
            name="runtime-service-boundary",
        ),
        created_at="fixed",
    )


def test_runtime_service_boundary_matrix_exists_and_contains_all_boundaries_in_order() -> None:
    snapshot = _runtime_service_snapshot()
    report = build_paper_preview_runtime_service_boundary_matrix(snapshot)

    assert report.report_kind == "local_runtime_service_boundary_no_loop_matrix"
    assert report.row_count == len(PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES)
    assert report.all_refused is True
    assert tuple(row.boundary_kind for row in report.rows) == _RUNTIME_SERVICE_BOUNDARY_ORDER
    assert PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES == _RUNTIME_SERVICE_BOUNDARY_ORDER
    assert len(set(row.boundary_kind for row in report.rows)) == report.row_count


def test_runtime_service_boundary_rows_mirror_service_safety_flags() -> None:
    snapshot = _runtime_service_snapshot()
    report = build_paper_preview_runtime_service_boundary_matrix(snapshot)

    assert report.service_kind == snapshot.service_kind
    assert report.scenario_name == snapshot.scenario_name
    assert report.single_shot is True
    assert report.runtime_loop_started is False
    assert report.runtime_backed is False
    assert report.ui_bound is False
    assert report.read_only is True
    assert report.paper_only is True
    assert report.integration_gate_status == "blocked"
    assert report.ready_for_ui_runtime_integration is False
    assert report.ready_for_decision_engine is False
    assert report.ready_for_export is False
    assert report.generated_order_count == 0
    assert report.generated_decision_count == 0
    assert report.export_sink == "none"
    assert report.cloud_sink == "none"
    assert report.external_export is False
    for row in report.rows:
        assert row.refused is True
        assert row.service_kind == snapshot.service_kind
        assert row.scenario_name == snapshot.scenario_name
        assert row.single_shot is True
        assert row.runtime_loop_started is False
        assert row.runtime_backed is False
        assert row.ui_bound is False
        assert row.read_only is True
        assert row.paper_only is True
        assert row.integration_gate_status == "blocked"
        assert row.ready_for_ui_runtime_integration is False
        assert row.ready_for_decision_engine is False
        assert row.ready_for_export is False
        assert row.generated_order_count == 0
        assert row.generated_decision_count == 0
        assert row.export_sink == "none"
        assert row.cloud_sink == "none"
        assert row.external_export is False


@pytest.mark.parametrize("boundary", _RUNTIME_SERVICE_BOUNDARY_ORDER[:6])
def test_runtime_service_boundary_refuses_lifecycle_boundaries(boundary: str) -> None:
    report = build_paper_preview_runtime_service_boundary_matrix(_runtime_service_snapshot())
    rows = {row.boundary_kind: row for row in report.rows}
    assert rows[boundary].refused is True


@pytest.mark.parametrize("boundary", _RUNTIME_SERVICE_BOUNDARY_ORDER[6:11])
def test_runtime_service_boundary_refuses_ui_runtime_controller_boundaries(boundary: str) -> None:
    report = build_paper_preview_runtime_service_boundary_matrix(_runtime_service_snapshot())
    rows = {row.boundary_kind: row for row in report.rows}
    assert rows[boundary].refused is True


@pytest.mark.parametrize("boundary", _RUNTIME_SERVICE_BOUNDARY_ORDER[11:17])
def test_runtime_service_boundary_refuses_decision_order_boundaries(boundary: str) -> None:
    report = build_paper_preview_runtime_service_boundary_matrix(_runtime_service_snapshot())
    rows = {row.boundary_kind: row for row in report.rows}
    assert rows[boundary].refused is True


@pytest.mark.parametrize("boundary", _RUNTIME_SERVICE_BOUNDARY_ORDER[17:])
def test_runtime_service_boundary_refuses_adapter_live_export_boundaries(boundary: str) -> None:
    report = build_paper_preview_runtime_service_boundary_matrix(_runtime_service_snapshot())
    rows = {row.boundary_kind: row for row in report.rows}
    assert rows[boundary].refused is True


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("service_kind", "unsafe"),
        ("single_shot", False),
        ("runtime_loop_started", True),
        ("runtime_backed", True),
        ("ui_bound", True),
        ("read_only", False),
        ("paper_only", False),
        ("integration_gate_status", "ready"),
        ("ready_for_ui_runtime_integration", True),
        ("ready_for_decision_engine", True),
        ("ready_for_export", True),
        ("generated_order_count", 1),
        ("generated_decision_count", 1),
        ("export_sink", "file"),
        ("cloud_sink", "prod"),
        ("external_export", True),
    ),
)
def test_runtime_service_boundary_matrix_fails_closed(field: str, unsafe_value: object) -> None:
    snapshot = dataclasses.replace(_runtime_service_snapshot(), **{field: unsafe_value})
    with pytest.raises(PaperPreviewRuntimeServiceBoundaryError, match=field):
        build_paper_preview_runtime_service_boundary_matrix(snapshot)


def test_runtime_service_boundary_matrix_has_no_side_effects_or_forbidden_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _runtime_service_snapshot()

    original_open = builtins.open

    def fail_on_write(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("matrix must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_on_write)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no network")),
    )
    monkeypatch.setattr(
        socket, "socket", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no socket"))
    )

    import threading

    monkeypatch.setattr(
        threading,
        "Thread",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no thread")),
    )
    monkeypatch.setattr(
        threading,
        "Timer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no timer")),
    )

    report = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    forbidden = {
        "start",
        "start_loop",
        "run_loop",
        "stop_loop",
        "schedule",
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
        "runtime_handle",
    }
    assert forbidden.isdisjoint(set(report.__dataclass_fields__))
    for row in report.rows:
        assert forbidden.isdisjoint(set(row.__dataclass_fields__))


def test_runtime_service_boundary_matrix_is_deterministic_and_immutable() -> None:
    snapshot = _runtime_service_snapshot()
    first = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    second = build_paper_preview_runtime_service_boundary_matrix(snapshot)

    assert first == second
    assert isinstance(first.rows, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.row_count = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.rows[0].refused = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.rows[0] = first.rows[0]  # type: ignore[index]


def test_runtime_service_boundary_matrix_does_not_mutate_snapshot_or_paper_flow() -> None:
    snapshot = _runtime_service_snapshot()
    before = (
        snapshot.order_event_count,
        snapshot.trade_count,
        snapshot.audit_event_count,
        snapshot.blocking_items,
        snapshot.scenario_result.summary,
    )

    build_paper_preview_runtime_service_boundary_matrix(snapshot)

    after = (
        snapshot.order_event_count,
        snapshot.trade_count,
        snapshot.audit_event_count,
        snapshot.blocking_items,
        snapshot.scenario_result.summary,
    )
    assert after == before


def test_runtime_service_boundary_matrix_does_not_add_forbidden_surfaces() -> None:
    snapshot = _runtime_service_snapshot()
    service = PaperPreviewRuntimeService(created_at="fixed")
    report = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    objects = (report, report.rows[0], snapshot, service)
    forbidden = {
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "start_runtime",
        "run_loop",
        "connect_signal",
        "emit_signal",
        "create_controller",
        "serialize_for_ui",
        "qml",
        "qml_object",
        "QObject",
        "signal",
        "slot",
        "runtime_handle",
        "start",
        "start_loop",
        "stop_loop",
        "schedule",
        "worker",
        "thread",
        "timer",
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "generate_order",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }
    for obj in objects:
        assert forbidden.isdisjoint(set(dir(obj)))


def test_runtime_service_boundary_preview_policy_keeps_live_capabilities_blocked() -> None:
    policy = PaperPreviewScenarioRunner(created_at="fixed").policy
    read_only_policy = build_preview_mode_policy(
        PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    )
    assert RuntimeCapability.READ_ONLY_MARKET_FETCH in read_only_policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in policy.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


from bot_core.runtime.paper_preview_runtime_service_lifecycle import (
    PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS,
    PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS,
    PaperPreviewRuntimeServiceLifecycleError,
    build_paper_preview_runtime_service_lifecycle_contract,
)

_LIFECYCLE_ALLOWED_COMMANDS = (
    "run_once_local_scenario",
    "read_local_snapshot",
    "inspect_integration_gate",
    "inspect_boundary_matrix",
)
_LIFECYCLE_REFUSED_COMMANDS = (
    "start_runtime_loop",
    "stop_runtime_loop",
    "restart_runtime_loop",
    "schedule_worker",
    "start_worker",
    "start_background_thread",
    "start_background_timer",
    "start_async_task",
    "bind_qml",
    "bind_pyside",
    "attach_ui",
    "controller_handoff",
    "trading_controller_handoff",
    "decision_envelope_handoff",
    "strategy_engine_handoff",
    "ai_model_inference_handoff",
    "scoring_handoff",
    "recommendation_handoff",
    "generate_order",
    "submit_order",
    "real_market_adapter_handoff",
    "testnet_sandbox_adapter_handoff",
    "live_exchange_io",
    "account_balance_fetch",
    "live_credentials_read",
    "file_export",
    "serialized_export",
    "cloud_sink",
    "external_export",
)


def _runtime_lifecycle_contract():
    snapshot = _runtime_service_snapshot()
    matrix = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    return (
        snapshot,
        matrix,
        build_paper_preview_runtime_service_lifecycle_contract(snapshot, matrix),
    )


def test_runtime_service_lifecycle_command_contract_exists_and_counts() -> None:
    snapshot, matrix, contract = _runtime_lifecycle_contract()

    assert isinstance(snapshot, PaperPreviewRuntimeServiceSnapshot)
    assert matrix.report_kind == "local_runtime_service_boundary_no_loop_matrix"
    assert contract.contract_kind == "local_runtime_service_lifecycle_command_contract"
    assert contract.command_count == contract.allowed_command_count + contract.refused_command_count
    assert contract.command_count == len(contract.command_decisions)


def test_runtime_service_lifecycle_allowed_commands_only_local_static() -> None:
    _, _, contract = _runtime_lifecycle_contract()

    assert contract.allowed_commands == _LIFECYCLE_ALLOWED_COMMANDS
    assert PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS == _LIFECYCLE_ALLOWED_COMMANDS
    decisions = {row.command: row for row in contract.command_decisions}
    forbidden_reason_terms = ("ui", "controller", "decision", "export", "live")
    for command in contract.allowed_commands:
        row = decisions[command]
        assert row.allowed is True
        assert row.refused is False
        assert "local" in row.reason
        assert "static" in row.reason
        assert "single_shot" in row.reason
        assert "loop" not in command
        assert all(term not in command for term in forbidden_reason_terms)


def test_runtime_service_lifecycle_refused_commands_complete() -> None:
    _, _, contract = _runtime_lifecycle_contract()
    assert contract.refused_commands == _LIFECYCLE_REFUSED_COMMANDS
    assert PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS == _LIFECYCLE_REFUSED_COMMANDS
    decisions = {row.command: row for row in contract.command_decisions}
    for command in _LIFECYCLE_REFUSED_COMMANDS:
        assert decisions[command].allowed is False
        assert decisions[command].refused is True


def test_runtime_service_lifecycle_command_order_is_deterministic() -> None:
    _, _, contract = _runtime_lifecycle_contract()
    commands = tuple(row.command for row in contract.command_decisions)
    assert commands == _LIFECYCLE_ALLOWED_COMMANDS + _LIFECYCLE_REFUSED_COMMANDS
    assert len(commands) == len(set(commands))
    assert contract.command_count == len(commands)


def test_runtime_service_lifecycle_decisions_and_report_mirror_safety_flags() -> None:
    snapshot, matrix, contract = _runtime_lifecycle_contract()

    assert contract.allowed_command_count == 4
    assert contract.refused_command_count == len(_LIFECYCLE_REFUSED_COMMANDS)
    assert contract.service_kind == snapshot.service_kind
    assert contract.scenario_name == snapshot.scenario_name
    assert contract.boundary_matrix_report_kind == matrix.report_kind
    rows = (contract, *contract.command_decisions)
    for row in rows:
        assert row.service_kind == snapshot.service_kind
        assert row.scenario_name == snapshot.scenario_name
        assert row.single_shot is True
        assert row.runtime_loop_started is False
        assert row.runtime_backed is False
        assert row.ui_bound is False
        assert row.read_only is True
        assert row.paper_only is True
        assert row.integration_gate_status == "blocked"
        assert row.ready_for_ui_runtime_integration is False
        assert row.ready_for_decision_engine is False
        assert row.ready_for_export is False
        assert row.generated_order_count == 0
        assert row.generated_decision_count == 0
        assert row.export_sink == "none"
        assert row.cloud_sink == "none"
        assert row.external_export is False


@pytest.mark.parametrize(
    ("target", "field", "unsafe_value"),
    (
        ("snapshot", "service_kind", "unsafe"),
        ("matrix", "report_kind", "unsafe"),
        ("matrix", "service_kind", "unsafe"),
        ("matrix", "scenario_name", "unsafe"),
        ("matrix", "all_refused", False),
        ("matrix", "row_count", -1),
        ("snapshot", "single_shot", False),
        ("matrix", "single_shot", False),
        ("snapshot", "runtime_loop_started", True),
        ("matrix", "runtime_loop_started", True),
        ("snapshot", "runtime_backed", True),
        ("matrix", "runtime_backed", True),
        ("snapshot", "ui_bound", True),
        ("matrix", "ui_bound", True),
        ("snapshot", "read_only", False),
        ("matrix", "read_only", False),
        ("snapshot", "paper_only", False),
        ("matrix", "paper_only", False),
        ("snapshot", "integration_gate_status", "ready"),
        ("matrix", "integration_gate_status", "ready"),
        ("snapshot", "ready_for_ui_runtime_integration", True),
        ("matrix", "ready_for_ui_runtime_integration", True),
        ("snapshot", "ready_for_decision_engine", True),
        ("matrix", "ready_for_decision_engine", True),
        ("snapshot", "ready_for_export", True),
        ("matrix", "ready_for_export", True),
        ("snapshot", "generated_order_count", 1),
        ("matrix", "generated_order_count", 1),
        ("snapshot", "generated_decision_count", 1),
        ("matrix", "generated_decision_count", 1),
        ("snapshot", "export_sink", "file"),
        ("matrix", "export_sink", "file"),
        ("snapshot", "cloud_sink", "prod"),
        ("matrix", "cloud_sink", "prod"),
        ("snapshot", "external_export", True),
        ("matrix", "external_export", True),
    ),
)
def test_runtime_service_lifecycle_contract_fails_closed(
    target: str, field: str, unsafe_value: object
) -> None:
    snapshot = _runtime_service_snapshot()
    matrix = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    if target == "snapshot":
        snapshot = dataclasses.replace(snapshot, **{field: unsafe_value})
    else:
        matrix = dataclasses.replace(matrix, **{field: unsafe_value})
    with pytest.raises(PaperPreviewRuntimeServiceLifecycleError, match=field):
        build_paper_preview_runtime_service_lifecycle_contract(snapshot, matrix)


def test_runtime_service_lifecycle_has_no_side_effects_or_forbidden_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _runtime_service_snapshot()
    matrix = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    original_open = builtins.open

    def fail_on_write(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("lifecycle contract must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_on_write)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no network")),
    )
    monkeypatch.setattr(
        socket, "socket", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no socket"))
    )
    import threading

    monkeypatch.setattr(
        threading,
        "Thread",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no thread")),
    )
    monkeypatch.setattr(
        threading,
        "Timer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no timer")),
    )

    contract = build_paper_preview_runtime_service_lifecycle_contract(snapshot, matrix)
    forbidden = {
        "start",
        "start_loop",
        "run_loop",
        "stop_loop",
        "schedule",
        "worker",
        "thread",
        "timer",
        "async_task",
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
        "runtime_handle",
    }
    assert forbidden.isdisjoint(set(contract.__dataclass_fields__))
    for row in contract.command_decisions:
        assert forbidden.isdisjoint(set(row.__dataclass_fields__))


def test_runtime_service_lifecycle_is_deterministic_and_immutable() -> None:
    snapshot = _runtime_service_snapshot()
    matrix = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    first = build_paper_preview_runtime_service_lifecycle_contract(snapshot, matrix)
    second = build_paper_preview_runtime_service_lifecycle_contract(snapshot, matrix)

    assert first == second
    assert isinstance(first.command_decisions, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.command_count = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.command_decisions[0].allowed = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.command_decisions[0] = first.command_decisions[0]  # type: ignore[index]


def test_runtime_service_lifecycle_does_not_mutate_snapshot_or_paper_flow() -> None:
    snapshot = _runtime_service_snapshot()
    before = (
        snapshot.order_event_count,
        snapshot.trade_count,
        snapshot.audit_event_count,
        snapshot.blocking_items,
        snapshot.scenario_result.summary,
    )
    matrix = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    build_paper_preview_runtime_service_lifecycle_contract(snapshot, matrix)
    after = (
        snapshot.order_event_count,
        snapshot.trade_count,
        snapshot.audit_event_count,
        snapshot.blocking_items,
        snapshot.scenario_result.summary,
    )
    assert after == before


def test_runtime_service_lifecycle_does_not_add_forbidden_surfaces() -> None:
    snapshot, _, contract = _runtime_lifecycle_contract()
    service = PaperPreviewRuntimeService(created_at="fixed")
    objects = (contract, contract.command_decisions[0], snapshot, service)
    forbidden = {
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "start_runtime",
        "run_loop",
        "connect_signal",
        "emit_signal",
        "create_controller",
        "serialize_for_ui",
        "qml",
        "qml_object",
        "QObject",
        "signal",
        "slot",
        "runtime_handle",
        "start",
        "start_loop",
        "stop_loop",
        "schedule",
        "worker",
        "thread",
        "timer",
        "async_task",
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }
    for obj in objects:
        assert forbidden.isdisjoint(set(dir(obj)))


def test_runtime_service_lifecycle_preview_policy_keeps_live_capabilities_blocked() -> None:
    policy = PaperPreviewScenarioRunner(created_at="fixed").policy
    read_only_policy = build_preview_mode_policy(
        PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    )
    assert RuntimeCapability.READ_ONLY_MARKET_FETCH in read_only_policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in policy.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))
