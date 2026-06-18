from __future__ import annotations

import builtins
import os
import socket
from pathlib import Path

import pytest

from bot_core.runtime.paper_audit_journal import PaperAuditSeverity
from bot_core.runtime.paper_preview_scenario import (
    PaperPreviewDecisionContext,
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


def test_dry_run_artifact_is_immutable_and_deterministic() -> None:
    scenario = _scenario(
        PaperPreviewScenarioStep(
            action="submit", order_id="det-dry", symbol="BTCUSDT", side="buy", quantity=1
        ),
        PaperPreviewScenarioStep(action="fill", order_id="det-dry", fill_price=100),
        name="det-dry",
    )
    first = PaperPreviewScenarioRunner(created_at="fixed").run(scenario).dry_run_artifact
    second = PaperPreviewScenarioRunner(created_at="fixed").run(scenario).dry_run_artifact

    assert first == second
    assert first is not None
    with pytest.raises(AttributeError):
        first.generated_order_count = 1  # type: ignore[misc]


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
        "cloud_sink",
    }

    assert forbidden.isdisjoint(set(dir(artifact)))
    assert forbidden.isdisjoint(set(dir(PaperPreviewScenarioRunner())))


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
