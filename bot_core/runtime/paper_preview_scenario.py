"""Deterministic in-memory paper preview scenario runner.

The runner drives ``PaperPreviewFlow`` with local scenario objects only. It does
not read scenario files, write exports, start runtime loops, use sockets, fetch
market/account data, read secrets, or send telemetry to cloud/external sinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Sequence

from bot_core.runtime.paper_event_spine import TERMINAL_PAPER_ORDER_STATUSES
from bot_core.runtime.paper_preview_flow import PaperPreviewFlow, PaperPreviewSnapshot
from bot_core.runtime.read_only_market_data import (
    MarketCandle,
    ReadOnlyMarketDataError,
    ReadOnlyMarketDataProvider,
    ReadOnlyMarketDataSnapshot,
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


class PaperPreviewScenarioAction(StrEnum):
    """Actions supported by the local in-memory paper preview scenario runner."""

    SUBMIT = "submit"
    PARTIAL_FILL = "partial_fill"
    FILL = "fill"
    REJECT = "reject"
    CANCEL = "cancel"


class PaperPreviewScenarioError(ValueError):
    """Raised when a local paper preview scenario is invalid or cannot run."""


@dataclass(frozen=True, slots=True)
class PaperPreviewScenarioStep:
    """One deterministic in-memory step for ``PaperPreviewScenarioRunner``."""

    action: str | PaperPreviewScenarioAction
    order_id: str
    symbol: str | None = None
    side: str | None = None
    quantity: float | None = None
    fill_quantity: float | None = None
    fill_price: float | None = None
    reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    label: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class PaperPreviewRiskPlaceholder:
    """Immutable context-only risk placeholder; it does not enforce limits."""

    max_position_notional: float | None = None
    max_order_quantity: float | None = None
    max_daily_loss: float | None = None
    risk_checks_enabled: bool = False
    source: str = "placeholder"


@dataclass(frozen=True, slots=True)
class PaperPreviewScenario:
    """In-memory paper preview scenario definition."""

    name: str
    steps: tuple[PaperPreviewScenarioStep, ...]
    market_symbols: tuple[str, ...] = ()
    market_timeframe: str | None = None
    market_candle_limit: int | None = None
    risk: PaperPreviewRiskPlaceholder | None = None


@dataclass(frozen=True, slots=True)
class PaperPreviewMarketCandles:
    """Immutable read-only candle slice captured for one symbol/timeframe."""

    symbol: str
    timeframe: str
    candles: tuple[MarketCandle, ...]


@dataclass(frozen=True, slots=True)
class PaperPreviewMarketContext:
    """Immutable read-only market context captured before paper execution."""

    snapshot: ReadOnlyMarketDataSnapshot
    candle_sets: tuple[PaperPreviewMarketCandles, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.snapshot.symbols

    @property
    def quotes(self):
        return self.snapshot.quotes


@dataclass(frozen=True, slots=True)
class PaperPreviewScenarioStepResult:
    """Immutable deterministic result of one successfully executed step."""

    step_index: int
    action: PaperPreviewScenarioAction
    order_id: str
    success: bool
    order_event_id: str | None
    trade_count_after: int
    audit_event_count_after: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PaperPreviewScenarioSummary:
    """Small deterministic summary for a completed local paper scenario."""

    scenario_name: str
    step_count: int
    order_event_count: int
    trade_count: int
    position_count: int
    audit_event_count: int
    realized_pnl_total: float
    symbols: tuple[str, ...]
    terminal_order_count: int
    failed_step_count: int


@dataclass(frozen=True, slots=True)
class PaperPreviewDecisionContext:
    """Immutable context-only contract for future paper scenario decisions.

    This object only summarizes completed local paper scenario state. It does not
    evaluate strategies, score opportunities, generate decisions, create orders,
    submit orders, enforce risk, read accounts, read credentials, or export data.
    """

    scenario_name: str
    step_count: int
    market_symbols: tuple[str, ...]
    has_market_context: bool
    trade_count: int
    position_count: int
    audit_event_count: int
    realized_pnl_total: float
    risk: PaperPreviewRiskPlaceholder
    decision_status: str = "context_only"
    generated_order_count: int = 0
    generated_decision_count: int = 0


@dataclass(frozen=True, slots=True)
class PaperPreviewDecisionContextSummary:
    """Small immutable summary proving the decision context is no-action only."""

    decision_status: str
    generated_order_count: int
    generated_decision_count: int


_BLOCKED_ENGINE_INTEGRATIONS = (
    "strategy_engine",
    "ai_model_inference",
    "decision_envelope",
    "trading_controller",
    "order_generation",
)


@dataclass(frozen=True, slots=True)
class PaperPreviewDecisionDryRunArtifact:
    """Immutable context-only dry-run artifact for future decision inputs.

    The artifact is diagnostic only: it mirrors already-built scenario, market,
    paper, and risk context without evaluating strategies, scoring, recommending,
    generating decisions, creating orders, reading accounts/secrets, or exporting.
    """

    scenario_name: str
    step_count: int
    decision_status: str
    market_symbols: tuple[str, ...]
    has_market_context: bool
    trade_count: int
    position_count: int
    audit_event_count: int
    realized_pnl_total: float
    risk_source: str
    risk_checks_enabled: bool
    quote_count: int = 0
    candle_set_count: int = 0
    order_event_count: int = 0
    terminal_order_count: int = 0
    paper_symbols: tuple[str, ...] = ()
    artifact_kind: str = "context_only_dry_run"
    generated_order_count: int = 0
    generated_decision_count: int = 0
    no_action_reason: str = "dry_run_context_only"
    blocked_engine_integrations: tuple[str, ...] = _BLOCKED_ENGINE_INTEGRATIONS


@dataclass(frozen=True, slots=True)
class PaperPreviewScenarioResult:
    """Result returned after running a local paper preview scenario."""

    scenario_name: str
    step_results: tuple[PaperPreviewScenarioStepResult, ...]
    final_snapshot: PaperPreviewSnapshot
    summary: PaperPreviewScenarioSummary
    market_context: PaperPreviewMarketContext | None = None
    decision_context: PaperPreviewDecisionContext | None = None
    dry_run_artifact: PaperPreviewDecisionDryRunArtifact | None = None


class PaperPreviewScenarioRunner:
    """Run deterministic in-memory scenarios against ``PaperPreviewFlow``."""

    def __init__(
        self,
        *,
        flow: PaperPreviewFlow | None = None,
        created_at: str | None = None,
        policy: PreviewModePolicy | None = None,
        market_data_provider: ReadOnlyMarketDataProvider | None = None,
    ) -> None:
        self.policy = policy or build_preview_mode_policy(PreviewMode.PAPER, _REQUIRED_CAPABILITIES)
        if self.policy.mode is not PreviewMode.PAPER:
            raise PreviewModeContractError(
                "PaperPreviewScenarioRunner requires preview mode 'paper'"
            )
        if set(self.policy.capabilities) != set(_REQUIRED_CAPABILITIES):
            raise PreviewModeContractError(
                "PaperPreviewScenarioRunner requires paper preview capabilities"
            )
        self.flow = flow or PaperPreviewFlow(created_at=created_at)
        self.market_data_provider = market_data_provider

    def run(self, scenario: PaperPreviewScenario) -> PaperPreviewScenarioResult:
        """Validate and run all scenario steps sequentially, failing fast on errors."""

        validated_actions = self._preflight_validate(scenario)
        market_context = self._build_market_context(scenario)
        step_results: list[PaperPreviewScenarioStepResult] = []
        for index, (step, action) in enumerate(zip(scenario.steps, validated_actions, strict=True)):
            result = self._execute_step(action, step)
            snapshot = self.flow.snapshot()
            step_results.append(
                PaperPreviewScenarioStepResult(
                    step_index=index,
                    action=action,
                    order_id=step.order_id,
                    success=True,
                    order_event_id=result.order_event.event_id,
                    trade_count_after=len(snapshot.portfolio.trades),
                    audit_event_count_after=len(snapshot.audit_events),
                )
            )
        final_snapshot = self.flow.snapshot()
        summary = self._build_summary(scenario.name, final_snapshot, step_results)
        decision_context = self._build_decision_context(scenario, summary, market_context)
        dry_run_artifact = self._build_dry_run_artifact(
            decision_context, summary, final_snapshot, market_context
        )
        return PaperPreviewScenarioResult(
            scenario_name=scenario.name,
            step_results=tuple(step_results),
            final_snapshot=final_snapshot,
            summary=summary,
            market_context=market_context,
            decision_context=decision_context,
            dry_run_artifact=dry_run_artifact,
        )

    @staticmethod
    def _build_dry_run_artifact(
        decision_context: PaperPreviewDecisionContext,
        summary: PaperPreviewScenarioSummary,
        final_snapshot: PaperPreviewSnapshot,
        market_context: PaperPreviewMarketContext | None,
    ) -> PaperPreviewDecisionDryRunArtifact:
        quote_count = len(market_context.quotes) if market_context is not None else 0
        candle_set_count = len(market_context.candle_sets) if market_context is not None else 0
        return PaperPreviewDecisionDryRunArtifact(
            scenario_name=decision_context.scenario_name,
            step_count=decision_context.step_count,
            decision_status=decision_context.decision_status,
            market_symbols=decision_context.market_symbols,
            has_market_context=decision_context.has_market_context,
            trade_count=decision_context.trade_count,
            position_count=decision_context.position_count,
            audit_event_count=decision_context.audit_event_count,
            realized_pnl_total=decision_context.realized_pnl_total,
            risk_source=decision_context.risk.source,
            risk_checks_enabled=decision_context.risk.risk_checks_enabled,
            quote_count=quote_count,
            candle_set_count=candle_set_count,
            order_event_count=len(final_snapshot.order_events),
            terminal_order_count=summary.terminal_order_count,
            paper_symbols=summary.symbols,
            generated_order_count=decision_context.generated_order_count,
            generated_decision_count=decision_context.generated_decision_count,
        )

    @staticmethod
    def _build_decision_context(
        scenario: PaperPreviewScenario,
        summary: PaperPreviewScenarioSummary,
        market_context: PaperPreviewMarketContext | None,
    ) -> PaperPreviewDecisionContext:
        market_symbols = market_context.symbols if market_context is not None else ()
        return PaperPreviewDecisionContext(
            scenario_name=summary.scenario_name,
            step_count=summary.step_count,
            market_symbols=tuple(sorted(market_symbols)),
            has_market_context=market_context is not None,
            trade_count=summary.trade_count,
            position_count=summary.position_count,
            audit_event_count=summary.audit_event_count,
            realized_pnl_total=summary.realized_pnl_total,
            risk=scenario.risk or PaperPreviewRiskPlaceholder(),
        )

    def _build_market_context(
        self, scenario: PaperPreviewScenario
    ) -> PaperPreviewMarketContext | None:
        symbols = tuple(
            sorted({self._normalize_market_symbol(symbol) for symbol in scenario.market_symbols})
        )
        if not symbols:
            return None
        if self.market_data_provider is None:
            raise PaperPreviewScenarioError("market_data_provider is required for market_symbols")
        try:
            snapshot = self.market_data_provider.snapshot(symbols)
            candle_sets: tuple[PaperPreviewMarketCandles, ...] = ()
            if scenario.market_timeframe is not None and scenario.market_candle_limit is not None:
                timeframe = scenario.market_timeframe.strip()
                candle_sets = tuple(
                    PaperPreviewMarketCandles(
                        symbol=symbol,
                        timeframe=timeframe,
                        candles=self.market_data_provider.get_candles(
                            symbol, timeframe, scenario.market_candle_limit
                        ),
                    )
                    for symbol in snapshot.symbols
                )
        except ReadOnlyMarketDataError as exc:
            raise PaperPreviewScenarioError(str(exc)) from exc
        return PaperPreviewMarketContext(snapshot=snapshot, candle_sets=candle_sets)

    def _validate_market_request(self, scenario: PaperPreviewScenario) -> None:
        for symbol in scenario.market_symbols:
            self._normalize_market_symbol(symbol)
        has_timeframe = scenario.market_timeframe is not None
        has_limit = scenario.market_candle_limit is not None
        if has_timeframe != has_limit:
            raise PaperPreviewScenarioError(
                "market_timeframe and market_candle_limit must be provided together"
            )
        if scenario.market_timeframe is not None and not scenario.market_timeframe.strip():
            raise PaperPreviewScenarioError("market_timeframe must be non-empty")
        if scenario.market_candle_limit is not None and scenario.market_candle_limit <= 0:
            raise PaperPreviewScenarioError("market_candle_limit must be > 0")

    @staticmethod
    def _normalize_market_symbol(symbol: str) -> str:
        normalized = str(symbol).strip().upper()
        if not normalized:
            raise PaperPreviewScenarioError("market symbol must be non-empty")
        return normalized

    def _execute_step(self, action: PaperPreviewScenarioAction, step: PaperPreviewScenarioStep):
        if action is PaperPreviewScenarioAction.SUBMIT:
            return self.flow.submit_order(
                order_id=step.order_id,
                symbol=str(step.symbol),
                side=str(step.side),
                quantity=float(step.quantity),
                metadata=self._safe_metadata(step.metadata),
            )
        if action is PaperPreviewScenarioAction.PARTIAL_FILL:
            return self.flow.partial_fill_order(
                step.order_id,
                fill_quantity=float(step.fill_quantity),
                fill_price=float(step.fill_price),
            )
        if action is PaperPreviewScenarioAction.FILL:
            return self.flow.fill_order(
                step.order_id,
                fill_quantity=step.fill_quantity,
                fill_price=float(step.fill_price),
            )
        if action is PaperPreviewScenarioAction.REJECT:
            return self.flow.reject_order(step.order_id, reason=str(step.reason))
        if action is PaperPreviewScenarioAction.CANCEL:
            return self.flow.cancel_order(step.order_id, reason=step.reason)
        raise PaperPreviewScenarioError(f"Unsupported scenario action: {action!r}")

    def _preflight_validate(
        self, scenario: PaperPreviewScenario
    ) -> tuple[PaperPreviewScenarioAction, ...]:
        self._validate_scenario(scenario)
        self._validate_market_request(scenario)
        return tuple(self._validate_step(step) for step in scenario.steps)

    @staticmethod
    def _validate_scenario(scenario: PaperPreviewScenario) -> None:
        if not scenario.name.strip():
            raise PaperPreviewScenarioError("scenario name is required")
        if not scenario.steps:
            raise PaperPreviewScenarioError("scenario must contain at least one step")

    def _validate_step(self, step: PaperPreviewScenarioStep) -> PaperPreviewScenarioAction:
        action = self._coerce_action(step.action)
        if not step.order_id.strip():
            raise PaperPreviewScenarioError("order_id is required")
        self._safe_metadata(step.metadata)
        if action is PaperPreviewScenarioAction.SUBMIT:
            self._require_text(step.symbol, "symbol")
            self._require_text(step.side, "side")
            self._require_positive(step.quantity, "quantity")
        elif action in {PaperPreviewScenarioAction.PARTIAL_FILL, PaperPreviewScenarioAction.FILL}:
            if action is PaperPreviewScenarioAction.PARTIAL_FILL:
                self._require_positive(step.fill_quantity, "fill_quantity")
            elif step.fill_quantity is not None:
                self._require_positive(step.fill_quantity, "fill_quantity")
            self._require_positive(step.fill_price, "fill_price")
        elif action is PaperPreviewScenarioAction.REJECT:
            self._require_text(step.reason, "reason")
        return action

    @staticmethod
    def _coerce_action(action: str | PaperPreviewScenarioAction) -> PaperPreviewScenarioAction:
        if isinstance(action, PaperPreviewScenarioAction):
            return action
        try:
            return PaperPreviewScenarioAction(str(action).strip().lower())
        except ValueError as exc:
            raise PaperPreviewScenarioError(f"Unsupported scenario action: {action!r}") from exc

    @staticmethod
    def _require_text(value: str | None, field_name: str) -> None:
        if value is None or not value.strip():
            raise PaperPreviewScenarioError(f"{field_name} is required")

    @staticmethod
    def _require_positive(value: float | None, field_name: str) -> None:
        if value is None:
            raise PaperPreviewScenarioError(f"{field_name} is required")
        if value <= 0:
            raise PaperPreviewScenarioError(f"{field_name} must be > 0")

    @staticmethod
    def _safe_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
        unsafe = [str(key) for key in metadata if _metadata_key_is_secret(key)]
        if unsafe:
            raise PaperPreviewScenarioError(
                f"metadata contains forbidden credential-like keys: {', '.join(sorted(unsafe))}"
            )
        return MappingProxyType(dict(metadata))

    @staticmethod
    def _build_summary(
        scenario_name: str,
        snapshot: PaperPreviewSnapshot,
        step_results: Sequence[PaperPreviewScenarioStepResult],
    ) -> PaperPreviewScenarioSummary:
        positions = snapshot.portfolio.positions
        symbols = sorted(
            {event.symbol for event in snapshot.order_events} | {p.symbol for p in positions}
        )
        return PaperPreviewScenarioSummary(
            scenario_name=scenario_name,
            step_count=len(step_results),
            order_event_count=len(snapshot.order_events),
            trade_count=len(snapshot.portfolio.trades),
            position_count=len(positions),
            audit_event_count=len(snapshot.audit_events),
            realized_pnl_total=sum(position.realized_pnl for position in positions),
            symbols=tuple(symbols),
            terminal_order_count=sum(
                1
                for event in snapshot.order_events
                if event.status in TERMINAL_PAPER_ORDER_STATUSES
            ),
            failed_step_count=sum(1 for result in step_results if not result.success),
        )


def _metadata_key_is_secret(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _SECRET_METADATA_TOKENS)


__all__ = [
    "PaperPreviewDecisionContext",
    "PaperPreviewDecisionContextSummary",
    "PaperPreviewDecisionDryRunArtifact",
    "PaperPreviewScenario",
    "PaperPreviewScenarioAction",
    "PaperPreviewScenarioError",
    "PaperPreviewRiskPlaceholder",
    "PaperPreviewMarketCandles",
    "PaperPreviewMarketContext",
    "PaperPreviewScenarioResult",
    "PaperPreviewScenarioRunner",
    "PaperPreviewScenarioStep",
    "PaperPreviewScenarioStepResult",
    "PaperPreviewScenarioSummary",
    "PreviewModePolicy",
]
