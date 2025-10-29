"""Harmonogram wieloportfelowy obsługujący copy trading i rebalancing."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.portfolio.governor import (
    PortfolioAdjustment,
    PortfolioAdvisory,
    PortfolioDecision,
)
from bot_core.strategies.catalog import (
    PresetLicenseState,
    StrategyCatalog,
    StrategyPresetDescriptor,
)

_LOGGER = logging.getLogger(__name__)

AuditLogger = Callable[[Mapping[str, object]], None]
Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class CopyTradingFollowerConfig:
    """Konfiguracja follow-era dla copy tradingu."""

    portfolio_id: str
    scaling: float = 1.0
    risk_multiplier: float = 1.0
    enabled: bool = True
    max_position_value: float | None = None
    allow_partial: bool = True

    def __post_init__(self) -> None:
        if self.scaling <= 0:
            raise ValueError("Scaling factor must be > 0")
        if self.risk_multiplier <= 0:
            raise ValueError("Risk multiplier must be > 0")


@dataclass(slots=True)
class PortfolioBinding:
    """Powiązanie portfela z presetem i fallbackami."""

    portfolio_id: str
    primary_preset: str
    fallback_presets: tuple[str, ...] = ()
    followers: tuple[CopyTradingFollowerConfig, ...] = ()
    rebalance_cooldown: timedelta = timedelta(minutes=5)


@dataclass(slots=True)
class CopyTradeInstruction:
    follower_id: str
    master_id: str
    issued_at: datetime
    scale: float
    risk_multiplier: float
    adjustments: tuple[PortfolioAdjustment, ...]
    advisories: tuple[PortfolioAdvisory, ...]


@dataclass(slots=True)
class RebalanceInstruction:
    portfolio_id: str
    issued_at: datetime
    adjustments: tuple[PortfolioAdjustment, ...]
    advisories: tuple[PortfolioAdvisory, ...]
    reason: str


@dataclass(slots=True)
class SchedulerEvent:
    timestamp: datetime
    level: str
    message: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
        }
        if self.payload:
            data["payload"] = dict(self.payload)
        return data


@dataclass(slots=True)
class PortfolioScheduleResult:
    master_decision: PortfolioDecision
    rebalance: tuple[RebalanceInstruction, ...]
    copy_trades: tuple[CopyTradeInstruction, ...]
    events: tuple[SchedulerEvent, ...]
    active_preset: str


class StrategyHealthMonitor:
    """Minimalny interfejs detekcji stanu strategii."""

    def evaluate(
        self,
        portfolio_id: str,
        decision: PortfolioDecision,
        metadata: Mapping[str, object] | None = None,
    ) -> bool:
        raise NotImplementedError


class _DefaultHealthMonitor(StrategyHealthMonitor):
    def evaluate(
        self,
        portfolio_id: str,
        decision: PortfolioDecision,
        metadata: Mapping[str, object] | None = None,
    ) -> bool:
        if metadata:
            state = metadata.get("strategy_health")
            if isinstance(state, str):
                normalized = state.strip().lower()
                if normalized in {"failed", "error", "offline", "degraded"}:
                    return False
        for advisory in decision.advisories:
            try:
                severity = advisory.severity.lower()
            except AttributeError:
                continue
            if severity in {"error", "critical", "fatal"}:
                return False
        return True


@dataclass(slots=True)
class _PortfolioState:
    config: PortfolioBinding
    current_preset: str
    last_run: datetime | None = None
    last_health_ok: bool = True
    fallback_index: int = 0
    last_failure_reason: str | None = None


class MultiPortfolioScheduler:
    """Zarządza portfelami master/follower i scenariuszami self-healing."""

    def __init__(
        self,
        catalog: StrategyCatalog,
        *,
        audit_logger: AuditLogger | None = None,
        health_monitor: StrategyHealthMonitor | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._catalog = catalog
        self._audit_logger = audit_logger
        self._clock = clock or _default_clock
        self._health_monitor = health_monitor or _DefaultHealthMonitor()
        self._portfolios: MutableMapping[str, _PortfolioState] = {}
        self._followers_by_master: MutableMapping[str, tuple[CopyTradingFollowerConfig, ...]] = {}

    def register_portfolio(self, binding: PortfolioBinding) -> None:
        if binding.portfolio_id in self._portfolios:
            raise ValueError(f"Portfolio {binding.portfolio_id} already registered")
        descriptor = self._catalog.preset(binding.primary_preset)
        self._portfolios[binding.portfolio_id] = _PortfolioState(
            config=binding,
            current_preset=descriptor.preset_id,
        )
        self._followers_by_master[binding.portfolio_id] = binding.followers
        self._emit_event(
            SchedulerEvent(
                timestamp=self._clock(),
                level="info",
                message="portfolio-registered",
                payload={
                    "portfolio_id": binding.portfolio_id,
                    "preset": descriptor.preset_id,
                    "followers": [f.portfolio_id for f in binding.followers],
                },
            )
        )

    def update_followers(
        self, portfolio_id: str, followers: Iterable[CopyTradingFollowerConfig]
    ) -> None:
        state = self._ensure_state(portfolio_id)
        normalized = tuple(followers)
        self._followers_by_master[portfolio_id] = normalized
        state.config = PortfolioBinding(
            portfolio_id=state.config.portfolio_id,
            primary_preset=state.config.primary_preset,
            fallback_presets=state.config.fallback_presets,
            followers=normalized,
            rebalance_cooldown=state.config.rebalance_cooldown,
        )
        self._emit_event(
            SchedulerEvent(
                timestamp=self._clock(),
                level="info",
                message="followers-updated",
                payload={
                    "portfolio_id": portfolio_id,
                    "followers": [f.portfolio_id for f in normalized],
                },
            )
        )

    def portfolio_state(self, portfolio_id: str) -> Mapping[str, object]:
        state = self._ensure_state(portfolio_id)
        return {
            "portfolio_id": portfolio_id,
            "preset": state.current_preset,
            "last_run": state.last_run.isoformat() if state.last_run else None,
            "last_health_ok": state.last_health_ok,
            "fallback_index": state.fallback_index,
            "last_failure_reason": state.last_failure_reason,
            "followers": [f.portfolio_id for f in self._followers_by_master.get(portfolio_id, ())],
        }

    def registered_portfolios(self) -> tuple[str, ...]:
        return tuple(self._portfolios.keys())

    def process_decision(
        self,
        decision: PortfolioDecision,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> PortfolioScheduleResult:
        state = self._ensure_state(decision.portfolio_id)
        now = self._clock()
        health_ok = self._health_monitor.evaluate(
            decision.portfolio_id, decision, metadata
        )
        events: list[SchedulerEvent] = []
        if not health_ok:
            events.extend(self._handle_failure(state, decision, metadata))
        else:
            state.last_failure_reason = None
            state.last_health_ok = True
            state.fallback_index = 0

        state.last_run = now

        rebalance_instructions: list[RebalanceInstruction] = []
        copy_instructions: list[CopyTradeInstruction] = []

        if decision.rebalance_required:
            rebalance_instructions.append(
                RebalanceInstruction(
                    portfolio_id=decision.portfolio_id,
                    issued_at=now,
                    adjustments=tuple(decision.adjustments),
                    advisories=tuple(decision.advisories),
                    reason="master-rebalance",
                )
            )

        followers = self._followers_by_master.get(decision.portfolio_id, ())
        if followers:
            scaled = self._build_copy_trades(decision, followers, issued_at=now)
            copy_instructions.extend(scaled)

        result = PortfolioScheduleResult(
            master_decision=decision,
            rebalance=tuple(rebalance_instructions),
            copy_trades=tuple(copy_instructions),
            events=tuple(events),
            active_preset=state.current_preset,
        )

        for event in result.events:
            self._emit_event(event)

        return result

    # ------------------------------------------------------------------
    # Wewnętrzne helpery
    # ------------------------------------------------------------------
    def _ensure_state(self, portfolio_id: str) -> _PortfolioState:
        if portfolio_id not in self._portfolios:
            raise KeyError(f"Portfolio {portfolio_id} is not registered")
        return self._portfolios[portfolio_id]

    def _handle_failure(
        self,
        state: _PortfolioState,
        decision: PortfolioDecision,
        metadata: Mapping[str, object] | None,
    ) -> list[SchedulerEvent]:
        events: list[SchedulerEvent] = []
        state.last_health_ok = False
        failure_reason = self._detect_failure_reason(decision, metadata)
        state.last_failure_reason = failure_reason
        events.append(
            SchedulerEvent(
                timestamp=self._clock(),
                level="warning",
                message="strategy-health-degraded",
                payload={
                    "portfolio_id": state.config.portfolio_id,
                    "reason": failure_reason,
                    "preset": state.current_preset,
                },
            )
        )

        fallback = self._select_next_fallback(state)
        if fallback is None:
            events.append(
                SchedulerEvent(
                    timestamp=self._clock(),
                    level="error",
                    message="strategy-self-healing-failed",
                    payload={
                        "portfolio_id": state.config.portfolio_id,
                        "reason": "no-fallback-available",
                        "preset": state.current_preset,
                    },
                )
            )
            return events

        previous = state.current_preset
        state.current_preset = fallback.preset_id
        events.append(
            SchedulerEvent(
                timestamp=self._clock(),
                level="info",
                message="strategy-self-healing",
                payload={
                    "portfolio_id": state.config.portfolio_id,
                    "previous_preset": previous,
                    "new_preset": fallback.preset_id,
                },
            )
        )
        return events

    def _detect_failure_reason(
        self,
        decision: PortfolioDecision,
        metadata: Mapping[str, object] | None,
    ) -> str:
        if metadata:
            reason = metadata.get("failure_reason")
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
        for advisory in decision.advisories:
            if advisory.severity.lower() in {"error", "critical", "fatal"}:
                return advisory.code or advisory.message or "advisory-error"
        return "strategy-health-check"

    def _select_next_fallback(self, state: _PortfolioState) -> StrategyPresetDescriptor | None:
        candidates = state.config.fallback_presets
        if not candidates:
            return None
        start_index = state.fallback_index
        for offset in range(len(candidates)):
            index = (start_index + offset) % len(candidates)
            preset_id = candidates[index]
            try:
                descriptor = self._catalog.preset(preset_id)
            except KeyError:
                _LOGGER.warning("Unknown preset %s referenced as fallback", preset_id)
                continue
            license_state = descriptor.license_status.status
            if license_state != PresetLicenseState.ACTIVE:
                _LOGGER.warning(
                    "Preset %s skipped due to license state %s", preset_id, license_state
                )
                continue
            state.fallback_index = index + 1
            return descriptor
        return None

    def _build_copy_trades(
        self,
        decision: PortfolioDecision,
        followers: Sequence[CopyTradingFollowerConfig],
        *,
        issued_at: datetime,
    ) -> Sequence[CopyTradeInstruction]:
        instructions: list[CopyTradeInstruction] = []
        adjustments = tuple(decision.adjustments)
        advisories = tuple(decision.advisories)
        for follower in followers:
            if not follower.enabled:
                continue
            scaled_adjustments = tuple(
                self._scale_adjustment(adj, follower.scaling) for adj in adjustments
            )
            instructions.append(
                CopyTradeInstruction(
                    follower_id=follower.portfolio_id,
                    master_id=decision.portfolio_id,
                    issued_at=issued_at,
                    scale=follower.scaling,
                    risk_multiplier=follower.risk_multiplier,
                    adjustments=scaled_adjustments,
                    advisories=advisories,
                )
            )
        return instructions

    def _scale_adjustment(
        self, adjustment: PortfolioAdjustment, scale: float
    ) -> PortfolioAdjustment:
        drift = adjustment.proposed_weight - adjustment.current_weight
        scaled = adjustment.current_weight + drift * scale
        metadata = dict(adjustment.metadata)
        metadata["copy_scale"] = scale
        metadata["copy_source_weight"] = adjustment.proposed_weight
        return PortfolioAdjustment(
            symbol=adjustment.symbol,
            current_weight=adjustment.current_weight,
            proposed_weight=max(0.0, min(1.0, scaled)),
            reason=adjustment.reason,
            severity=adjustment.severity,
            metadata=metadata,
        )

    def _emit_event(self, event: SchedulerEvent) -> None:
        if self._audit_logger is not None:
            try:
                self._audit_logger(event.as_dict())
            except Exception:  # pragma: no cover - logujemy w tle
                _LOGGER.exception("Audit logger failure")
        else:
            _LOGGER.debug("Scheduler event: %s", event.as_dict())


__all__ = [
    "CopyTradingFollowerConfig",
    "PortfolioBinding",
    "CopyTradeInstruction",
    "RebalanceInstruction",
    "SchedulerEvent",
    "PortfolioScheduleResult",
    "StrategyHealthMonitor",
    "MultiPortfolioScheduler",
]
