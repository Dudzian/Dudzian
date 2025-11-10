"""Lifecycle helpers orchestrating fully autonomous AutoTrader deployments."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .app import AutoTrader, DecisionCycleReport
from .decision_scheduler import AutoTraderDecisionScheduler, AutoTraderSchedulerHooks
from bot_core.runtime.journal import TradingDecisionJournal


LOGGER = logging.getLogger(__name__)
_SCHEDULE_SYMBOL = "<schedule>"


@dataclass(slots=True)
class LifecycleBootstrapSnapshot:
    """State recovered from the decision journal during bootstrap."""

    risk_profile: str | None = None
    market_regime: str | None = None
    decision_state: str | None = None
    decision_signal: str | None = None

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.risk_profile:
            payload["risk_profile"] = self.risk_profile
        if self.market_regime:
            payload["market_regime"] = self.market_regime
        if self.decision_state:
            payload["decision_state"] = self.decision_state
        if self.decision_signal:
            payload["decision_signal"] = self.decision_signal
        return payload


@dataclass
class AutoTraderLifecycleManager(AutoTraderSchedulerHooks):
    """Orchestrates autonomous scheduling, guardrails and recovery."""

    trader: AutoTrader
    scheduler: AutoTraderDecisionScheduler
    decision_journal: TradingDecisionJournal | None = None
    guardrail_alert_severity: str = "warning"
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _bootstrap_snapshot: LifecycleBootstrapSnapshot | None = field(default=None, init=False, repr=False)
    _bootstrap_applied: bool = field(default=False, init=False, repr=False)
    _guardrail_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = field(
        default=None, init=False, repr=False
    )
    _failure_attempts: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.decision_journal is None:
            self.decision_journal = getattr(self.trader, "_decision_journal", None)
        self.scheduler.hooks = self

    # ------------------------------------------------------------------
    # Public API -------------------------------------------------------
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the associated scheduler in background mode."""

        self.scheduler.start_in_background()

    def stop(self) -> None:
        """Stop the scheduler and flush any outstanding lifecycle state."""

        self.scheduler.stop_background()

    # ------------------------------------------------------------------
    # Scheduler hooks --------------------------------------------------
    # ------------------------------------------------------------------
    def on_bootstrap(self, scheduler: AutoTraderDecisionScheduler) -> None:  # noqa: D401
        """Restore state from decision logs and enable autonomous trading."""

        del scheduler
        with self._lock:
            if self._bootstrap_applied:
                return
            snapshot = self._restore_state_from_journal()
            self._bootstrap_snapshot = snapshot
            try:
                self.trader.confirm_auto_trade(True)
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("AutoTraderLifecycleManager failed to confirm auto trade", exc_info=True)
            try:
                self.trader.apply_lifecycle_bootstrap(
                    risk_profile=snapshot.risk_profile,
                    market_regime=snapshot.market_regime,
                    decision_state=snapshot.decision_state,
                    decision_signal=snapshot.decision_signal,
                )
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("AutoTraderLifecycleManager bootstrap synchronisation failed", exc_info=True)
            self._record_audit_stage("lifecycle_bootstrap", payload=snapshot.to_metadata())
            self._log_lifecycle_event(
                "scheduler_bootstrap",
                status="started",
                metadata=snapshot.to_metadata(),
            )
            self._bootstrap_applied = True

    def on_cycle_success(self, report: DecisionCycleReport) -> None:
        """Observe successful decision cycles for guardrail integration."""

        self._failure_attempts = 0
        metadata = getattr(report, "metadata", None)
        if isinstance(metadata, Mapping) and metadata:
            snapshot = self._bootstrap_snapshot or LifecycleBootstrapSnapshot()
            regime = metadata.get("market_regime")
            if regime:
                snapshot.market_regime = str(regime)
            state = metadata.get("decision_state")
            if state:
                snapshot.decision_state = str(state)
            signal = metadata.get("decision_signal")
            if signal:
                snapshot.decision_signal = str(signal)
            self._bootstrap_snapshot = snapshot

        decision = getattr(report, "decision", None)
        if decision is None:
            return
        details = getattr(decision, "details", {}) or {}
        reasons = self._extract_guardrail_reasons(details)
        triggers = self._extract_guardrail_triggers(details)
        if not reasons:
            return
        signature = (tuple(reasons), tuple(sorted(triggers)))
        if signature == self._guardrail_signature:
            return
        self._guardrail_signature = signature
        context = {"mode": getattr(decision, "mode", "unknown")}
        if triggers:
            context["triggers"] = ",".join(triggers)
        body = "; ".join(reasons)
        if not self._emit_alert(
            "auto_trader.guardrail",
            "Guardrail zablokował cykl decyzyjny",
            f"Aktywny guardrail zatrzymał decyzję: {body}",
            context=context,
        ):
            LOGGER.debug("AutoTraderLifecycleManager guardrail alert dispatch skipped (no router)")
        payload = {
            "reasons": list(reasons),
            "triggers": list(triggers),
            "mode": getattr(decision, "mode", "unknown"),
        }
        self._log_lifecycle_event("guardrail_alert", status="active", metadata=payload)
        self._record_audit_stage("guardrail_alert", payload=payload)

    def on_cycle_failure(self, exc: BaseException) -> float | None:  # noqa: D401
        """Provide exponential backoff for scheduler restarts."""

        self._failure_attempts += 1
        base = max(1.0, float(getattr(self.scheduler, "restart_backoff_s", 1.0)) or 1.0)
        limit = max(base, float(getattr(self.scheduler, "restart_backoff_max_s", base)) or base)
        delay = min(limit, base * (2 ** (self._failure_attempts - 1)))
        payload = {
            "attempt": self._failure_attempts,
            "delay_s": f"{delay:.2f}",
            "error": repr(exc),
        }
        self._log_lifecycle_event("scheduler_failure", status="error", metadata=payload)
        self._record_audit_stage("scheduler_failure", payload=payload)
        return delay

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def _restore_state_from_journal(self) -> LifecycleBootstrapSnapshot:
        journal = self.decision_journal
        snapshot = LifecycleBootstrapSnapshot()
        if journal is None:
            return snapshot
        try:
            entries = list(journal.export())
        except Exception:  # pragma: no cover - diagnostic only
            LOGGER.debug("AutoTraderLifecycleManager journal export failed", exc_info=True)
            return snapshot
        for entry in reversed(entries):
            if not isinstance(entry, Mapping):
                continue
            event = str(entry.get("event") or "")
            if not snapshot.risk_profile and event == "risk_profile_transition":
                selected = str(entry.get("selected") or "").strip()
                if selected:
                    snapshot.risk_profile = selected
            regime = entry.get("market_regime")
            if snapshot.market_regime is None and regime:
                snapshot.market_regime = str(regime)
                state = entry.get("decision_state")
                if state:
                    snapshot.decision_state = str(state)
                signal = entry.get("decision_signal")
                if signal:
                    snapshot.decision_signal = str(signal)
            if snapshot.risk_profile and snapshot.market_regime:
                break
        return snapshot

    def _extract_guardrail_reasons(self, details: Mapping[str, object]) -> list[str]:
        raw = details.get("guardrail_reasons")
        if isinstance(raw, Sequence):
            return [str(reason) for reason in raw if reason]
        return []

    def _extract_guardrail_triggers(self, details: Mapping[str, object]) -> list[str]:
        raw = details.get("guardrail_triggers")
        if not isinstance(raw, Sequence):
            return []
        triggers: list[str] = []
        for entry in raw:
            if isinstance(entry, Mapping):
                label = entry.get("label") or entry.get("name")
                if label:
                    triggers.append(str(label))
        return triggers

    def _emit_alert(
        self,
        category: str,
        title: str,
        body: str,
        *,
        severity: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        emitter = getattr(self.trader, "_emit_alert", None)
        if not callable(emitter):
            return False
        try:
            return bool(
                emitter(
                    category,
                    title,
                    body,
                    severity=severity or self.guardrail_alert_severity,
                    context=context or {},
                )
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.debug("AutoTraderLifecycleManager alert dispatch failed", exc_info=True)
            return False

    def _log_lifecycle_event(
        self,
        event: str,
        *,
        status: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        logger = getattr(self.trader, "_log_decision_event", None)
        if not callable(logger):
            return
        try:
            logger(event, symbol=_SCHEDULE_SYMBOL, status=status, metadata=dict(metadata or {}))
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.debug("AutoTraderLifecycleManager failed to log lifecycle event", exc_info=True)

    def _record_audit_stage(self, stage: str, *, payload: Mapping[str, object] | None = None) -> None:
        recorder = getattr(self.trader, "_record_decision_audit_stage", None)
        if not callable(recorder):
            return
        try:
            recorder(
                stage,
                symbol=_SCHEDULE_SYMBOL,
                payload=dict(payload or {}),
                metadata={"lifecycle": "autonomous"},
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.debug("AutoTraderLifecycleManager failed to record audit stage", exc_info=True)


__all__ = ["AutoTraderLifecycleManager", "LifecycleBootstrapSnapshot"]
