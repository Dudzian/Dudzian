"""Thin runtime shadow adapter for Opportunity AI proposals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Literal

from bot_core.ai.trading_engine import OpportunityCandidate, TradingOpportunityAI
from bot_core.ai.trading_opportunity_shadow import (
    OpportunityShadowContext,
    OpportunityShadowRepository,
)
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal

_LOGGER = logging.getLogger(__name__)
_REQUIRED_FEATURES: tuple[str, ...] = (
    "signal_strength",
    "momentum_5m",
    "volatility_30m",
    "spread_bps",
    "fee_bps",
    "slippage_bps",
    "liquidity_score",
    "risk_penalty_bps",
)


@dataclass(slots=True)
class OpportunityRuntimeShadowAdapter:
    """Emituje shadow proposals Opportunity AI bez wpływu na execution decisions."""

    journal: TradingDecisionJournal | None
    engine: TradingOpportunityAI
    model_reference: str = "latest"
    decision_source: str = "opportunity_ai_shadow"
    mode: str = "shadow"
    shadow_repository: OpportunityShadowRepository | None = None
    model_retry_cooldown_seconds: float = 60.0
    degraded_event_cooldown_seconds: float = 120.0

    _model_ready: bool = False
    _model_unavailable_reason: str | None = None
    _last_model_load_attempt_at: datetime | None = None
    _last_degraded_event_key: tuple[str, ...] | None = None
    _last_degraded_event_at: datetime | None = None

    @dataclass(slots=True, frozen=True)
    class PolicyProbeResult:
        status: Literal["proposal", "degraded", "skipped"]
        decision_available: bool
        accepted: bool | None
        model_version: str | None = None
        decision_source: str | None = None
        rejection_reason: str | None = None
        degraded_reason: str | None = None
        shadow_record_key: str | None = None
        shadow_persistence_status: str = "disabled"
        shadow_persistence_error: str | None = None
        mode: str = "shadow"

    def emit_shadow_proposal(
        self,
        *,
        candidate: object,
        signal: object,
        evaluation: object,
        timestamp: datetime,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        environment: str,
        portfolio: str,
    ) -> "OpportunityRuntimeShadowAdapter.PolicyProbeResult":
        if self.journal is None:
            return self.PolicyProbeResult(
                status="skipped",
                decision_available=False,
                accepted=None,
                degraded_reason="journal_unavailable",
                mode=self.mode,
            )
        timestamp_utc = self._ensure_utc(timestamp)

        opportunity_candidate = self._build_opportunity_candidate(candidate)
        if opportunity_candidate is None:
            self._emit_degraded_event(
                timestamp=timestamp_utc,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                environment=environment,
                portfolio=portfolio,
                symbol=str(getattr(candidate, "symbol", "")),
                side=str(getattr(signal, "side", "")),
                reason="missing_or_invalid_required_features",
            )
            return self.PolicyProbeResult(
                status="degraded",
                decision_available=False,
                accepted=None,
                degraded_reason="missing_or_invalid_required_features",
                mode=self.mode,
            )

        if not self._ensure_model_loaded(now=timestamp_utc):
            self._emit_degraded_event(
                timestamp=timestamp_utc,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                environment=environment,
                portfolio=portfolio,
                symbol=opportunity_candidate.symbol,
                side=str(getattr(signal, "side", "")),
            )
            return self.PolicyProbeResult(
                status="degraded",
                decision_available=False,
                accepted=None,
                degraded_reason=self._model_unavailable_reason or "model_unavailable",
                mode=self.mode,
            )

        try:
            decisions = self.engine.rank((opportunity_candidate,))
        except Exception as exc:  # pragma: no cover - defensywnie, inference nie może zatrzymać runtime
            reason = f"inference_error:{type(exc).__name__}"
            self._emit_degraded_event(
                timestamp=timestamp_utc,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                environment=environment,
                portfolio=portfolio,
                symbol=opportunity_candidate.symbol,
                side=str(getattr(signal, "side", "")),
                reason=reason,
            )
            return self.PolicyProbeResult(
                status="degraded",
                decision_available=False,
                accepted=None,
                degraded_reason=reason,
                mode=self.mode,
            )
        if not decisions:
            reason = "empty_decision_set"
            self._emit_degraded_event(
                timestamp=timestamp_utc,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                environment=environment,
                portfolio=portfolio,
                symbol=opportunity_candidate.symbol,
                side=str(getattr(signal, "side", "")),
                reason=reason,
            )
            return self.PolicyProbeResult(
                status="degraded",
                decision_available=False,
                accepted=None,
                degraded_reason=reason,
                mode=self.mode,
            )
        decision = decisions[0]
        shadow_record_key, persistence_status, persistence_error = self._persist_shadow_record(
            decision=decision,
            decision_timestamp=timestamp_utc,
            candidate=opportunity_candidate,
            strategy_name=strategy_name,
            schedule_name=schedule_name,
            risk_profile=risk_profile,
            environment=environment,
        )
        provenance = dict(decision.provenance)
        provenance.update(
            {
                "mode": self.mode,
                "shadow": True,
                "runtime_integration": "decision_aware_signal_sink",
            }
        )

        event = TradingDecisionEvent(
            event_type="opportunity_shadow_proposal",
            timestamp=timestamp_utc,
            environment=environment,
            portfolio=portfolio,
            risk_profile=risk_profile,
            symbol=decision.symbol,
            side=str(getattr(signal, "side", "")),
            schedule=schedule_name,
            strategy=strategy_name,
            status="proposal",
            confidence=float(decision.confidence),
            telemetry_namespace=f"{environment}.decision.{schedule_name}",
            metadata={
                "decision_source": self.decision_source,
                "mode": self.mode,
                "shadow": "true",
                "model_version": decision.model_version,
                "proposal_rank": str(decision.rank),
                "proposal_direction": decision.proposed_direction,
                "proposal_accepted": "true" if decision.accepted else "false",
                "proposal_rejection_reason": decision.rejection_reason or "",
                "proposal_expected_edge_bps": f"{float(decision.expected_edge_bps):.6f}",
                "proposal_success_probability": f"{float(decision.success_probability):.6f}",
                "runtime_evaluation_accepted": "true"
                if bool(getattr(evaluation, "accepted", False))
                else "false",
                "shadow_record_key": shadow_record_key or "",
                "shadow_persistence_status": persistence_status,
                "shadow_persistence_error": persistence_error or "",
                "provenance": provenance,
            },
        )
        self.journal.record(event)
        return self.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=bool(decision.accepted),
            model_version=decision.model_version,
            decision_source=self.decision_source,
            rejection_reason=decision.rejection_reason,
            shadow_record_key=shadow_record_key,
            shadow_persistence_status=persistence_status,
            shadow_persistence_error=persistence_error,
            mode=self.mode,
        )

    def _persist_shadow_record(
        self,
        *,
        decision: object,
        decision_timestamp: datetime,
        candidate: OpportunityCandidate,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        environment: str,
    ) -> tuple[str | None, str, str | None]:
        if self.shadow_repository is None:
            return None, "disabled", None
        try:
            records = self.engine.build_shadow_records(
                [decision],
                decision_timestamp=decision_timestamp,
                snapshot={
                    "strategy_name": strategy_name,
                    "schedule_name": schedule_name,
                    "risk_profile": risk_profile,
                    "candidate_metadata": self._json_safe_payload(dict(candidate.metadata)),
                },
                context=OpportunityShadowContext(
                    environment=environment,
                    notes={"adapter_mode": self.mode},
                ),
            )
            self.shadow_repository.append_shadow_records(records)
            return records[0].record_key, "persisted", None
        except Exception as exc:  # pragma: no cover - runtime degradacja persistence
            reason = f"shadow_persistence_error:{type(exc).__name__}"
            self._emit_degraded_event(
                timestamp=decision_timestamp,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                environment=environment,
                portfolio="shadow_persistence",
                symbol=str(getattr(decision, "symbol", "")),
                side=str(getattr(decision, "proposed_direction", "")),
                reason=reason,
            )
            _LOGGER.warning("Opportunity shadow persistence failed", exc_info=True)
            return None, "error", str(exc)

    @classmethod
    def _json_safe_payload(cls, payload: Mapping[str, object]) -> dict[str, object]:
        return {str(key): cls._json_safe_value(value) for key, value in payload.items()}

    @classmethod
    def _json_safe_value(cls, value: object) -> object:
        if isinstance(value, datetime):
            return cls._ensure_utc(value).isoformat()
        if isinstance(value, Mapping):
            return cls._json_safe_payload({str(key): item for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return [cls._json_safe_value(item) for item in value]
        return value

    def _ensure_model_loaded(self, *, now: datetime) -> bool:
        if self._model_ready:
            return True
        if self._last_model_load_attempt_at is not None:
            elapsed = (now - self._last_model_load_attempt_at).total_seconds()
            if elapsed < max(0.0, float(self.model_retry_cooldown_seconds)):
                return False
        self._last_model_load_attempt_at = now
        try:
            repository = getattr(self.engine, "_repository", None)
            if repository is not None and hasattr(repository, "_manifest_cache"):
                try:
                    setattr(repository, "_manifest_cache", None)
                except Exception:  # pragma: no cover - defensywne
                    pass
            self.engine.load_model(self.model_reference)
            self._model_ready = True
            self._model_unavailable_reason = None
            return True
        except Exception as exc:  # pragma: no cover - runtime degradacja
            self._model_unavailable_reason = str(exc)
            _LOGGER.debug("Opportunity shadow adapter model unavailable", exc_info=True)
            return False

    def _emit_degraded_event(
        self,
        *,
        timestamp: datetime,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        environment: str,
        portfolio: str,
        symbol: str,
        side: str,
        reason: str | None = None,
    ) -> None:
        if self.journal is None:
            return
        event_key = (
            str(environment),
            str(portfolio),
            str(risk_profile),
            str(schedule_name),
            str(strategy_name),
            str(symbol),
            str(reason or self._model_unavailable_reason or "model_unavailable"),
        )
        if self._last_degraded_event_key == event_key and self._last_degraded_event_at is not None:
            elapsed = (timestamp - self._last_degraded_event_at).total_seconds()
            if elapsed < max(0.0, float(self.degraded_event_cooldown_seconds)):
                return
        event = TradingDecisionEvent(
            event_type="opportunity_shadow_proposal",
            timestamp=self._ensure_utc(timestamp),
            environment=environment,
            portfolio=portfolio,
            risk_profile=risk_profile,
            symbol=symbol,
            side=side,
            schedule=schedule_name,
            strategy=strategy_name,
            status="degraded",
            telemetry_namespace=f"{environment}.decision.{schedule_name}",
            metadata={
                "decision_source": self.decision_source,
                "mode": self.mode,
                "shadow": "true",
                "degraded": "true",
                "degraded_reason": reason or self._model_unavailable_reason or "model_unavailable",
            },
        )
        self._last_degraded_event_key = event_key
        self._last_degraded_event_at = timestamp
        self.journal.record(event)

    def _build_opportunity_candidate(self, candidate: object) -> OpportunityCandidate | None:
        metadata = getattr(candidate, "metadata", None)
        if not isinstance(metadata, Mapping):
            return None

        numeric: dict[str, float] = {}
        for key in _REQUIRED_FEATURES:
            value = metadata.get(key)
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                return None

        return OpportunityCandidate(
            symbol=str(getattr(candidate, "symbol", "")),
            signal_strength=numeric["signal_strength"],
            momentum_5m=numeric["momentum_5m"],
            volatility_30m=numeric["volatility_30m"],
            spread_bps=numeric["spread_bps"],
            fee_bps=numeric["fee_bps"],
            slippage_bps=numeric["slippage_bps"],
            liquidity_score=numeric["liquidity_score"],
            risk_penalty_bps=numeric["risk_penalty_bps"],
            direction_hint=self._normalize_direction(getattr(candidate, "action", None)),
            metadata={"as_of": self._coerce_datetime(metadata.get("as_of"))},
        )

    @staticmethod
    def _normalize_direction(action: object) -> str | None:
        normalized = str(action or "").strip().lower()
        if normalized == "exit":
            return "short"
        if normalized == "enter":
            return "long"
        return None

    @staticmethod
    def _coerce_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return OpportunityRuntimeShadowAdapter._ensure_utc(value)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return OpportunityRuntimeShadowAdapter._ensure_utc(parsed)
        return None

    @staticmethod
    def _ensure_utc(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)


__all__ = ["OpportunityRuntimeShadowAdapter"]
