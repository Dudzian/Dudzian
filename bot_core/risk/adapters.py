"""Adaptery integrujące silnik ryzyka z opcjonalnymi modułami."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol

try:  # pragma: no cover - moduł decision może być opcjonalny
    from bot_core.decision.evaluators import DecisionEvaluator
except Exception:  # pragma: no cover - fallback gdy moduł decision nie jest dostępny

    class DecisionEvaluator(Protocol):
        def evaluate_candidate(self, candidate: Any, context: Any) -> Any: ...


from bot_core.exchanges.base import AccountSnapshot
from bot_core.risk.state import RiskState, build_risk_snapshot


class DecisionBackend(DecisionEvaluator, Protocol):
    """Minimalny kontrakt backendu decyzyjnego używanego przez silnik ryzyka."""


@dataclass(slots=True)
class _DecisionArtifacts:
    candidate_cls: type[Any]
    context_cls: type[Any]
    evaluation_cls: type[Any] | None
    snapshot_cls: type[Any] | None


def _load_decision_artifacts() -> _DecisionArtifacts | None:
    try:  # pragma: no cover - moduł decision może być opcjonalny
        from bot_core.decision.models import (  # type: ignore[import-not-found]
            DecisionCandidate,
            DecisionContext,
            DecisionEvaluation,
            RiskSnapshot,
        )
    except Exception:  # pragma: no cover - brak modułu decision
        return None

    return _DecisionArtifacts(
        candidate_cls=DecisionCandidate,
        context_cls=DecisionContext,
        evaluation_cls=DecisionEvaluation,
        snapshot_cls=RiskSnapshot,
    )


@dataclass(slots=True)
class DecisionOrchestratorAdapter:
    """Adapter budujący kandydatów i kontekst dla DecisionOrchestratora."""

    backend: DecisionBackend
    _artifacts: _DecisionArtifacts | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._artifacts = _load_decision_artifacts()

    @property
    def available(self) -> bool:
        return self._artifacts is not None

    def build_candidate(self, payload: Mapping[str, object]) -> Any:
        if self._artifacts is None:
            raise RuntimeError("Decision models are not available")
        return self._artifacts.candidate_cls.from_mapping(payload)

    def build_context(self, *, snapshot: Mapping[str, object] | Any, account_id: str | None) -> Any:
        if self._artifacts is None:
            raise RuntimeError("Decision models are not available")
        runtime: MutableMapping[str, object] = {}
        if account_id:
            runtime["account"] = account_id
        return self._artifacts.context_cls(risk_snapshot=snapshot, runtime=runtime)

    def build_snapshot(
        self, profile_name: str, state: RiskState, account: AccountSnapshot
    ) -> Mapping[str, object] | Any:
        last_equity = state.last_equity or account.total_equity
        metrics = state.metrics(
            equity=last_equity,
            gross_notional=state.gross_notional(),
            active_positions=state.active_positions(),
        )
        snapshot_payload: MutableMapping[str, object] = build_risk_snapshot(
            state,
            equity=last_equity,
            gross_notional=metrics.gross_notional,
            active_positions=metrics.active_positions,
        ).to_mapping()
        snapshot_payload["last_equity"] = last_equity
        snapshot_payload["start_of_day_equity"] = state.start_of_day_equity or account.total_equity
        snapshot_payload["daily_realized_pnl"] = state.daily_realized_pnl
        snapshot_payload["force_liquidation"] = state.force_liquidation

        if self._artifacts and self._artifacts.snapshot_cls is not None:
            try:
                return self._artifacts.snapshot_cls.from_mapping(profile_name, snapshot_payload)
            except Exception:  # pragma: no cover - defensywne logowanie w silniku
                pass

        snapshot_payload["profile"] = profile_name
        return snapshot_payload

    def evaluate(self, candidate: Any, context: Any) -> Any:
        return self.backend.evaluate_candidate(candidate, context)

    def serialize_evaluation(self, evaluation: Any) -> Mapping[str, object]:
        artifacts = self._artifacts
        if (
            artifacts
            and artifacts.evaluation_cls is not None
            and isinstance(evaluation, artifacts.evaluation_cls)
        ):
            payload: dict[str, object] = {
                "status": "evaluated",
                "accepted": evaluation.accepted,
                "cost_bps": evaluation.cost_bps,
                "net_edge_bps": evaluation.net_edge_bps,
                "reasons": list(evaluation.reasons),
                "risk_flags": list(evaluation.risk_flags),
                "stress_failures": list(evaluation.stress_failures),
                "candidate": evaluation.candidate.to_mapping(),
                "model_expected_return_bps": evaluation.model_expected_return_bps,
                "model_success_probability": evaluation.model_success_probability,
                "model_name": evaluation.model_name,
                "model_selection": (
                    evaluation.model_selection.to_mapping()
                    if evaluation.model_selection is not None
                    else None
                ),
            }
            if evaluation.thresholds_snapshot is not None:
                payload["thresholds"] = dict(evaluation.thresholds_snapshot)
            return payload

        if isinstance(evaluation, Mapping):
            return dict(evaluation)
        return {"status": "evaluated", "accepted": bool(getattr(evaluation, "accepted", False))}

    def format_denial_reason(self, evaluation: Any) -> str:
        artifacts = self._artifacts
        reasons: list[str] = []
        if (
            artifacts
            and artifacts.evaluation_cls is not None
            and isinstance(evaluation, artifacts.evaluation_cls)
        ):
            reasons.extend(str(reason) for reason in evaluation.reasons)
            reasons.extend(str(flag) for flag in evaluation.risk_flags)
            reasons.extend(str(flag) for flag in evaluation.stress_failures)
        elif isinstance(evaluation, Mapping):
            reasons = [
                *(str(reason) for reason in evaluation.get("reasons", []) or []),
                *(str(reason) for reason in evaluation.get("risk_flags", []) or []),
                *(str(reason) for reason in evaluation.get("stress_failures", []) or []),
            ]

        reasons = [reason for reason in reasons if reason]
        if not reasons:
            return "DecisionOrchestrator odrzucił decyzję bez szczegółów."
        return "DecisionOrchestrator odrzucił decyzję: " + "; ".join(reasons)


__all__ = ["DecisionBackend", "DecisionOrchestratorAdapter"]
