"""Interfejsy i serwis ewaluacji kandydatów decyzyjnych."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Mapping, Protocol, Sequence, TYPE_CHECKING

from bot_core.ai import ModelScore
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.metrics import DecisionMetricSet
from bot_core.decision.models import (
    DecisionCandidate,
    DecisionContext,
    DecisionEvaluation,
    RiskSnapshot,
)

if TYPE_CHECKING:  # pragma: no cover - tylko dla statycznego typowania
    from bot_core.decision.orchestrator import StrategyAdvisor


_LOGGER = logging.getLogger(__name__)


class DecisionProvider(Protocol):
    """Źródło informacji potrzebnych do ewaluacji kandydata."""

    def score_with_model(
        self, candidate: DecisionCandidate
    ) -> tuple[str | None, ModelScore | None, object | None]:
        """Zwraca wynik modelu, nazwę oraz metadane selekcji modeli."""

    def thresholds_for_profile(self, profile: str) -> DecisionOrchestratorThresholds:
        """Udostępnia progi dla danego profilu ryzyka."""

    def threshold_snapshot(
        self, thresholds: DecisionOrchestratorThresholds
    ) -> Mapping[str, float | None]:
        """Tworzy snapshot progów użyty w odpowiedzi."""

    def ensure_snapshot(
        self, profile: str, snapshot: Mapping[str, object] | RiskSnapshot
    ) -> RiskSnapshot:
        """Normalizuje snapshot ryzyka."""

    def resolve_cost(self, candidate: DecisionCandidate) -> tuple[float | None, bool]:
        """Zwraca koszt strategii oraz informację o brakujących danych."""

    def resolve_regime(self, candidate: DecisionCandidate):
        """Określa reżim rynkowy dla kandydata."""


class DecisionEvaluator(Protocol):
    """Kontrakt dla serwisu ewaluującego kandydatów."""

    def evaluate_candidate(
        self,
        candidate: DecisionCandidate,
        context: DecisionContext,
    ) -> DecisionEvaluation:
        """Ewaluacja pojedynczego kandydata."""

    def evaluate_candidates(
        self,
        candidates: Sequence[DecisionCandidate],
        contexts: Mapping[str, DecisionContext],
    ) -> Sequence[DecisionEvaluation]:
        """Ewaluacja wielu kandydatów."""


class DecisionEvaluationService(DecisionEvaluator):
    """Serwis agregujący ewaluację i metryki kandydatów."""

    def __init__(
        self,
        *,
        config: DecisionEngineConfig,
        provider: DecisionProvider,
        strategy_advisor: "StrategyAdvisor",
        metrics: DecisionMetricSet | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._strategy_advisor = strategy_advisor
        self._metrics = metrics or DecisionMetricSet()

    def evaluate_candidate(
        self,
        candidate: DecisionCandidate,
        context: DecisionContext,
    ) -> DecisionEvaluation:
        risk_snapshot = context.risk_snapshot
        thresholds = self._provider.thresholds_for_profile(candidate.risk_profile)
        thresholds_snapshot = self._provider.threshold_snapshot(thresholds)
        snapshot = self._provider.ensure_snapshot(candidate.risk_profile, risk_snapshot)
        model_name, model_score, selection_metadata = self._provider.score_with_model(
            candidate
        )

        if model_score is not None:
            expected_probability = model_score.success_probability
            expected_value_bps = (
                model_score.expected_return_bps * model_score.success_probability
            )
            expected_return_bps = model_score.expected_return_bps
        else:
            expected_probability = candidate.expected_probability
            expected_value_bps = candidate.expected_value_bps
            expected_return_bps = candidate.expected_return_bps

        reasons: list[str] = []
        risk_flags: list[str] = []

        if expected_probability < self._config.min_probability:
            reasons.append(
                (
                    "prawdopodobieństwo {p:.3f} poniżej progu {threshold:.3f}"
                ).format(p=expected_probability, threshold=self._config.min_probability)
            )

        cost_bps, missing_cost = self._provider.resolve_cost(candidate)
        if missing_cost and self._config.require_cost_data:
            reasons.append("brak danych kosztowych dla strategii/profilu")
        effective_cost = cost_bps or 0.0

        if cost_bps is not None and cost_bps > thresholds.max_cost_bps:
            reasons.append(
                (
                    "koszt {cost:.2f} bps przekracza limit {limit:.2f} bps"
                ).format(cost=cost_bps, limit=thresholds.max_cost_bps)
            )

        net_edge = expected_value_bps - effective_cost
        if net_edge < thresholds.min_net_edge_bps:
            reasons.append(
                (
                    "net edge {edge:.2f} bps poniżej progu {limit:.2f} bps"
                ).format(edge=net_edge, limit=thresholds.min_net_edge_bps)
            )

        if thresholds.max_latency_ms is not None and candidate.latency_ms is not None:
            if candidate.latency_ms > thresholds.max_latency_ms:
                reasons.append(
                    (
                        "latencja {lat:.1f} ms przekracza limit {limit:.1f} ms"
                    ).format(lat=candidate.latency_ms, limit=thresholds.max_latency_ms)
                )

        self._check_risk_limits(candidate, snapshot, thresholds, reasons, risk_flags)

        stress_failures = self._run_stress_tests(candidate, thresholds, cost_bps or 0.0)

        recommendation = self._strategy_advisor.recommend(
            candidate,
            regime=self._provider.resolve_regime(candidate),
            model_score=model_score,
            selection=selection_metadata,
            cost_bps=cost_bps,
            net_edge_bps=net_edge,
        )
        if selection_metadata is not None:
            selection_metadata = replace(
                selection_metadata,
                recommended_modes=recommendation.modes,
                recommended_position_size=recommendation.position_size,
                recommended_risk_score=recommendation.risk_score,
            )

        accepted = not reasons and not stress_failures
        evaluation = DecisionEvaluation(
            candidate=candidate,
            accepted=accepted,
            cost_bps=cost_bps,
            net_edge_bps=net_edge,
            reasons=tuple(reasons),
            risk_flags=tuple(risk_flags),
            stress_failures=tuple(stress_failures),
            model_expected_return_bps=(model_score.expected_return_bps if model_score else None),
            model_success_probability=(
                model_score.success_probability if model_score else None
            ),
            model_name=model_name if model_score else None,
            model_selection=selection_metadata,
            thresholds_snapshot=thresholds_snapshot,
            recommended_modes=recommendation.modes,
            recommended_position_size=recommendation.position_size,
            recommended_risk_score=recommendation.risk_score,
        )
        if not accepted:
            _LOGGER.info(
                "Decision evaluation rejected candidate strategy=%s symbol=%s reasons=%s risk_flags=%s stress_failures=%s thresholds=%s",
                candidate.strategy,
                candidate.symbol or "<unknown>",
                tuple(reasons),
                tuple(risk_flags),
                tuple(stress_failures),
                thresholds_snapshot,
            )
        self._strategy_advisor.observe(candidate, evaluation)
        self._record_metrics(evaluation)
        return evaluation

    def evaluate_candidates(
        self,
        candidates: Sequence[DecisionCandidate],
        contexts: Mapping[str, DecisionContext],
    ) -> Sequence[DecisionEvaluation]:
        evaluations: list[DecisionEvaluation] = []
        for candidate in candidates:
            context = contexts.get(candidate.risk_profile)
            if context is None:
                evaluation = DecisionEvaluation(
                    candidate=candidate,
                    accepted=False,
                    cost_bps=None,
                    net_edge_bps=None,
                    reasons=("brak snapshotu ryzyka dla profilu",),
                    risk_flags=(),
                    stress_failures=(),
                )
                evaluations.append(evaluation)
                self._record_metrics(evaluation)
                continue
            evaluations.append(self.evaluate_candidate(candidate, context))
        return evaluations

    # ------------------------------------------------------------ helpery --
    def _record_metrics(self, evaluation: DecisionEvaluation) -> None:
        try:
            self._metrics.observe_evaluation(evaluation)
        except Exception:  # pragma: no cover - metryki nie powinny blokować decyzji
            _LOGGER.exception("Failed to record decision metrics", exc_info=True)

    def _check_risk_limits(
        self,
        candidate: DecisionCandidate,
        snapshot: RiskSnapshot,
        thresholds: DecisionOrchestratorThresholds,
        reasons: list[str],
        risk_flags: list[str],
    ) -> None:
        if snapshot.force_liquidation:
            risk_flags.append("force_liquidation_active")
            reasons.append("profil w stanie force_liquidation")

        if snapshot.daily_loss_pct > thresholds.max_daily_loss_pct:
            risk_flags.append("daily_loss_limit")
            reasons.append(
                (
                    "przekroczony dzienny limit straty: {value:.4f} > {limit:.4f}"
                ).format(value=snapshot.daily_loss_pct, limit=thresholds.max_daily_loss_pct)
            )

        if snapshot.drawdown_pct > thresholds.max_drawdown_pct:
            risk_flags.append("drawdown_limit")
            reasons.append(
                (
                    "przekroczony limit obsunięcia: {value:.4f} > {limit:.4f}"
                ).format(value=snapshot.drawdown_pct, limit=thresholds.max_drawdown_pct)
            )

        equity = max(snapshot.start_of_day_equity, 1.0)
        future_gross = snapshot.gross_notional + max(candidate.notional, 0.0)
        position_ratio = future_gross / equity
        if position_ratio > thresholds.max_position_ratio:
            risk_flags.append("gross_notional_limit")
            reasons.append(
                (
                    "ekspozycja {ratio:.4f} przekracza limit {limit:.4f}"
                ).format(ratio=position_ratio, limit=thresholds.max_position_ratio)
            )

        new_positions = snapshot.active_positions
        if not snapshot.contains_symbol(candidate.symbol):
            new_positions += 1
        if new_positions > thresholds.max_open_positions:
            risk_flags.append("open_positions_limit")
            reasons.append(
                (
                    "liczba pozycji {count} przekracza limit {limit}"
                ).format(count=new_positions, limit=thresholds.max_open_positions)
            )

        if (
            thresholds.max_trade_notional is not None
            and candidate.notional > thresholds.max_trade_notional
        ):
            risk_flags.append("trade_notional_limit")
            reasons.append(
                (
                    "wartość zlecenia {notional:.2f} przekracza limit {limit:.2f}"
                ).format(
                    notional=candidate.notional,
                    limit=thresholds.max_trade_notional,
                )
            )

    def _run_stress_tests(
        self,
        candidate: DecisionCandidate,
        thresholds: DecisionOrchestratorThresholds,
        base_cost_bps: float,
    ) -> Sequence[str]:
        stress_config = self._config.stress_tests
        if stress_config is None:
            return ()
        failures: list[str] = []
        stressed_cost = (base_cost_bps + stress_config.cost_shock_bps) * max(
            stress_config.slippage_multiplier, 0.0
        )
        stressed_net_edge = candidate.expected_value_bps - stressed_cost
        if stressed_net_edge < thresholds.min_net_edge_bps:
            failures.append(
                (
                    "stress cost edge {edge:.2f} bps poniżej progu {limit:.2f} bps"
                ).format(edge=stressed_net_edge, limit=thresholds.min_net_edge_bps)
            )
        if (
            thresholds.max_latency_ms is not None
            and candidate.latency_ms is not None
            and candidate.latency_ms + stress_config.latency_spike_ms
            > thresholds.max_latency_ms
        ):
            failures.append(
                (
                    "latency stress {lat:.1f} ms przekracza limit {limit:.1f} ms"
                ).format(
                    lat=candidate.latency_ms + stress_config.latency_spike_ms,
                    limit=thresholds.max_latency_ms,
                )
            )
        return failures


__all__ = [
    "DecisionEvaluator",
    "DecisionEvaluationService",
    "DecisionProvider",
]

