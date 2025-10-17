"""DecisionOrchestrator oceniający kandydatów decyzji inwestycyjnych."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from bot_core.ai import DecisionModelInference, ModelScore
from bot_core.config.models import (
    DecisionEngineConfig,
    DecisionOrchestratorThresholds,
    DecisionStressTestConfig,
)
from bot_core.decision.models import DecisionCandidate, DecisionEvaluation, RiskSnapshot

try:
    from bot_core.tco.models import ProfileCostSummary, StrategyCostSummary, TCOReport
except Exception:  # pragma: no cover - moduł TCO może nie być dostępny w niektórych gałęziach
    ProfileCostSummary = None  # type: ignore
    StrategyCostSummary = None  # type: ignore
    TCOReport = None  # type: ignore


@dataclass(slots=True)
class _CostIndex:
    lookup: MutableMapping[tuple[str, str], float]
    default_cost: float | None


class DecisionOrchestrator:
    """Klasa realizująca logikę DecisionOrchestratora Etapu 5."""

    def __init__(
        self,
        config: DecisionEngineConfig,
        *,
        inference: DecisionModelInference | None = None,
    ) -> None:
        self._config = config
        self._cost_index = _CostIndex(lookup={}, default_cost=None)
        self._inference = inference

    def attach_inference_service(self, inference: DecisionModelInference | None) -> None:
        """Podłącza usługę inference dostarczającą prognozy modelu AI."""

        self._inference = inference

    # ------------------------------------------------------------------ koszty --
    def update_costs_from_report(
        self, report: TCOReport | Mapping[str, object]
    ) -> None:
        """Buduje indeks kosztów (bps) na podstawie raportu TCO."""

        index: MutableMapping[tuple[str, str], float] = {}
        default_cost: float | None = None

        if TCOReport is not None and isinstance(report, TCOReport):
            strategies = report.strategies.values()
            default_cost = float(report.total.cost_bps)
            for summary in strategies:
                self._ingest_strategy_summary(summary, index)
        else:
            data = dict(report)
            strategies_data = data.get("strategies", {}) or {}
            for strategy_name, summary_raw in strategies_data.items():
                index[(str(strategy_name), "__total__")] = self._extract_cost_bps(
                    summary_raw.get("total")
                )
                profiles = summary_raw.get("profiles", {}) or {}
                for profile_name, profile_raw in profiles.items():
                    index[(str(strategy_name), str(profile_name))] = self._extract_cost_bps(
                        profile_raw
                    )
            total_raw = data.get("total")
            if total_raw is not None:
                default_cost = self._extract_cost_bps(total_raw)

        self._cost_index = _CostIndex(lookup=index, default_cost=default_cost)

    def _ingest_strategy_summary(
        self,
        summary: StrategyCostSummary,
        index: MutableMapping[tuple[str, str], float],
    ) -> None:
        index[(summary.strategy, "__total__")] = float(summary.total.cost_bps)
        for profile_name, profile_summary in summary.profiles.items():
            index[(summary.strategy, profile_name)] = float(profile_summary.cost_bps)

    def _extract_cost_bps(self, payload: object) -> float:
        if payload is None:
            return 0.0
        if ProfileCostSummary is not None and isinstance(payload, ProfileCostSummary):
            return float(payload.cost_bps)
        if isinstance(payload, Mapping):
            value = payload.get("cost_bps")
            if value is None:
                return 0.0
            return float(value)
        return float(payload)

    # --------------------------------------------------------------- ewaluacja --
    def evaluate_candidate(
        self,
        candidate: DecisionCandidate,
        risk_snapshot: Mapping[str, object] | RiskSnapshot,
    ) -> DecisionEvaluation:
        thresholds = self._thresholds_for_profile(candidate.risk_profile)
        snapshot = self._ensure_snapshot(candidate.risk_profile, risk_snapshot)
        model_score = self._score_with_model(candidate)
        if model_score is not None:
            expected_probability = model_score.success_probability
            expected_value_bps = model_score.expected_return_bps * model_score.success_probability
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

        cost_bps, missing_cost = self._resolve_cost(candidate)
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

        accepted = not reasons and not stress_failures
        return DecisionEvaluation(
            candidate=candidate,
            accepted=accepted,
            cost_bps=cost_bps,
            net_edge_bps=net_edge,
            reasons=tuple(reasons),
            risk_flags=tuple(risk_flags),
            stress_failures=tuple(stress_failures),
            model_expected_return_bps=expected_return_bps if model_score else None,
            model_success_probability=expected_probability if model_score else None,
        )

    def evaluate_candidates(
        self,
        candidates: Sequence[DecisionCandidate],
        risk_snapshots: Mapping[str, Mapping[str, object] | RiskSnapshot],
    ) -> Sequence[DecisionEvaluation]:
        evaluations: list[DecisionEvaluation] = []
        for candidate in candidates:
            snapshot_raw = risk_snapshots.get(candidate.risk_profile)
            if snapshot_raw is None:
                evaluations.append(
                    DecisionEvaluation(
                        candidate=candidate,
                        accepted=False,
                        cost_bps=None,
                        net_edge_bps=None,
                        reasons=("brak snapshotu ryzyka dla profilu",),
                        risk_flags=(),
                        stress_failures=(),
                    )
                )
                continue
            evaluations.append(self.evaluate_candidate(candidate, snapshot_raw))
        return evaluations

    # ----------------------------------------------------------------- helpery --
    def _thresholds_for_profile(
        self, profile: str
    ) -> DecisionOrchestratorThresholds:
        overrides = self._config.profile_overrides
        if overrides and profile in overrides:
            return overrides[profile]
        return self._config.orchestrator

    def _ensure_snapshot(
        self,
        profile: str,
        snapshot: Mapping[str, object] | RiskSnapshot,
    ) -> RiskSnapshot:
        if isinstance(snapshot, RiskSnapshot):
            return snapshot
        if isinstance(snapshot, Mapping):
            return RiskSnapshot.from_mapping(profile, snapshot)
        raise TypeError("Nieobsługiwany typ snapshotu ryzyka")

    def _resolve_cost(self, candidate: DecisionCandidate) -> tuple[float | None, bool]:
        if candidate.cost_bps_override is not None:
            return candidate.cost_bps_override, False
        cost = self._cost_index.lookup.get((candidate.strategy, candidate.risk_profile))
        if cost is None:
            cost = self._cost_index.lookup.get((candidate.strategy, "__total__"))
        if cost is None:
            cost = self._cost_index.default_cost
        missing = cost is None
        if cost is None and self._config.penalty_cost_bps > 0:
            cost = self._config.penalty_cost_bps
        return cost, missing

    def _score_with_model(self, candidate: DecisionCandidate) -> ModelScore | None:
        inference = self._inference
        if inference is None or not getattr(inference, "is_ready", True):
            return None
        features = self._extract_features(candidate)
        if not features:
            return None
        try:
            return inference.score(features)
        except Exception:  # pragma: no cover - inference nie powinno zatrzymać ewaluacji
            return None

    def _extract_features(self, candidate: DecisionCandidate) -> Mapping[str, float] | None:
        metadata = candidate.metadata or {}
        sources = [
            metadata.get("model_features"),
            metadata.get("features"),
            metadata.get("decision_engine", {}).get("features")
            if isinstance(metadata.get("decision_engine"), Mapping)
            else None,
        ]
        for raw in sources:
            if isinstance(raw, Mapping):
                features: dict[str, float] = {}
                for key, value in raw.items():
                    try:
                        features[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
                if features:
                    return features
        return None

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
        stress_config: DecisionStressTestConfig | None = self._config.stress_tests
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


__all__ = ["DecisionOrchestrator"]
