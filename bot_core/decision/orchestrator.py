"""DecisionOrchestrator oceniający kandydatów decyzji inwestycyjnych."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.ai import DecisionModelInference, MarketRegime, ModelScore
from bot_core.ai.inference import ModelRepository
from bot_core.ai.validation import ModelQualityReport, load_latest_quality_report
from bot_core.reporting.model_quality import DEFAULT_QUALITY_DIR
from bot_core.config.models import (
    DecisionEngineConfig,
    DecisionOrchestratorThresholds,
    DecisionStressTestConfig,
)
from bot_core.decision.models import (
    DecisionCandidate,
    DecisionEvaluation,
    ModelSelectionDetail,
    ModelSelectionMetadata,
    RiskSnapshot,
)

try:
    from bot_core.tco.models import ProfileCostSummary, StrategyCostSummary, TCOReport
except Exception:  # pragma: no cover - moduł TCO może nie być dostępny w niektórych gałęziach
    ProfileCostSummary = None  # type: ignore
    StrategyCostSummary = None  # type: ignore
    TCOReport = None  # type: ignore


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _CostIndex:
    lookup: MutableMapping[tuple[str, str], float]
    default_cost: float | None


@dataclass(slots=True)
class ModelPerformanceSummary:
    """Agregaty skuteczności modeli inference."""

    mae: float
    directional_accuracy: float
    score: float
    weight: float
    updates: int
    updated_at: datetime


@dataclass(slots=True)
class StrategyPerformanceSummary:
    """Aggregated effectiveness metrics for an execution strategy."""

    strategy: str
    regime: MarketRegime | str
    hit_rate: float
    pnl: float
    sharpe: float
    updated_at: datetime
    observations: int = 0


@dataclass(slots=True)
class StrategyRecalibrationSchedule:
    strategy: str
    interval: timedelta
    next_run: datetime


class DecisionOrchestrator:
    """Klasa realizująca logikę DecisionOrchestratora Etapu 5."""

    def __init__(
        self,
        config: DecisionEngineConfig,
        *,
        inference: DecisionModelInference | None = None,
        performance_half_life_hours: float | None = 24.0,
        performance_history_limit: int = 100,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._cost_index = _CostIndex(lookup={}, default_cost=None)
        self._inference: DecisionModelInference | None = None
        self._inferences: MutableMapping[str, DecisionModelInference] = {}
        self._default_model_name: str | None = None
        self._model_usage: MutableMapping[str, int] = {}
        self._model_performance: MutableMapping[
            tuple[str, str], MutableMapping[str, ModelPerformanceSummary]
        ] = {}
        self._performance_half_life_hours = (
            float(performance_half_life_hours)
            if performance_half_life_hours is not None
            else None
        )
        self._performance_history_limit = max(1, int(performance_history_limit))
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
        self._strategy_performance: MutableMapping[str, StrategyPerformanceSummary] = {}
        self._strategy_schedules: MutableMapping[str, StrategyRecalibrationSchedule] = {}
        if inference is not None:
            self.attach_named_inference("__default__", inference, set_default=True)

    def attach_inference_service(self, inference: DecisionModelInference | None) -> None:
        """Podłącza usługę inference dostarczającą prognozy modelu AI."""
        if inference is None:
            self._inference = None
            self._default_model_name = None
            return
        self.attach_named_inference("__default__", inference, set_default=True)

    def attach_named_inference(
        self,
        name: str,
        inference: DecisionModelInference,
        *,
        set_default: bool = False,
    ) -> None:
        key = name.lower()
        self._inferences[key] = inference
        if set_default or self._inference is None:
            self._default_model_name = key
            self._inference = inference

    def detach_named_inference(self, name: str) -> None:
        key = name.lower()
        self._inferences.pop(key, None)
        if self._default_model_name == key:
            self._default_model_name = None
            self._inference = None

    def load_repository_inference(
        self,
        repository: ModelRepository,
        *,
        model_name: str,
        alias: str = "latest",
        inference_name: str = "__default__",
        quality_history: str | Path | None = None,
        fallback_alias: str | None = None,
    ) -> tuple[str, ModelQualityReport | None]:
        """Ładuje model z repozytorium respektując raporty jakości."""

        quality_root = quality_history if quality_history is not None else DEFAULT_QUALITY_DIR
        report = load_latest_quality_report(model_name, history_root=quality_root)

        reference = alias
        if report is not None and report.status == "degraded":
            baseline = report.baseline_version or fallback_alias
            if baseline:
                try:
                    repository.resolve(baseline)
                except (KeyError, FileNotFoundError):
                    baseline = None
            if baseline:
                _LOGGER.warning(
                    "Rolling back model %s to baseline %s after degraded quality", model_name, baseline
                )
                reference = baseline

        inference = DecisionModelInference(repository)
        inference.model_label = model_name
        inference.load_weights(reference)
        self.attach_named_inference(inference_name, inference, set_default=(inference_name == "__default__"))
        return reference, report

    def update_model_performance(
        self,
        name: str,
        metrics: Mapping[str, object],
        *,
        strategy: str | None = None,
        risk_profile: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        key = (strategy or "__any__", risk_profile or "__any__")
        summary_map = self._model_performance.setdefault(key, {})
        summary_metrics = self._extract_metrics_block(metrics)
        score = self._score_from_metrics(summary_metrics)
        mae = float(summary_metrics.get("mae", 0.0))
        directional = float(summary_metrics.get("directional_accuracy", 0.0))
        now = timestamp or self._now()
        lower_name = name.lower()
        previous = summary_map.get(lower_name)
        if previous is not None:
            previous = self._decay_summary(previous, now)
        summary_map[lower_name] = self._compose_summary(
            previous,
            mae,
            directional,
            score,
            now,
        )

    # ---------------------------------------------------------- strategie --
    def record_strategy_performance(
        self,
        strategy: str,
        regime: MarketRegime | str,
        *,
        hit_rate: float,
        pnl: float,
        sharpe: float,
        observations: int = 1,
        timestamp: datetime | None = None,
    ) -> None:
        key = strategy.lower()
        now = timestamp or self._now()
        previous = self._strategy_performance.get(key)
        if previous is not None and previous.regime == regime:
            total_obs = max(previous.observations + observations, 1)
            hit_rate = (previous.hit_rate * previous.observations + hit_rate * observations) / total_obs
            pnl = (previous.pnl * previous.observations + pnl * observations) / total_obs
            sharpe = (previous.sharpe * previous.observations + sharpe * observations) / total_obs
            observations = total_obs
        self._strategy_performance[key] = StrategyPerformanceSummary(
            strategy=strategy,
            regime=regime,
            hit_rate=float(hit_rate),
            pnl=float(pnl),
            sharpe=float(sharpe),
            observations=int(observations),
            updated_at=now,
        )

    def select_strategy(self, regime: MarketRegime | str) -> str | None:
        if isinstance(regime, str):
            try:
                regime = MarketRegime(regime)
            except ValueError:
                regime = MarketRegime.TREND
        best_name: str | None = None
        best_score: float | None = None
        for summary in self._strategy_performance.values():
            summary_regime = summary.regime
            if isinstance(summary_regime, str):
                try:
                    summary_regime = MarketRegime(summary_regime)
                except ValueError:
                    continue
            if summary_regime != regime:
                continue
            score = summary.hit_rate * (1.0 + max(summary.sharpe, 0.0)) + summary.pnl
            if best_score is None or score > best_score:
                best_score = score
                best_name = summary.strategy
        return best_name

    def strategy_performance_snapshot(self) -> Mapping[str, StrategyPerformanceSummary]:
        return dict(self._strategy_performance)

    def schedule_strategy_recalibration(
        self,
        strategy: str,
        interval: timedelta,
        *,
        first_run: datetime | None = None,
    ) -> StrategyRecalibrationSchedule:
        key = strategy.lower()
        next_run = (first_run or self._now()) + interval
        schedule = StrategyRecalibrationSchedule(strategy=strategy, interval=interval, next_run=next_run)
        self._strategy_schedules[key] = schedule
        return schedule

    def due_recalibrations(
        self, now: datetime | None = None
    ) -> Sequence[StrategyRecalibrationSchedule]:
        reference = now or self._now()
        return tuple(schedule for schedule in self._strategy_schedules.values() if schedule.next_run <= reference)

    def mark_recalibrated(
        self,
        strategy: str,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        key = strategy.lower()
        schedule = self._strategy_schedules.get(key)
        if schedule is None:
            return
        reference = timestamp or self._now()
        self._strategy_schedules[key] = StrategyRecalibrationSchedule(
            strategy=schedule.strategy,
            interval=schedule.interval,
            next_run=reference + schedule.interval,
        )

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
        thresholds_snapshot = self._threshold_snapshot(thresholds)
        snapshot = self._ensure_snapshot(candidate.risk_profile, risk_snapshot)
        model_name, model_score, selection_metadata = self._score_with_model(candidate)
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
        evaluation = DecisionEvaluation(
            candidate=candidate,
            accepted=accepted,
            cost_bps=cost_bps,
            net_edge_bps=net_edge,
            reasons=tuple(reasons),
            risk_flags=tuple(risk_flags),
            stress_failures=tuple(stress_failures),
            model_expected_return_bps=(
                model_score.expected_return_bps if model_score else None
            ),
            model_success_probability=(
                model_score.success_probability if model_score else None
            ),
            model_name=model_name if model_score else None,
            model_selection=selection_metadata,
            thresholds_snapshot=thresholds_snapshot,
        )
        if not accepted:
            _LOGGER.info(
                "DecisionOrchestrator rejected candidate strategy=%s symbol=%s reasons=%s risk_flags=%s stress_failures=%s thresholds=%s",
                candidate.strategy,
                candidate.symbol or "<unknown>",
                tuple(reasons),
                tuple(risk_flags),
                tuple(stress_failures),
                thresholds_snapshot,
            )
        return evaluation

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

    def _threshold_snapshot(
        self, thresholds: DecisionOrchestratorThresholds
    ) -> Mapping[str, float | None]:
        return {
            "max_cost_bps": thresholds.max_cost_bps,
            "min_net_edge_bps": thresholds.min_net_edge_bps,
            "max_latency_ms": thresholds.max_latency_ms,
            "max_daily_loss_pct": thresholds.max_daily_loss_pct,
            "max_drawdown_pct": thresholds.max_drawdown_pct,
            "max_position_ratio": thresholds.max_position_ratio,
            "max_open_positions": thresholds.max_open_positions,
            "max_trade_notional": getattr(thresholds, "max_trade_notional", None),
            "min_probability": self._config.min_probability,
            "penalty_cost_bps": getattr(self._config, "penalty_cost_bps", 0.0),
        }

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

    def model_usage_statistics(self, *, reset: bool = False) -> Mapping[str, int]:
        """Zwraca liczbę ewaluacji wykonanych przez poszczególne modele."""

        stats = dict(self._model_usage)
        if reset:
            self._model_usage.clear()
        return stats

    def _score_with_model(
        self, candidate: DecisionCandidate
    ) -> tuple[str | None, ModelScore | None, ModelSelectionMetadata | None]:
        name, inference, details = self._resolve_model_selection(candidate)
        selection_metadata: ModelSelectionMetadata | None = None
        detail_map = {detail.name: detail for detail in details}
        if name is None or inference is None:
            if details:
                selection_metadata = ModelSelectionMetadata(
                    selected=name,
                    candidates=tuple(details),
                )
            return None, None, selection_metadata

        features = self._extract_features(candidate)
        if not features:
            if name in detail_map:
                detail_map[name].reason = "brak cech numerycznych"
            selection_metadata = ModelSelectionMetadata(
                selected=name,
                candidates=tuple(details),
            )
            return name, None, selection_metadata
        try:
            score = inference.score(features)
        except Exception:  # pragma: no cover - inference nie powinno zatrzymać ewaluacji
            if name in detail_map:
                detail_map[name].reason = "błąd podczas score()"
            selection_metadata = ModelSelectionMetadata(
                selected=name,
                candidates=tuple(details),
            )
            return name, None, selection_metadata

        if name in detail_map:
            detail_map[name].reason = None
        selection_metadata = ModelSelectionMetadata(
            selected=name,
            candidates=tuple(details),
        )
        self._model_usage[name] = self._model_usage.get(name, 0) + 1
        return name, score, selection_metadata

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

    def _resolve_model_selection(
        self, candidate: DecisionCandidate
    ) -> tuple[str | None, DecisionModelInference | None, Sequence[ModelSelectionDetail]]:
        strategy = candidate.strategy or "__any__"
        risk_profile = candidate.risk_profile or "__any__"
        keys = [
            (strategy, risk_profile),
            (strategy, "__any__"),
            ("__any__", risk_profile),
            ("__any__", "__any__"),
        ]
        now = self._now()
        details: dict[str, ModelSelectionDetail] = {}
        for key in keys:
            summary_map = self._model_performance.get(key)
            if not summary_map:
                continue
            best_name: str | None = None
            best_value: float | None = None
            for name, summary in sorted(summary_map.items()):
                decayed = self._decay_summary(summary, now)
                if decayed.weight <= 1e-6:
                    self._remove_model_summary(key, name)
                    continue
                available, reason = self._inference_availability(self._inferences.get(name))
                effective_score = decayed.score * decayed.weight
                existing = details.get(name)
                if existing is None:
                    details[name] = ModelSelectionDetail(
                        name=name,
                        score=decayed.score,
                        weight=decayed.weight,
                        effective_score=effective_score,
                        updated_at=decayed.updated_at,
                        available=available,
                        reason=reason,
                    )
                else:
                    existing.score = decayed.score
                    existing.weight = decayed.weight
                    existing.effective_score = effective_score
                    existing.updated_at = decayed.updated_at
                    existing.available = available
                    if reason and existing.reason is None:
                        existing.reason = reason
                if available and (best_value is None or effective_score > best_value):
                    best_name = name
                    best_value = effective_score
            if best_name is not None:
                break
        else:
            best_name = None

        # Uzupełnij szczegóły dla zarejestrowanych inference bez historii
        for name, inference in sorted(self._inferences.items()):
            if name in details:
                detail = details[name]
                if detail.reason is None and detail.effective_score is None:
                    detail.reason = "brak historii skuteczności"
                continue
            available, reason = self._inference_availability(inference)
            details[name] = ModelSelectionDetail(
                name=name,
                available=available,
                reason=reason or "brak historii skuteczności",
            )

        selected_name = best_name
        selected_inference = self._inferences.get(best_name) if best_name else None
        available, reason = self._inference_availability(selected_inference)
        if not available:
            if best_name and best_name in details and reason:
                details[best_name].reason = reason
            selected_name = None
            selected_inference = None

        default_name = self._default_model_name
        default_inference = self._inferences.get(default_name) if default_name else None
        default_available, default_reason = self._inference_availability(default_inference)

        if selected_name is None and default_available:
            selected_name = default_name
            selected_inference = default_inference
        elif selected_name is None:
            for name, inference in sorted(self._inferences.items()):
                available, reason = self._inference_availability(inference)
                if available:
                    selected_name = name
                    selected_inference = inference
                    break
                if name in details and reason:
                    details[name].reason = reason

        if default_name and default_name in details and default_reason and not default_available:
            details[default_name].reason = default_reason
        elif default_name and default_name in details and details[default_name].reason is None:
            details[default_name].reason = "brak historii skuteczności"

        return (
            selected_name,
            selected_inference,
            tuple(details[name] for name in sorted(details)),
        )

    def _inference_availability(
        self, inference: DecisionModelInference | None
    ) -> tuple[bool, str | None]:
        if inference is None:
            return False, "model nie jest zarejestrowany"
        if not getattr(inference, "is_ready", True):
            return False, "model niedostępny"
        return True, None

    def _score_from_metrics(self, metrics: Mapping[str, float]) -> float:
        mae = float(metrics.get("mae", 0.0))
        directional = float(metrics.get("directional_accuracy", 0.0))
        penalty = mae / 100.0
        return directional - penalty

    def _extract_metrics_block(self, metrics: Mapping[str, object]) -> Mapping[str, float]:
        if not isinstance(metrics, Mapping):
            return {}
        if metrics and any(isinstance(value, Mapping) for value in metrics.values()):
            summary = metrics.get("summary")
            if isinstance(summary, Mapping):
                return {
                    str(key): float(value)
                    for key, value in summary.items()
                    if isinstance(value, (int, float))
                }
            for candidate_key in ("test", "validation", "train"):
                candidate = metrics.get(candidate_key)
                if isinstance(candidate, Mapping) and candidate:
                    return {
                        str(key): float(value)
                        for key, value in candidate.items()
                        if isinstance(value, (int, float))
                    }
            for value in metrics.values():
                if isinstance(value, Mapping):
                    return {
                        str(key): float(val)
                        for key, val in value.items()
                        if isinstance(val, (int, float))
                    }
            return {}
        return {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

    def _compose_summary(
        self,
        previous: ModelPerformanceSummary | None,
        mae: float,
        directional: float,
        score: float,
        now: datetime,
    ) -> ModelPerformanceSummary:
        weight = 1.0
        updates = 1
        if previous is not None:
            max_weight = float(self._performance_history_limit)
            effective_weight = min(max(previous.weight, 0.0), max_weight)
            total_weight = effective_weight + 1.0
            score = (previous.score * effective_weight + score) / total_weight
            mae = (previous.mae * effective_weight + mae) / total_weight
            directional = (
                previous.directional_accuracy * effective_weight + directional
            ) / total_weight
            weight = min(total_weight, max_weight)
            updates = previous.updates + 1
        return ModelPerformanceSummary(
            mae=mae,
            directional_accuracy=directional,
            score=score,
            weight=weight,
            updates=updates,
            updated_at=now,
        )

    def _decay_summary(
        self, summary: ModelPerformanceSummary, now: datetime
    ) -> ModelPerformanceSummary:
        half_life = self._performance_half_life_hours
        if half_life is None or half_life <= 0:
            return summary
        if summary.updated_at >= now:
            return summary
        delta = now - summary.updated_at
        hours = delta.total_seconds() / 3600.0
        if hours <= 0:
            return summary
        factor = math.exp(-math.log(2.0) * (hours / half_life))
        if factor <= 0.0:
            summary.weight = 0.0
        else:
            summary.weight *= factor
        summary.updated_at = now
        return summary

    def _remove_model_summary(self, key: tuple[str, str], name: str) -> None:
        summary_map = self._model_performance.get(key)
        if summary_map is not None:
            summary_map.pop(name, None)
            if not summary_map:
                self._model_performance.pop(key, None)

    def prune_model_performance(
        self,
        *,
        before: datetime | None = None,
        older_than: timedelta | None = None,
    ) -> None:
        """Usuwa przestarzałe wpisy historii skuteczności modeli."""

        cutoff = before
        if older_than is not None:
            cutoff = self._now() - older_than
        if cutoff is None:
            return
        for key, summary_map in list(self._model_performance.items()):
            for name, summary in list(summary_map.items()):
                if summary.updated_at < cutoff:
                    self._remove_model_summary(key, name)

    def performance_snapshot(
        self, strategy: str | None = None, risk_profile: str | None = None
    ) -> Mapping[str, ModelPerformanceSummary]:
        """Zwraca kopię zagregowanych metryk skuteczności modeli."""

        key = (strategy or "__any__", risk_profile or "__any__")
        summary_map = self._model_performance.get(key)
        if not summary_map:
            return {}
        now = self._now()
        snapshot: dict[str, ModelPerformanceSummary] = {}
        for name, summary in list(summary_map.items()):
            decayed = self._decay_summary(summary, now)
            if decayed.weight <= 1e-6:
                self._remove_model_summary(key, name)
                continue
            snapshot[name] = replace(decayed)
        return snapshot

    def _now(self) -> datetime:
        return self._clock()

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


__all__ = ["DecisionOrchestrator", "ModelPerformanceSummary"]
