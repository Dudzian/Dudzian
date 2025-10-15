"""Implementacja PortfolioGovernora z Etapu 6."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, MutableMapping

from bot_core.config.models import (
    PortfolioGovernorConfig,
    PortfolioGovernorStrategyConfig,
)

from .models import (
    PortfolioRebalanceDecision,
    StrategyAllocationDecision,
    StrategyMetricsSnapshot,
)

try:  # pragma: no cover - moduł TCO jest opcjonalny
    from bot_core.tco.models import ProfileCostSummary, StrategyCostSummary, TCOReport
except Exception:  # pragma: no cover - kompatybilność między gałęziami
    ProfileCostSummary = None  # type: ignore
    StrategyCostSummary = None  # type: ignore
    TCOReport = None  # type: ignore


@dataclass(slots=True)
class _CostIndex:
    lookup: MutableMapping[tuple[str, str], float]
    default_cost: float


@dataclass(slots=True)
class _StrategyState:
    config: PortfolioGovernorStrategyConfig
    baseline_weight: float
    min_weight: float
    max_weight: float
    smoothed_alpha: float = 0.0
    smoothed_slo: float = 0.0
    risk_penalty: float = 0.0
    net_edge_bps: float = 0.0
    last_timestamp: datetime | None = None
    samples: float = 0.0
    cost_override_bps: float | None = None


class PortfolioGovernor:
    """Autonomiczny moduł zarządzania alokacją między strategiami."""

    def __init__(
        self,
        config: PortfolioGovernorConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._states: dict[str, _StrategyState] = {}
        self._current_weights: dict[str, float] = {}
        self._last_rebalance: datetime | None = None
        self._last_decision: PortfolioRebalanceDecision | None = None
        self._cost_index = _CostIndex(lookup={}, default_cost=max(0.0, config.default_cost_bps))
        self._initialize_states()

    # ------------------------------------------------------------------ helpers --
    def _initialize_states(self) -> None:
        strategies = self._config.strategies or {}
        for name, strategy_cfg in strategies.items():
            state = self._build_state(name, strategy_cfg)
            self._states[name] = state
        if not self._states:
            return
        self._renormalize_baselines()
        for name, state in self._states.items():
            self._current_weights[name] = state.baseline_weight

    def _build_state(
        self,
        name: str,
        strategy_cfg: PortfolioGovernorStrategyConfig,
        *,
        risk_profile: str | None = None,
    ) -> _StrategyState:
        baseline = float(strategy_cfg.baseline_weight or self._config.default_baseline_weight)
        min_weight = float(strategy_cfg.min_weight if strategy_cfg.min_weight is not None else self._config.default_min_weight)
        max_weight = float(strategy_cfg.max_weight if strategy_cfg.max_weight is not None else self._config.default_max_weight)
        if max_weight < min_weight:
            max_weight = min_weight
        baseline = min(max_weight, max(min_weight, baseline))
        if risk_profile and not strategy_cfg.risk_profile:
            strategy_cfg.risk_profile = risk_profile
        return _StrategyState(
            config=strategy_cfg,
            baseline_weight=baseline,
            min_weight=min_weight,
            max_weight=max_weight,
        )

    def _renormalize_baselines(self) -> None:
        if not self._states:
            return
        total = sum(state.baseline_weight for state in self._states.values())
        if total <= 0:
            equal = 1.0 / len(self._states)
            for state in self._states.values():
                state.baseline_weight = min(state.max_weight, max(state.min_weight, equal))
        else:
            for state in self._states.values():
                state.baseline_weight = min(
                    state.max_weight,
                    max(state.min_weight, state.baseline_weight / total),
                )

    def _ensure_state(
        self, strategy: str, risk_profile: str | None
    ) -> _StrategyState:
        state = self._states.get(strategy)
        if state is not None:
            if risk_profile and not state.config.risk_profile:
                state.config.risk_profile = risk_profile
            return state
        cfg = PortfolioGovernorStrategyConfig(
            baseline_weight=self._config.default_baseline_weight,
            min_weight=self._config.default_min_weight,
            max_weight=self._config.default_max_weight,
            baseline_max_signals=None,
            max_signal_factor=1.0,
            risk_profile=risk_profile,
        )
        state = self._build_state(strategy, cfg, risk_profile=risk_profile)
        self._states[strategy] = state
        self._renormalize_baselines()
        self._current_weights.setdefault(strategy, state.baseline_weight)
        return state

    def _coerce_snapshot(
        self,
        metrics: StrategyMetricsSnapshot | Mapping[str, float],
        *,
        timestamp: datetime | None,
    ) -> StrategyMetricsSnapshot:
        if isinstance(metrics, StrategyMetricsSnapshot):
            if timestamp is not None and metrics.timestamp != timestamp:
                return StrategyMetricsSnapshot(
                    timestamp=timestamp,
                    alpha_score=metrics.alpha_score,
                    slo_violation_rate=metrics.slo_violation_rate,
                    risk_penalty=metrics.risk_penalty,
                    cost_bps=metrics.cost_bps,
                    net_edge_bps=metrics.net_edge_bps,
                    sample_weight=metrics.sample_weight,
                    metrics=metrics.metrics,
                )
            return metrics
        payload = dict(metrics)
        ts = timestamp or self._clock()
        alpha = float(payload.get("alpha_score", payload.get("avg_confidence", 0.0)))
        slo_raw = payload.get("slo_violation_rate")
        if slo_raw is None:
            slo_raw = payload.get("allocation_error_pct", payload.get("slo_breach_pct", 0.0))
        slo = float(slo_raw or 0.0)
        risk_penalty = float(payload.get("risk_penalty", payload.get("drawdown_pct", 0.0)) or 0.0)
        cost_value = payload.get("cost_bps")
        cost_bps = float(cost_value) if cost_value not in (None, "") else None
        net_edge_value = payload.get("net_edge_bps", payload.get("net_edge", None))
        net_edge = float(net_edge_value) if net_edge_value not in (None, "") else None
        sample_weight = float(payload.get("sample_weight", 1.0) or 0.0)
        metrics_map = {
            str(key): float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float))
        }
        return StrategyMetricsSnapshot(
            timestamp=ts,
            alpha_score=alpha,
            slo_violation_rate=max(0.0, slo),
            risk_penalty=max(0.0, risk_penalty),
            cost_bps=cost_bps,
            net_edge_bps=net_edge,
            sample_weight=max(0.0, sample_weight),
            metrics=metrics_map,
        )

    def _resolve_cost(self, strategy: str, state: _StrategyState) -> float:
        if state.cost_override_bps is not None:
            return max(0.0, float(state.cost_override_bps))
        profile = state.config.risk_profile or "__total__"
        value = self._cost_index.lookup.get((strategy, profile))
        if value is None:
            value = self._cost_index.lookup.get((strategy, "__total__"))
        if value is None:
            value = self._cost_index.default_cost
        return max(0.0, float(value))

    def _distribution_weights(self, scores: Mapping[str, float]) -> dict[str, float]:
        positive = {
            name: max(0.0, float(scores.get(name, 0.0)))
            for name in self._states
        }
        total = sum(positive.values())
        if total <= 0.0:
            baseline = {
                name: state.baseline_weight
                for name, state in self._states.items()
            }
            total_baseline = sum(baseline.values())
            if total_baseline <= 0:
                equal = 1.0 / len(self._states) if self._states else 1.0
                return {name: equal for name in self._states}
            return {
                name: baseline[name] / total_baseline
                for name in self._states
            }
        return {
            name: positive.get(name, 0.0) / total
            for name in self._states
        }

    def _allocate_weights(self, scores: Mapping[str, float]) -> dict[str, float]:
        if not self._states:
            return {}
        weights = {
            name: state.min_weight
            for name, state in self._states.items()
        }
        remaining = max(0.0, 1.0 - sum(weights.values()))
        if remaining <= 1e-9:
            return weights
        distribution = self._distribution_weights(scores)
        available = {
            name: max(0.0, state.max_weight - weights[name])
            for name, state in self._states.items()
        }
        active = {name for name, avail in available.items() if avail > 1e-9}
        while active and remaining > 1e-9:
            total_share = sum(distribution[name] for name in active)
            if total_share <= 0:
                share = remaining / len(active)
                consumed = 0.0
                for name in list(active):
                    delta = min(available[name], share)
                    weights[name] += delta
                    available[name] -= delta
                    consumed += delta
                    if available[name] <= 1e-9:
                        active.remove(name)
                if consumed <= 1e-9:
                    break
                remaining = max(0.0, remaining - consumed)
                continue
            consumed = 0.0
            for name in list(active):
                share = distribution[name] / total_share
                delta = min(available[name], remaining * share)
                weights[name] += delta
                available[name] -= delta
                consumed += delta
                if available[name] <= 1e-9:
                    active.remove(name)
            if consumed <= 1e-9:
                break
            remaining = max(0.0, remaining - consumed)
        return weights

    def _build_decision(
        self,
        timestamp: datetime,
        weights: Mapping[str, float],
        scores: Mapping[str, float],
        alpha: Mapping[str, float],
        slo: Mapping[str, float],
        costs: Mapping[str, float],
        remaining: float,
    ) -> PortfolioRebalanceDecision:
        metadata = {
            "remaining_cash": max(0.0, remaining),
            "require_complete_metrics": bool(self._config.require_complete_metrics),
        }
        return PortfolioRebalanceDecision(
            timestamp=timestamp,
            weights=dict(weights),
            scores=dict(scores),
            alpha_components=dict(alpha),
            slo_components=dict(slo),
            cost_components=dict(costs),
            metadata=metadata,
        )

    # ------------------------------------------------------------------ API --
    @property
    def min_signal_floor(self) -> int:
        return max(0, int(self._config.max_signal_floor))

    @property
    def current_weights(self) -> Mapping[str, float]:
        return dict(self._current_weights)

    @property
    def last_decision(self) -> PortfolioRebalanceDecision | None:
        return self._last_decision

    def observe_strategy_metrics(
        self,
        strategy: str,
        metrics: StrategyMetricsSnapshot | Mapping[str, float],
        *,
        timestamp: datetime | None = None,
        risk_profile: str | None = None,
    ) -> None:
        state = self._ensure_state(strategy, risk_profile)
        snapshot = self._coerce_snapshot(metrics, timestamp=timestamp)
        factor = min(max(self._config.smoothing, 0.0), 1.0)
        alpha = float(snapshot.alpha_score)
        slo = max(0.0, float(snapshot.slo_violation_rate))
        if state.samples <= 0 or factor >= 1.0:
            state.smoothed_alpha = alpha
            state.smoothed_slo = slo
        else:
            state.smoothed_alpha = factor * alpha + (1.0 - factor) * state.smoothed_alpha
            state.smoothed_slo = factor * slo + (1.0 - factor) * state.smoothed_slo
        state.risk_penalty = max(0.0, float(snapshot.risk_penalty))
        state.net_edge_bps = float(snapshot.net_edge_bps or 0.0)
        if snapshot.cost_bps is not None:
            state.cost_override_bps = float(snapshot.cost_bps)
        state.last_timestamp = snapshot.timestamp
        state.samples += max(0.0, snapshot.sample_weight)

    def maybe_rebalance(
        self,
        *,
        timestamp: datetime | None = None,
        force: bool = False,
    ) -> PortfolioRebalanceDecision | None:
        if not self._config.enabled or not self._states:
            return None
        now = timestamp or self._clock()
        if not force and self._last_rebalance is not None:
            interval = timedelta(minutes=max(0.0, self._config.rebalance_interval_minutes))
            if now - self._last_rebalance < interval:
                return None
        if self._config.require_complete_metrics and any(
            state.samples <= 0 for state in self._states.values()
        ):
            return None
        scores: dict[str, float] = {}
        alpha: dict[str, float] = {}
        slo: dict[str, float] = {}
        costs: dict[str, float] = {}
        for name, state in self._states.items():
            cost = self._resolve_cost(name, state)
            score = (
                state.smoothed_alpha * self._config.scoring.alpha
                - cost * self._config.scoring.cost
                - state.smoothed_slo * self._config.scoring.slo
                - state.risk_penalty * self._config.scoring.risk
            )
            if score <= self._config.min_score_threshold:
                score = 0.0
            else:
                score -= self._config.min_score_threshold
            scores[name] = max(0.0, float(score))
            alpha[name] = state.smoothed_alpha
            slo[name] = state.smoothed_slo
            costs[name] = cost
        weights = self._allocate_weights(scores)
        if not weights:
            return None
        remaining = max(0.0, 1.0 - sum(weights.values()))
        decision = self._build_decision(now, weights, scores, alpha, slo, costs, remaining)
        self._current_weights = dict(weights)
        self._last_rebalance = now
        self._last_decision = decision
        return decision

    def resolve_allocation(
        self,
        strategy: str,
        risk_profile: str | None = None,
    ) -> StrategyAllocationDecision:
        state = self._ensure_state(strategy, risk_profile)
        weight = float(self._current_weights.get(strategy, state.baseline_weight))
        baseline = state.baseline_weight or max(weight, 1e-9)
        factor = weight / baseline if baseline else 1.0
        max_factor = max(0.0, float(state.config.max_signal_factor or 0.0))
        if max_factor > 0:
            factor = min(factor, max_factor)
        metadata = {
            "weight": weight,
            "baseline_weight": baseline,
        }
        return StrategyAllocationDecision(
            strategy=strategy,
            weight=weight,
            baseline_weight=baseline,
            signal_factor=max(0.0, factor),
            max_signal_hint=state.config.baseline_max_signals,
            metadata=metadata,
        )

    # -------------------------------------------------------------- koszty --
    def set_strategy_cost(
        self,
        strategy: str,
        cost_bps: float,
        *,
        risk_profile: str | None = None,
    ) -> None:
        profile = risk_profile or "__total__"
        self._cost_index.lookup[(strategy, profile)] = max(0.0, float(cost_bps))

    def update_costs_from_report(self, report: Mapping[str, object] | object) -> None:
        lookup: MutableMapping[tuple[str, str], float] = {}
        default_cost = self._cost_index.default_cost
        if TCOReport is not None and isinstance(report, TCOReport):
            strategies = getattr(report, "strategies", {}) or {}
            default_cost = float(getattr(getattr(report, "total", None), "cost_bps", default_cost))
            for summary in getattr(report, "strategies", {}).values():
                self._ingest_strategy_summary(summary, lookup)
        else:
            data = dict(report) if isinstance(report, Mapping) else {}
            strategies_data = data.get("strategies", {}) or {}
            for strategy_name, summary_raw in strategies_data.items():
                if not isinstance(summary_raw, Mapping):
                    continue
                total_raw = summary_raw.get("total")
                if total_raw is not None:
                    lookup[(str(strategy_name), "__total__")] = self._extract_cost_bps(total_raw)
                profiles = summary_raw.get("profiles", {}) or {}
                for profile_name, profile_raw in profiles.items():
                    lookup[(str(strategy_name), str(profile_name))] = self._extract_cost_bps(profile_raw)
            total_raw = data.get("total")
            if total_raw is not None:
                default_cost = self._extract_cost_bps(total_raw)
        self._cost_index = _CostIndex(lookup=lookup, default_cost=max(0.0, float(default_cost)))

    def _ingest_strategy_summary(
        self,
        summary: object,
        lookup: MutableMapping[tuple[str, str], float],
    ) -> None:
        if StrategyCostSummary is None or not isinstance(summary, StrategyCostSummary):  # pragma: no cover - kompatybilność
            return
        lookup[(summary.strategy, "__total__")] = float(summary.total.cost_bps)
        for profile_name, profile_summary in summary.profiles.items():
            lookup[(summary.strategy, profile_name)] = float(profile_summary.cost_bps)

    def _extract_cost_bps(self, payload: object) -> float:
        if payload is None:
            return 0.0
        if ProfileCostSummary is not None and isinstance(payload, ProfileCostSummary):  # pragma: no cover - kompatybilność
            return float(payload.cost_bps)
        if isinstance(payload, Mapping):
            value = payload.get("cost_bps")
            if value is None:
                return 0.0
            return float(value)
        return float(payload)


__all__ = ["PortfolioGovernor"]
