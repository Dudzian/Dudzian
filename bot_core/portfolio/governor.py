"""Scalony PortfolioGovernor: wariant Stage6 (asset-level z HEAD) + wariant strategy-level (z main)

Zawiera DWIE kompletne implementacje pod jedną biblioteką:

1) AssetPortfolioGovernor  (alias: PortfolioGovernor)
   - bazuje na Market Intel snapshotach
   - wspiera SLO overrides, Stress overrides
   - emituje PortfolioDecision i zapis do PortfolioDecisionLog (opcjonalnie)

2) StrategyPortfolioGovernor
   - zarządza alokacją między strategiami (alpha / SLO / risk / cost)
   - smoothing, progi, TCO/report koszty
   - emituje PortfolioRebalanceDecision i StrategyAllocationDecision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, MutableMapping, Sequence, Any
import logging

from bot_core.config.models import (
    PortfolioGovernorConfig as _StrategyPortfolioGovernorConfig,
    PortfolioGovernorScoringWeights as _PortfolioGovernorScoringWeights,
    PortfolioGovernorStrategyConfig as _PortfolioGovernorStrategyConfig,
)

# -----------------------------------------------------------------------------
# Opcjonalne zależności domenowe — zapewniamy fallbacki, by plik był samowystarczalny
# -----------------------------------------------------------------------------

try:  # decision log (opcjonalne)
    from bot_core.portfolio.decision_log import PortfolioDecisionLog  # type: ignore
except Exception:  # pragma: no cover
    class PortfolioDecisionLog:  # type: ignore
        def record(self, *_args: Any, **_kw: Any) -> None:
            pass

try:  # Market Intel snapshot (HEAD)
    from bot_core.market_intel import MarketIntelSnapshot  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(slots=True)
    class MarketIntelSnapshot:  # minimalny fallback
        symbol: str
        interval: str
        start: datetime | None
        end: datetime | None
        bar_count: int
        price_change_pct: float | None = None
        volatility_pct: float | None = None
        max_drawdown_pct: float | None = None
        average_volume: float | None = None
        liquidity_usd: float | None = None
        momentum_score: float | None = None
        metadata: Mapping[str, float] = field(default_factory=dict)

try:  # SLO status (HEAD)
    from bot_core.observability.slo import SLOStatus  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(slots=True)
    class SLOStatus:  # fallback minimalny
        status: str | None = None
        severity: str = "warning"
        error_budget_pct: float | None = None

        @property
        def is_breach(self) -> bool:
            return (self.status or "").lower() == "breach"

try:  # Stress overrides (HEAD)
    from bot_core.risk import StressOverrideRecommendation  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(slots=True)
    class StressOverrideRecommendation:  # fallback minimalny
        symbol: str | None = None
        risk_budget: str | None = None
        reason: str = ""
        severity: str | None = "warning"
        weight_multiplier: float | None = None
        min_weight: float | None = None
        max_weight: float | None = None
        force_rebalance: bool = False

# TCO/report (opcjonalne dla wariantu strategii)
try:  # pragma: no cover
    from bot_core.tco.models import ProfileCostSummary, StrategyCostSummary, TCOReport  # type: ignore
except Exception:  # pragma: no cover
    ProfileCostSummary = None  # type: ignore
    StrategyCostSummary = None  # type: ignore
    TCOReport = None  # type: ignore


# =============================================================================
# WARIANT 1 — STAGE6 (HEAD): Asset-level Portfolio Governor
# =============================================================================

_SEVERITY_ORDER = {
    "debug": -1,
    "info": 0,
    "notice": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}


@dataclass(slots=True)
class PortfolioDriftTolerance:
    """Parametry tolerancji dryfu od alokacji docelowej."""
    absolute: float = 0.01
    relative: float = 0.25


@dataclass(slots=True)
class PortfolioRiskBudgetConfig:
    """Budżety ryzyka przypisane do aktywów."""
    name: str
    max_var_pct: float | None = None
    max_drawdown_pct: float | None = None
    max_leverage: float | None = None
    severity: str = "warning"
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class PortfolioAssetConfig:
    """Konfiguracja pojedynczego aktywa."""
    symbol: str
    target_weight: float
    min_weight: float | None = None
    max_weight: float | None = None
    max_volatility_pct: float | None = None
    min_liquidity_usd: float | None = None
    risk_budget: str | None = None
    notes: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class PortfolioSloOverrideConfig:
    """Reguły reagujące na statusy SLO."""
    slo_name: str
    apply_on: Sequence[str] = field(default_factory=lambda: ("warning", "breach"))
    weight_multiplier: float | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    severity: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    force_rebalance: bool = False


@dataclass(slots=True)
class AssetPortfolioGovernorConfig:
    """Konfiguracja asset-level governora."""
    name: str
    portfolio_id: str
    drift_tolerance: PortfolioDriftTolerance = field(default_factory=PortfolioDriftTolerance)
    rebalance_cooldown_seconds: int = 900
    min_rebalance_value: float = 0.0
    min_rebalance_weight: float = 0.0
    assets: Sequence[PortfolioAssetConfig] = field(default_factory=tuple)
    risk_budgets: Mapping[str, PortfolioRiskBudgetConfig] = field(default_factory=dict)
    risk_overrides: Sequence[str] = field(default_factory=tuple)
    slo_overrides: Sequence[PortfolioSloOverrideConfig] = field(default_factory=tuple)
    market_intel_interval: str | None = None
    market_intel_lookback_bars: int = 168


@dataclass(slots=True)
class PortfolioAdjustment:
    """Sugestia korekty alokacji."""
    symbol: str
    current_weight: float
    proposed_weight: float
    reason: str
    severity: str
    metadata: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "symbol": self.symbol,
            "current_weight": self.current_weight,
            "proposed_weight": self.proposed_weight,
            "reason": self.reason,
            "severity": self.severity,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class PortfolioAdvisory:
    """Informacja o ryzyku."""
    code: str
    severity: str
    message: str
    symbols: Sequence[str] = field(default_factory=tuple)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "symbols": list(self.symbols),
        }
        if self.metrics:
            payload["metrics"] = dict(self.metrics)
        return payload


@dataclass(slots=True)
class PortfolioDecision:
    """Wynik ewaluacji alokacji portfela (asset-level)."""
    timestamp: datetime
    portfolio_id: str
    portfolio_value: float
    adjustments: Sequence[PortfolioAdjustment]
    advisories: Sequence[PortfolioAdvisory]
    rebalance_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_id": self.portfolio_id,
            "portfolio_value": self.portfolio_value,
            "rebalance_required": self.rebalance_required,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "advisories": [a.to_dict() for a in self.advisories],
        }


class AssetPortfolioGovernor:
    """Kontroluje dryf alokacji portfela (asset-level, Stage6)."""

    def __init__(
        self,
        config: AssetPortfolioGovernorConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        decision_log: PortfolioDecisionLog | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._asset_map = {asset.symbol: asset for asset in config.assets}
        self._risk_budgets = dict(config.risk_budgets)
        self._slo_overrides = tuple(config.slo_overrides)
        self._last_rebalance: datetime | None = None
        self._decision_log = decision_log

    @property
    def last_rebalance_at(self) -> datetime | None:
        return self._last_rebalance

    def acknowledge_rebalance(self, timestamp: datetime | None = None) -> None:
        self._last_rebalance = timestamp or self._clock()

    def evaluate(
        self,
        *,
        portfolio_value: float,
        allocations: Mapping[str, float],
        market_data: Mapping[str, MarketIntelSnapshot],
        stress_overrides: Sequence[StressOverrideRecommendation] | None = None,
        slo_statuses: Mapping[str, SLOStatus] | None = None,
        timestamp: datetime | None = None,
        log_context: Mapping[str, object] | None = None,
    ) -> PortfolioDecision:
        moment = timestamp or self._clock()
        adjustments: list[PortfolioAdjustment] = []
        advisories: list[PortfolioAdvisory] = []
        rebalance_required = False

        overrides_by_symbol, overrides_by_budget, generic_overrides = self._prepare_stress_index(stress_overrides)

        for symbol, asset in self._asset_map.items():
            current_weight = float(allocations.get(symbol, 0.0))
            proposed_weight = self._bounded_target(asset.target_weight, asset)
            severity = "info"
            reasons: list[str] = []
            metadata: MutableMapping[str, float] = {
                "target_weight": float(asset.target_weight),
                "current_weight": current_weight,
            }

            snapshot = market_data.get(symbol)
            if snapshot is None:
                reasons.append("brak metryk market intel")
                severity = "warning"
                proposed_weight = max(0.0, min(proposed_weight, asset.min_weight or 0.0))
                advisories.append(
                    PortfolioAdvisory(
                        code="market_intel.missing",
                        severity="warning",
                        message=f"Brak metryk market intelligence dla {symbol}",
                        symbols=(symbol,),
                    )
                )
            else:
                if asset.max_volatility_pct is not None and snapshot.volatility_pct is not None:
                    if snapshot.volatility_pct > asset.max_volatility_pct:
                        severity = "warning"
                        reasons.append(
                            "volatility {:.2f}% > limit {:.2f}%".format(
                                snapshot.volatility_pct, asset.max_volatility_pct
                            )
                        )
                        proposed_weight = min(proposed_weight, max(0.0, asset.min_weight or 0.0))
                        metadata["volatility_pct"] = float(snapshot.volatility_pct)
                if asset.min_liquidity_usd is not None and snapshot.liquidity_usd is not None:
                    if snapshot.liquidity_usd < asset.min_liquidity_usd:
                        severity = "warning"
                        reasons.append(
                            "liquidity {:.0f} < floor {:.0f}".format(
                                snapshot.liquidity_usd, asset.min_liquidity_usd
                            )
                        )
                        proposed_weight = min(proposed_weight, max(0.0, asset.min_weight or 0.0))
                        metadata["liquidity_usd"] = float(snapshot.liquidity_usd)

                self._evaluate_risk_budget(asset, snapshot, current_weight, metadata, advisories)

            relevant = self._collect_stress_overrides(
                symbol, asset, overrides_by_symbol, overrides_by_budget, generic_overrides
            )
            proposed_weight, severity, stress_force = self._apply_stress_overrides(
                asset, proposed_weight, severity, reasons, metadata, relevant
            )

            proposed_weight = max(0.0, proposed_weight)
            proposed_weight, severity, slo_force = self._apply_slo_overrides(
                asset, proposed_weight, severity, reasons, metadata, slo_statuses
            )

            tolerance = self._drift_tolerance(asset, proposed_weight)
            drift = proposed_weight - current_weight
            value_delta = abs(drift) * float(portfolio_value)
            metadata["proposed_weight"] = proposed_weight
            metadata["tolerance"] = tolerance
            metadata["value_delta"] = value_delta

            override_force = stress_force or slo_force
            if not override_force:
                if abs(drift) < tolerance:
                    continue
                if abs(drift) < self._config.min_rebalance_weight:
                    continue
                if value_delta < self._config.min_rebalance_value:
                    continue

            rebalance_required = True
            reason_text = ", ".join(reasons) if reasons else "dryf alokacji"
            adjustments.append(
                PortfolioAdjustment(
                    symbol=symbol,
                    current_weight=current_weight,
                    proposed_weight=proposed_weight,
                    reason=reason_text,
                    severity=severity,
                    metadata=dict(metadata),
                )
            )

        decision = PortfolioDecision(
            timestamp=moment,
            portfolio_id=self._config.portfolio_id,
            portfolio_value=float(portfolio_value),
            adjustments=tuple(adjustments),
            advisories=tuple(advisories),
            rebalance_required=rebalance_required,
        )
        if rebalance_required:
            self._last_rebalance = moment

        if self._decision_log is not None:
            meta = self._build_log_metadata(
                adjustments, advisories, market_data, stress_overrides, slo_statuses, log_context
            )
            try:
                self._decision_log.record(decision, metadata=meta)
            except Exception:  # pragma: no cover
                logging.getLogger(__name__).exception("Nie udało się zapisać wpisu decision logu portfela")

        return decision

    # ------------------------ helpers (asset-level) ------------------------

    def _bounded_target(self, target: float, asset: PortfolioAssetConfig) -> float:
        bounded = float(max(0.0, target))
        if asset.min_weight is not None:
            bounded = max(bounded, float(asset.min_weight))
        if asset.max_weight is not None:
            bounded = min(bounded, float(asset.max_weight))
        return bounded

    def _prepare_stress_index(
        self, overrides: Sequence[StressOverrideRecommendation] | None,
    ) -> tuple[
        Mapping[str, Sequence[StressOverrideRecommendation]],
        Mapping[str, Sequence[StressOverrideRecommendation]],
        Sequence[StressOverrideRecommendation],
    ]:
        if not overrides:
            return {}, {}, ()
        by_symbol: MutableMapping[str, list[StressOverrideRecommendation]] = {}
        by_budget: MutableMapping[str, list[StressOverrideRecommendation]] = {}
        generic: list[StressOverrideRecommendation] = []
        for override in overrides:
            symbol = (override.symbol or "").strip()
            risk_budget = (override.risk_budget or "").strip()
            if symbol:
                by_symbol.setdefault(symbol, []).append(override)
            elif risk_budget:
                by_budget.setdefault(risk_budget, []).append(override)
            else:
                generic.append(override)
        return by_symbol, by_budget, tuple(generic)

    def _collect_stress_overrides(
        self,
        symbol: str,
        asset: PortfolioAssetConfig,
        overrides_by_symbol: Mapping[str, Sequence[StressOverrideRecommendation]],
        overrides_by_budget: Mapping[str, Sequence[StressOverrideRecommendation]],
        generic_overrides: Sequence[StressOverrideRecommendation],
    ) -> Sequence[StressOverrideRecommendation]:
        candidates: list[StressOverrideRecommendation] = []
        symbol_overrides = overrides_by_symbol.get(symbol, ())
        if symbol_overrides:
            candidates.extend(symbol_overrides)
        if asset.risk_budget:
            budget_overrides = overrides_by_budget.get(asset.risk_budget, ())
            if budget_overrides:
                candidates.extend(budget_overrides)
        if generic_overrides:
            candidates.extend(generic_overrides)
        return tuple(candidates)

    def _apply_stress_overrides(
        self,
        asset: PortfolioAssetConfig,
        proposed_weight: float,
        severity: str,
        reasons: list[str],
        metadata: MutableMapping[str, float],
        overrides: Sequence[StressOverrideRecommendation],
    ) -> tuple[float, str, bool]:
        if not overrides:
            return proposed_weight, severity, False

        ordered = sorted(overrides, key=lambda item: self._severity_rank(item.severity), reverse=True)
        metadata["stress::count"] = float(len(ordered))
        override_force = False

        for index, override in enumerate(ordered, start=1):
            reasons.append(f"stress::{override.reason}")
            prefix = f"stress::{index}"
            metadata[f"{prefix}::severity_rank"] = float(self._severity_rank(override.severity))

            candidate_severity = override.severity or "warning"
            severity = self._combine_severity(severity, candidate_severity)

            adjusted = proposed_weight
            if override.weight_multiplier is not None:
                metadata[f"{prefix}::weight_multiplier"] = float(override.weight_multiplier)
                adjusted *= override.weight_multiplier
            if override.min_weight is not None:
                metadata[f"{prefix}::min_weight"] = float(override.min_weight)
                adjusted = max(adjusted, override.min_weight)
            if override.max_weight is not None:
                metadata[f"{prefix}::max_weight"] = float(override.max_weight)
                adjusted = min(adjusted, override.max_weight)

            proposed_weight = self._bounded_target(adjusted, asset)

            if override.force_rebalance:
                override_force = True
                metadata[f"{prefix}::force_rebalance"] = 1.0

        return proposed_weight, severity, override_force

    def _apply_slo_overrides(
        self,
        asset: PortfolioAssetConfig,
        proposed_weight: float,
        severity: str,
        reasons: list[str],
        metadata: MutableMapping[str, float],
        slo_statuses: Mapping[str, SLOStatus] | None,
    ) -> tuple[float, str, bool]:
        if not slo_statuses or not self._slo_overrides:
            return proposed_weight, severity, False

        override_force = False
        for override in self._slo_overrides:
            status = slo_statuses.get(override.slo_name)
            if status is None:
                continue
            status_value = (status.status or "").lower()
            valid_statuses = {item.lower() for item in override.apply_on}
            if status_value not in valid_statuses:
                continue
            if override.tags and not any(tag in asset.tags for tag in override.tags):
                continue

            reasons.append(f"SLO {override.slo_name}:{status_value}")
            key_prefix = f"slo::{override.slo_name}"
            if status.error_budget_pct is not None:
                metadata[f"{key_prefix}::error_budget_pct"] = status.error_budget_pct
            metadata[key_prefix] = (
                status.error_budget_pct if status.error_budget_pct is not None else 1.0
            )

            candidate_severity = override.severity or status.severity
            severity = self._combine_severity(severity, candidate_severity)

            adjusted = proposed_weight
            if override.weight_multiplier is not None:
                adjusted *= override.weight_multiplier
            if override.min_weight is not None:
                adjusted = max(adjusted, override.min_weight)
            if override.max_weight is not None:
                adjusted = min(adjusted, override.max_weight)
            proposed_weight = self._bounded_target(adjusted, asset)

            if override.force_rebalance or status.is_breach:
                override_force = True
                metadata[f"{key_prefix}::force_rebalance"] = 1.0

        return proposed_weight, severity, override_force

    def _combine_severity(self, current: str, candidate: str) -> str:
        normalized_current = (current or "info").lower()
        normalized_candidate = (candidate or "warning").lower()
        current_rank = _SEVERITY_ORDER.get(normalized_current, 1)
        candidate_rank = _SEVERITY_ORDER.get(normalized_candidate, 1)
        if candidate_rank > current_rank:
            return normalized_candidate
        return normalized_current

    def _severity_rank(self, severity: str | None) -> int:
        if not severity:
            return _SEVERITY_ORDER.get("warning", 1)
        return _SEVERITY_ORDER.get(severity.lower(), 1)

    def _drift_tolerance(self, asset: PortfolioAssetConfig, proposed: float) -> float:
        absolute = max(0.0, self._config.drift_tolerance.absolute)
        baseline = max(proposed, float(asset.target_weight))
        relative = baseline * max(0.0, self._config.drift_tolerance.relative)
        return max(absolute, relative)

    def _evaluate_risk_budget(
        self,
        asset: PortfolioAssetConfig,
        snapshot: MarketIntelSnapshot,
        current_weight: float,
        metadata: MutableMapping[str, float],
        advisories: list[PortfolioAdvisory],
    ) -> None:
        if not asset.risk_budget:
            return
        budget = self._risk_budgets.get(asset.risk_budget)
        if not budget:
            return

        metrics: MutableMapping[str, float] = {"current_weight": current_weight}
        if snapshot.volatility_pct is not None:
            metrics["volatility_pct"] = float(snapshot.volatility_pct)
        if snapshot.max_drawdown_pct is not None:
            metrics["max_drawdown_pct"] = float(snapshot.max_drawdown_pct)

        triggers: list[str] = []
        if budget.max_var_pct is not None and snapshot.volatility_pct is not None:
            if float(snapshot.volatility_pct) > budget.max_var_pct:
                triggers.append(
                    "volatility {:.2f}% > limit {:.2f}%".format(
                        float(snapshot.volatility_pct), budget.max_var_pct
                    )
                )
        if budget.max_drawdown_pct is not None and snapshot.max_drawdown_pct is not None:
            if float(snapshot.max_drawdown_pct) > budget.max_drawdown_pct:
                triggers.append(
                    "drawdown {:.2f}% > limit {:.2f}%".format(
                        float(snapshot.max_drawdown_pct), budget.max_drawdown_pct
                    )
                )
        if budget.max_leverage is not None:
            if abs(current_weight) > budget.max_leverage:
                triggers.append("weight {:.2f} > leverage {:.2f}".format(abs(current_weight), budget.max_leverage))

        if triggers:
            advisories.append(
                PortfolioAdvisory(
                    code=f"risk_budget.{budget.name}",
                    severity=budget.severity,
                    message="; ".join(triggers),
                    symbols=(asset.symbol,),
                    metrics=dict(metrics),
                )
            )

    def _build_log_metadata(
        self,
        adjustments: Sequence[PortfolioAdjustment],
        advisories: Sequence[PortfolioAdvisory],
        market_data: Mapping[str, MarketIntelSnapshot],
        stress_overrides: Sequence[StressOverrideRecommendation] | None,
        slo_statuses: Mapping[str, SLOStatus] | None,
        log_context: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        metadata: MutableMapping[str, object] = {
            "asset_coverage": len(self._asset_map),
            "adjustment_count": len(adjustments),
            "advisory_count": len(advisories),
        }
        missing = [symbol for symbol in self._asset_map if symbol not in market_data]
        if missing:
            metadata["missing_market_intel"] = missing
        if stress_overrides:
            # nie zakładamy .to_dict() — bierzemy __dict__ gdy dostępne
            metadata["stress_overrides"] = [
                getattr(override, "to_dict", lambda: getattr(override, "__dict__", {}))()
                for override in stress_overrides
            ]
        if slo_statuses:
            metadata["slo_statuses"] = {
                name: {
                    "status": status.status,
                    "severity": status.severity,
                    "error_budget_pct": status.error_budget_pct,
                }
                for name, status in slo_statuses.items()
            }
        if log_context:
            for key, value in log_context.items():
                metadata[str(key)] = value
        return metadata


# =============================================================================
# WARIANT 2 — Strategy-level Portfolio Governor (main)
# =============================================================================

PortfolioGovernorScoringWeights = _PortfolioGovernorScoringWeights
PortfolioGovernorStrategyConfig = _PortfolioGovernorStrategyConfig
StrategyPortfolioGovernorConfig = _StrategyPortfolioGovernorConfig


@dataclass(slots=True)
class StrategyMetricsSnapshot:
    timestamp: datetime
    alpha_score: float
    slo_violation_rate: float
    risk_penalty: float
    cost_bps: float | None = None
    net_edge_bps: float | None = None
    sample_weight: float = 1.0
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioRebalanceDecision:
    timestamp: datetime
    weights: Mapping[str, float]
    scores: Mapping[str, float]
    alpha_components: Mapping[str, float]
    slo_components: Mapping[str, float]
    cost_components: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyAllocationDecision:
    strategy: str
    weight: float
    baseline_weight: float
    signal_factor: float
    max_signal_hint: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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


class StrategyPortfolioGovernor:
    """Autonomiczny moduł zarządzania alokacją między strategiami."""

    def __init__(self, config: StrategyPortfolioGovernorConfig, *, clock: Callable[[], datetime] | None = None) -> None:
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

    def _ensure_state(self, strategy: str, risk_profile: str | None) -> _StrategyState:
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

    def maybe_rebalance(self, *, timestamp: datetime | None = None, force: bool = False) -> PortfolioRebalanceDecision | None:
        if not self._config.enabled or not self._states:
            return None
        now = timestamp or self._clock()
        if not force and self._last_rebalance is not None:
            interval = timedelta(minutes=max(0.0, self._config.rebalance_interval_minutes))
            if now - self._last_rebalance < interval:
                return None
        if self._config.require_complete_metrics and any(state.samples <= 0 for state in self._states.values()):
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

    def resolve_allocation(self, strategy: str, risk_profile: str | None = None) -> StrategyAllocationDecision:
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
    def set_strategy_cost(self, strategy: str, cost_bps: float, *, risk_profile: str | None = None) -> None:
        profile = risk_profile or "__total__"
        self._cost_index.lookup[(strategy, profile)] = max(0.0, float(cost_bps))

    def update_costs_from_report(self, report: Mapping[str, object] | object) -> None:
        lookup: MutableMapping[tuple[str, str], float] = {}
        default_cost = self._cost_index.default_cost
        if TCOReport is not None and isinstance(report, TCOReport):  # pragma: no cover
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

    def _ingest_strategy_summary(self, summary: object, lookup: MutableMapping[tuple[str, str], float]) -> None:
        if StrategyCostSummary is None or not isinstance(summary, StrategyCostSummary):  # pragma: no cover
            return
        lookup[(summary.strategy, "__total__")] = float(summary.total.cost_bps)
        for profile_name, profile_summary in summary.profiles.items():
            lookup[(summary.strategy, profile_name)] = float(profile_summary.cost_bps)

    def _extract_cost_bps(self, payload: object) -> float:
        if payload is None:
            return 0.0
        if ProfileCostSummary is not None and isinstance(payload, ProfileCostSummary):  # pragma: no cover
            return float(payload.cost_bps)
        if isinstance(payload, Mapping):
            value = payload.get("cost_bps")
            if value is None:
                return 0.0
            return float(value)
        return float(payload)


# =============================================================================
# Publiczne symbole
# =============================================================================

# Zgodność z HEAD: alias o tej samej nazwie
def _coerce_strategy_config(
    config: StrategyPortfolioGovernorConfig | Mapping[str, object] | object,
) -> StrategyPortfolioGovernorConfig:
    if isinstance(config, StrategyPortfolioGovernorConfig):
        return config

    base = StrategyPortfolioGovernorConfig()
    scoring_cfg = getattr(config, "scoring", None)
    if isinstance(scoring_cfg, PortfolioGovernorScoringWeights):
        scoring = PortfolioGovernorScoringWeights(
            alpha=float(scoring_cfg.alpha),
            cost=float(scoring_cfg.cost),
            slo=float(scoring_cfg.slo),
            risk=float(scoring_cfg.risk),
        )
    else:
        scoring = PortfolioGovernorScoringWeights(
            alpha=float(getattr(scoring_cfg, "alpha", base.scoring.alpha)) if scoring_cfg is not None else base.scoring.alpha,
            cost=float(getattr(scoring_cfg, "cost", base.scoring.cost)) if scoring_cfg is not None else base.scoring.cost,
            slo=float(getattr(scoring_cfg, "slo", base.scoring.slo)) if scoring_cfg is not None else base.scoring.slo,
            risk=float(getattr(scoring_cfg, "risk", base.scoring.risk)) if scoring_cfg is not None else base.scoring.risk,
        )

    default_baseline = float(getattr(config, "default_baseline_weight", base.default_baseline_weight))
    default_min = float(getattr(config, "default_min_weight", base.default_min_weight))
    default_max = float(getattr(config, "default_max_weight", base.default_max_weight))

    strategies_raw = getattr(config, "strategies", {}) or {}
    converted: dict[str, PortfolioGovernorStrategyConfig] = {}
    for name, raw_cfg in dict(strategies_raw).items():
        baseline = getattr(raw_cfg, "baseline_weight", default_baseline)
        min_weight = getattr(raw_cfg, "min_weight", default_min)
        max_weight = getattr(raw_cfg, "max_weight", default_max)
        baseline_signals = getattr(raw_cfg, "baseline_max_signals", None)
        max_factor = getattr(raw_cfg, "max_signal_factor", 1.0)
        risk_profile = getattr(raw_cfg, "risk_profile", None)
        tags_value = getattr(raw_cfg, "tags", ())
        converted[name] = PortfolioGovernorStrategyConfig(
            baseline_weight=float(baseline),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
            baseline_max_signals=None if baseline_signals is None else int(baseline_signals),
            max_signal_factor=float(max_factor),
            risk_profile=str(risk_profile) if isinstance(risk_profile, str) else None,
            tags=tuple(tags_value) if isinstance(tags_value, Sequence) else tuple(),
        )

    return StrategyPortfolioGovernorConfig(
        enabled=bool(getattr(config, "enabled", base.enabled)),
        rebalance_interval_minutes=float(
            getattr(config, "rebalance_interval_minutes", base.rebalance_interval_minutes)
        ),
        smoothing=float(getattr(config, "smoothing", base.smoothing)),
        scoring=scoring,
        strategies=converted,
        default_baseline_weight=default_baseline,
        default_min_weight=default_min,
        default_max_weight=default_max,
        require_complete_metrics=bool(
            getattr(config, "require_complete_metrics", base.require_complete_metrics)
        ),
        min_score_threshold=float(
            getattr(config, "min_score_threshold", base.min_score_threshold)
        ),
        default_cost_bps=float(getattr(config, "default_cost_bps", base.default_cost_bps)),
        max_signal_floor=int(getattr(config, "max_signal_floor", base.max_signal_floor)),
    )


def PortfolioGovernor(
    config: AssetPortfolioGovernorConfig | StrategyPortfolioGovernorConfig | object,
    *,
    clock: Callable[[], datetime] | None = None,
    decision_log: PortfolioDecisionLog | None = None,
):
    if hasattr(config, "assets"):
        if isinstance(config, AssetPortfolioGovernorConfig):
            asset_cfg = config
        else:
            drift_value = getattr(config, "drift_tolerance", None)
            if isinstance(drift_value, Mapping):
                drift_value = PortfolioDriftTolerance(
                    absolute=float(drift_value.get("absolute", 0.01)),
                    relative=float(drift_value.get("relative", 0.25)),
                )
            if not isinstance(drift_value, PortfolioDriftTolerance):
                drift_value = PortfolioDriftTolerance()

            risk_budgets_value = getattr(config, "risk_budgets", {})
            if isinstance(risk_budgets_value, Mapping):
                risk_budgets = dict(risk_budgets_value)
            else:
                risk_budgets = {}

            slo_overrides_value = getattr(config, "slo_overrides", ())
            if isinstance(slo_overrides_value, Sequence) and not isinstance(
                slo_overrides_value, (str, bytes)
            ):
                slo_overrides = tuple(slo_overrides_value)
            else:
                slo_overrides = tuple()

            risk_overrides_value = getattr(config, "risk_overrides", ())
            if isinstance(risk_overrides_value, Sequence) and not isinstance(
                risk_overrides_value, (str, bytes)
            ):
                risk_overrides = tuple(str(item) for item in risk_overrides_value)
            else:
                risk_overrides = tuple()

            asset_cfg = AssetPortfolioGovernorConfig(
                name=str(getattr(config, "name", getattr(config, "portfolio_id", "portfolio"))),
                portfolio_id=str(getattr(config, "portfolio_id", getattr(config, "name", "portfolio"))),
                drift_tolerance=drift_value,
                rebalance_cooldown_seconds=int(
                    getattr(config, "rebalance_cooldown_seconds", 900)
                ),
                min_rebalance_value=float(
                    getattr(config, "min_rebalance_value", 0.0)
                ),
                min_rebalance_weight=float(
                    getattr(config, "min_rebalance_weight", 0.0)
                ),
                assets=tuple(getattr(config, "assets", ())),
                risk_budgets=risk_budgets,
                risk_overrides=risk_overrides,
                slo_overrides=slo_overrides,
                market_intel_interval=getattr(config, "market_intel_interval", None),
                market_intel_lookback_bars=int(
                    getattr(config, "market_intel_lookback_bars", 168)
                ),
            )

        return AssetPortfolioGovernor(asset_cfg, clock=clock, decision_log=decision_log)

    strategy_cfg = _coerce_strategy_config(config)
    return StrategyPortfolioGovernor(strategy_cfg, clock=clock)

__all__ = [
    # Asset-level (HEAD)
    "AssetPortfolioGovernor",
    "AssetPortfolioGovernorConfig",
    "PortfolioGovernor",  # alias
    "PortfolioDriftTolerance",
    "PortfolioRiskBudgetConfig",
    "PortfolioAssetConfig",
    "PortfolioSloOverrideConfig",
    "PortfolioAdjustment",
    "PortfolioAdvisory",
    "PortfolioDecision",
    # Strategy-level (main)
    "StrategyPortfolioGovernor",
    "StrategyPortfolioGovernorConfig",
    "PortfolioGovernorScoringWeights",
    "PortfolioGovernorStrategyConfig",
    "PortfolioRebalanceDecision",
    "StrategyAllocationDecision",
    "StrategyMetricsSnapshot",
]
