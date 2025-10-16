"""PortfolioGovernor odpowiedzialny za adaptacyjną alokację Stage6."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.observability.slo import SLOStatus
from bot_core.portfolio.decision_log import PortfolioDecisionLog
from bot_core.risk import StressOverrideRecommendation

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
    """Deklaracja budżetów ryzyka przypisanych do aktywów."""

    name: str
    max_var_pct: float | None = None
    max_drawdown_pct: float | None = None
    max_leverage: float | None = None
    severity: str = "warning"
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class PortfolioAssetConfig:
    """Konfiguracja aktywa zarządzanego przez PortfolioGovernor."""

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
    """Reguła nakładająca ograniczenia w reakcji na statusy SLO."""

    slo_name: str
    apply_on: Sequence[str] = field(default_factory=lambda: ("warning", "breach"))
    weight_multiplier: float | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    severity: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    force_rebalance: bool = False


@dataclass(slots=True)
class PortfolioGovernorConfig:
    """Konfiguracja instancji PortfolioGovernora."""

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
    """Sugestia korekty alokacji dla aktywa."""

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
    """Informacja pomocnicza dotycząca ryzyka."""

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
    """Wynik ewaluacji alokacji portfelowej."""

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
            "adjustments": [adjustment.to_dict() for adjustment in self.adjustments],
            "advisories": [advisory.to_dict() for advisory in self.advisories],
        }


class PortfolioGovernor:
    """Kontroluje dryf alokacji portfela z wykorzystaniem metryk Stage6."""

    def __init__(
        self,
        config: PortfolioGovernorConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        decision_log: PortfolioDecisionLog | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._asset_map = {asset.symbol: asset for asset in config.assets}
        self._risk_budgets = dict(config.risk_budgets)
        self._last_rebalance: datetime | None = None
        self._slo_overrides = tuple(config.slo_overrides)
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

        overrides_by_symbol, overrides_by_budget, generic_overrides = (
            self._prepare_stress_override_index(stress_overrides)
        )

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
                        proposed_weight = min(
                            proposed_weight,
                            max(0.0, asset.min_weight or 0.0),
                        )
                        metadata["volatility_pct"] = snapshot.volatility_pct
                if asset.min_liquidity_usd is not None and snapshot.liquidity_usd is not None:
                    if snapshot.liquidity_usd < asset.min_liquidity_usd:
                        severity = "warning"
                        reasons.append(
                            "liquidity {:.0f} < floor {:.0f}".format(
                                snapshot.liquidity_usd, asset.min_liquidity_usd
                            )
                        )
                        proposed_weight = min(
                            proposed_weight,
                            max(0.0, asset.min_weight or 0.0),
                        )
                        metadata["liquidity_usd"] = snapshot.liquidity_usd
                self._evaluate_risk_budget(
                    asset, snapshot, current_weight, metadata, advisories
                )

            relevant_overrides = self._collect_stress_overrides(
                symbol,
                asset,
                overrides_by_symbol,
                overrides_by_budget,
                generic_overrides,
            )
            proposed_weight, severity, stress_force = self._apply_stress_overrides(
                asset,
                proposed_weight,
                severity,
                reasons,
                metadata,
                relevant_overrides,
            )
            proposed_weight = max(0.0, proposed_weight)
            proposed_weight, severity, slo_force = self._apply_slo_overrides(
                asset,
                proposed_weight,
                severity,
                reasons,
                metadata,
                slo_statuses,
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
            metadata = self._build_log_metadata(
                adjustments,
                advisories,
                market_data,
                stress_overrides,
                slo_statuses,
                log_context,
            )
            try:
                self._decision_log.record(decision, metadata=metadata)
            except Exception:  # pragma: no cover - logowanie nie może zatrzymać procesu
                logging.getLogger(__name__).exception(
                    "Nie udało się zapisać wpisu decision logu portfela"
                )
        return decision

    def _bounded_target(self, target: float, asset: PortfolioAssetConfig) -> float:
        bounded = float(max(0.0, target))
        if asset.min_weight is not None:
            bounded = max(bounded, float(asset.min_weight))
        if asset.max_weight is not None:
            bounded = min(bounded, float(asset.max_weight))
        return bounded

    def _prepare_stress_override_index(
        self,
        overrides: Sequence[StressOverrideRecommendation] | None,
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
                continue
            if risk_budget:
                by_budget.setdefault(risk_budget, []).append(override)
                continue
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

        ordered = sorted(
            overrides,
            key=lambda item: self._severity_rank(item.severity),
            reverse=True,
        )
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

        metrics: MutableMapping[str, float] = {
            "current_weight": current_weight,
        }
        if snapshot.volatility_pct is not None:
            metrics["volatility_pct"] = snapshot.volatility_pct
        if snapshot.max_drawdown_pct is not None:
            metrics["max_drawdown_pct"] = snapshot.max_drawdown_pct

        triggers: list[str] = []
        if budget.max_var_pct is not None and snapshot.volatility_pct is not None:
            if snapshot.volatility_pct > budget.max_var_pct:
                triggers.append(
                    "volatility {:.2f}% > limit {:.2f}%".format(
                        snapshot.volatility_pct, budget.max_var_pct
                    )
                )
        if budget.max_drawdown_pct is not None and snapshot.max_drawdown_pct is not None:
            if snapshot.max_drawdown_pct > budget.max_drawdown_pct:
                triggers.append(
                    "drawdown {:.2f}% > limit {:.2f}%".format(
                        snapshot.max_drawdown_pct, budget.max_drawdown_pct
                    )
                )
        if budget.max_leverage is not None:
            if abs(current_weight) > budget.max_leverage:
                triggers.append(
                    "weight {:.2f} > leverage {:.2f}".format(
                        abs(current_weight), budget.max_leverage
                    )
                )

        if not triggers:
            return

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
            metadata["stress_overrides"] = [override.to_dict() for override in stress_overrides]
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


__all__ = [
    "PortfolioGovernor",
    "PortfolioGovernorConfig",
    "PortfolioAssetConfig",
    "PortfolioDriftTolerance",
    "PortfolioRiskBudgetConfig",
    "PortfolioAdjustment",
    "PortfolioAdvisory",
    "PortfolioDecision",
    "PortfolioSloOverrideConfig",
]
