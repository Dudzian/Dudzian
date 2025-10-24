"""Risk guardrails integrating backtest performance with risk policies."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

from bot_core.backtest.engine import BacktestReport
from bot_core.risk.base import RiskCheckResult, RiskProfile


def _maybe_percentage(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _resolve_thresholds(
    *,
    risk_profile: RiskProfile | None,
    max_drawdown_pct: float | None,
    max_exposure_pct: float | None,
    min_sortino_ratio: float | None,
    min_omega_ratio: float | None,
    max_risk_of_ruin_pct: float | None,
    min_hit_ratio_pct: float | None,
) -> tuple[Dict[str, float | None], Dict[str, str | None]]:
    thresholds: Dict[str, float | None] = {
        "max_drawdown_pct": _maybe_percentage(max_drawdown_pct),
        "max_exposure_pct": _maybe_percentage(max_exposure_pct),
        "min_sortino_ratio": _maybe_percentage(min_sortino_ratio),
        "min_omega_ratio": _maybe_percentage(min_omega_ratio),
        "max_risk_of_ruin_pct": _maybe_percentage(max_risk_of_ruin_pct),
        "min_hit_ratio_pct": _maybe_percentage(min_hit_ratio_pct),
    }
    sources: Dict[str, str | None] = {
        "max_drawdown_pct": None,
        "max_exposure_pct": None,
        "min_sortino_ratio": None,
        "min_omega_ratio": None,
        "max_risk_of_ruin_pct": None,
        "min_hit_ratio_pct": None,
    }
    if risk_profile is not None:
        profile_name = getattr(risk_profile, "name", None)
        if thresholds["max_drawdown_pct"] is None:
            try:
                thresholds["max_drawdown_pct"] = float(risk_profile.drawdown_limit()) * 100.0
                sources["max_drawdown_pct"] = (
                    f"risk_profile:{profile_name}" if profile_name else "risk_profile"
                )
            except Exception:  # pragma: no cover - defensive for partial profiles
                thresholds["max_drawdown_pct"] = None
        if thresholds["max_exposure_pct"] is None:
            try:
                thresholds["max_exposure_pct"] = float(
                    risk_profile.max_position_exposure()
                ) * 100.0
                sources["max_exposure_pct"] = (
                    f"risk_profile:{profile_name}" if profile_name else "risk_profile"
                )
            except Exception:  # pragma: no cover - defensive for partial profiles
                thresholds["max_exposure_pct"] = None
        if thresholds["min_sortino_ratio"] is None:
            candidate = getattr(risk_profile, "min_sortino_ratio", None)
            if callable(candidate):  # pragma: no cover - defensive; prefer attributes
                try:
                    candidate = candidate()
                except Exception:
                    candidate = None
            value = _maybe_percentage(candidate)
            if value is not None:
                thresholds["min_sortino_ratio"] = value
                sources["min_sortino_ratio"] = (
                    f"risk_profile:{profile_name}" if profile_name else "risk_profile"
                )
        if thresholds["min_omega_ratio"] is None:
            candidate = getattr(risk_profile, "min_omega_ratio", None)
            if callable(candidate):  # pragma: no cover - defensive; prefer attributes
                try:
                    candidate = candidate()
                except Exception:
                    candidate = None
            value = _maybe_percentage(candidate)
            if value is not None:
                thresholds["min_omega_ratio"] = value
                sources["min_omega_ratio"] = (
                    f"risk_profile:{profile_name}" if profile_name else "risk_profile"
                )
        if thresholds["max_risk_of_ruin_pct"] is None:
            candidate = getattr(risk_profile, "max_risk_of_ruin_pct", None)
            if callable(candidate):  # pragma: no cover - defensive; prefer attributes
                try:
                    candidate = candidate()
                except Exception:
                    candidate = None
            value = _maybe_percentage(candidate)
            if value is not None:
                thresholds["max_risk_of_ruin_pct"] = value
                sources["max_risk_of_ruin_pct"] = (
                    f"risk_profile:{profile_name}" if profile_name else "risk_profile"
                )
        if thresholds["min_hit_ratio_pct"] is None:
            candidate = getattr(risk_profile, "min_hit_ratio_pct", None)
            if callable(candidate):  # pragma: no cover - defensive; prefer attributes
                try:
                    candidate = candidate()
                except Exception:
                    candidate = None
            value = _maybe_percentage(candidate)
            if value is not None:
                thresholds["min_hit_ratio_pct"] = value
                sources["min_hit_ratio_pct"] = (
                    f"risk_profile:{profile_name}" if profile_name else "risk_profile"
                )
    return thresholds, sources


def evaluate_backtest_guardrails(
    report: BacktestReport,
    *,
    risk_profile: RiskProfile | None = None,
    max_drawdown_pct: float | None = None,
    max_exposure_pct: float | None = None,
    min_sortino_ratio: float | None = None,
    min_omega_ratio: float | None = None,
    max_risk_of_ruin_pct: float | None = None,
    min_hit_ratio_pct: float | None = None,
) -> RiskCheckResult:
    """Evaluate whether a backtest report satisfies configured risk guardrails."""

    metrics = report.metrics
    profile_name = getattr(risk_profile, "name", None) if risk_profile else None

    if metrics is None:
        thresholds, threshold_sources = _resolve_thresholds(
            risk_profile=risk_profile,
            max_drawdown_pct=max_drawdown_pct,
            max_exposure_pct=max_exposure_pct,
            min_sortino_ratio=min_sortino_ratio,
            min_omega_ratio=min_omega_ratio,
            max_risk_of_ruin_pct=max_risk_of_ruin_pct,
            min_hit_ratio_pct=min_hit_ratio_pct,
        )
        return RiskCheckResult(
            allowed=False,
            reason="Backtest report does not contain performance metrics",
            metadata={
                "observed": {},
                "thresholds": thresholds,
                "threshold_sources": threshold_sources,
                "violations": [],
                "risk_profile": profile_name,
                "strategy_metadata": dict(report.strategy_metadata),
                "warnings": list(report.warnings),
            },
        )

    thresholds, threshold_sources = _resolve_thresholds(
        risk_profile=risk_profile,
        max_drawdown_pct=max_drawdown_pct,
        max_exposure_pct=max_exposure_pct,
        min_sortino_ratio=min_sortino_ratio,
        min_omega_ratio=min_omega_ratio,
        max_risk_of_ruin_pct=max_risk_of_ruin_pct,
        min_hit_ratio_pct=min_hit_ratio_pct,
    )

    observed: Dict[str, object] = {
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "max_exposure_pct": metrics.max_exposure_pct,
        "sortino_ratio": metrics.sortino_ratio,
        "omega_ratio": metrics.omega_ratio,
        "risk_of_ruin_pct": metrics.risk_of_ruin_pct,
        "hit_ratio_pct": metrics.hit_ratio_pct,
    }
    reasons: list[str] = []
    violations: list[Dict[str, object]] = []

    strategy_metadata = dict(report.strategy_metadata)
    missing_required_raw = strategy_metadata.get("required_data_missing")
    missing_required: tuple[str, ...]
    if missing_required_raw:
        if not isinstance(missing_required_raw, (list, tuple)):
            candidates = (str(missing_required_raw),)
        else:
            candidates = tuple(str(item) for item in missing_required_raw)
        normalized: list[str] = []
        seen: set[str] = set()
        for value in candidates:
            item = value.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        missing_required = tuple(normalized)
    else:
        missing_required = ()

    observed["required_data_missing"] = missing_required

    if missing_required:
        readable_missing = ", ".join(missing_required)
        reasons.append(
            "Missing required historical data for strategy: " f"{readable_missing}"
        )
        violations.append(
            {
                "metric": "required_data_missing",
                "observed": missing_required,
                "threshold": (),
                "source": "strategy_metadata",
            }
        )

    drawdown_limit = thresholds.get("max_drawdown_pct")
    if drawdown_limit is not None and metrics.max_drawdown_pct > drawdown_limit + 1e-9:
        reasons.append(
            "Max drawdown {:.2f}% exceeds guardrail {:.2f}%".format(
                metrics.max_drawdown_pct,
                drawdown_limit,
            )
        )
        violations.append(
            {
                "metric": "max_drawdown_pct",
                "observed": metrics.max_drawdown_pct,
                "threshold": drawdown_limit,
                "source": threshold_sources.get("max_drawdown_pct"),
            }
        )

    exposure_limit = thresholds.get("max_exposure_pct")
    exposure_observed = metrics.max_exposure_pct
    if exposure_limit is not None:
        if math.isfinite(exposure_observed):
            if exposure_observed > exposure_limit + 1e-9:
                reasons.append(
                    "Max exposure {:.2f}% exceeds guardrail {:.2f}%".format(
                        exposure_observed,
                        exposure_limit,
                    )
                )
                violations.append(
                    {
                        "metric": "max_exposure_pct",
                        "observed": exposure_observed,
                        "threshold": exposure_limit,
                        "source": threshold_sources.get("max_exposure_pct"),
                    }
                )
        elif exposure_observed > 0:
            reasons.append(
                "Exposure guardrail {:.2f}% breached with non-finite exposure".format(
                    exposure_limit,
                )
            )
            violations.append(
                {
                    "metric": "max_exposure_pct",
                    "observed": exposure_observed,
                    "threshold": exposure_limit,
                    "source": threshold_sources.get("max_exposure_pct"),
                }
            )

    sortino_limit = thresholds.get("min_sortino_ratio")
    sortino_observed = metrics.sortino_ratio
    if sortino_limit is not None:
        sortino_breach = False
        if math.isnan(sortino_observed):
            sortino_breach = True
        elif math.isinf(sortino_observed):
            sortino_breach = sortino_observed < 0
        elif sortino_observed + 1e-9 < sortino_limit:
            sortino_breach = True
        if sortino_breach:
            reasons.append(
                "Sortino ratio {:.2f} below guardrail {:.2f}".format(
                    sortino_observed,
                    sortino_limit,
                )
            )
            violations.append(
                {
                    "metric": "min_sortino_ratio",
                    "observed": sortino_observed,
                    "threshold": sortino_limit,
                    "source": threshold_sources.get("min_sortino_ratio"),
                }
            )

    omega_limit = thresholds.get("min_omega_ratio")
    omega_observed = metrics.omega_ratio
    if omega_limit is not None:
        omega_breach = False
        if math.isnan(omega_observed):
            omega_breach = True
        elif math.isinf(omega_observed):
            omega_breach = omega_observed < 0
        elif omega_observed + 1e-9 < omega_limit:
            omega_breach = True
        if omega_breach:
            reasons.append(
                "Omega ratio {:.2f} below guardrail {:.2f}".format(
                    omega_observed,
                    omega_limit,
                )
            )
            violations.append(
                {
                    "metric": "min_omega_ratio",
                    "observed": omega_observed,
                    "threshold": omega_limit,
                    "source": threshold_sources.get("min_omega_ratio"),
                }
            )

    ruin_limit = thresholds.get("max_risk_of_ruin_pct")
    ruin_observed = metrics.risk_of_ruin_pct
    if ruin_limit is not None:
        ruin_breach = False
        if math.isnan(ruin_observed):
            ruin_breach = True
        elif ruin_observed > ruin_limit + 1e-9:
            ruin_breach = True
        if ruin_breach:
            reasons.append(
                "Risk of ruin {:.2f}% exceeds guardrail {:.2f}%".format(
                    ruin_observed,
                    ruin_limit,
                )
            )
            violations.append(
                {
                    "metric": "max_risk_of_ruin_pct",
                    "observed": ruin_observed,
                    "threshold": ruin_limit,
                    "source": threshold_sources.get("max_risk_of_ruin_pct"),
                }
            )

    hit_limit = thresholds.get("min_hit_ratio_pct")
    hit_observed = metrics.hit_ratio_pct
    if hit_limit is not None:
        hit_breach = False
        if math.isnan(hit_observed):
            hit_breach = True
        elif hit_observed + 1e-9 < hit_limit:
            hit_breach = True
        if hit_breach:
            reasons.append(
                "Hit ratio {:.2f}% below guardrail {:.2f}%".format(
                    hit_observed,
                    hit_limit,
                )
            )
            violations.append(
                {
                    "metric": "min_hit_ratio_pct",
                    "observed": hit_observed,
                    "threshold": hit_limit,
                    "source": threshold_sources.get("min_hit_ratio_pct"),
                }
            )

    metadata: Dict[str, object] = {
        "thresholds": thresholds,
        "threshold_sources": threshold_sources,
        "observed": observed,
        "strategy_metadata": strategy_metadata,
        "warnings": list(report.warnings),
        "violations": violations,
        "risk_profile": profile_name,
    }
    reason_text = "; ".join(reasons) if reasons else None
    return RiskCheckResult(allowed=not reasons, reason=reason_text, metadata=metadata)


@dataclass(slots=True)
class GuardrailSummary:
    """Aggregate view over multiple :class:`RiskCheckResult` objects."""

    total: int
    allowed: int
    blocked: int
    blocked_scenarios: list[Dict[str, Any]] = field(default_factory=list)
    metrics_violations: Dict[str, int] = field(default_factory=dict)
    warnings: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the summary to a JSON-friendly mapping."""

        return {
            "total": self.total,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "blocked_scenarios": list(self.blocked_scenarios),
            "metrics_violations": dict(self.metrics_violations),
            "warnings": dict(self.warnings),
        }


def summarize_guardrail_results(
    results: Mapping[str, RiskCheckResult]
    | Iterable[tuple[str, RiskCheckResult]]
    | Sequence[tuple[str, RiskCheckResult]]
) -> GuardrailSummary:
    """Collapse multiple guardrail checks into a single summary structure.

    Parameters
    ----------
    results:
        Either a mapping from scenario name to :class:`RiskCheckResult` or an
        iterable of ``(name, result)`` tuples.
    """

    if isinstance(results, Mapping):
        items: Iterable[tuple[str, RiskCheckResult]] = results.items()
    else:
        items = list(results)

    total = 0
    allowed = 0
    blocked = 0
    blocked_payload: list[Dict[str, Any]] = []
    metrics_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()

    for name, result in items:
        total += 1
        metadata = result.metadata or {}
        warnings = metadata.get("warnings", ())
        for warning in warnings:
            warning_counter[str(warning)] += 1

        if result.allowed:
            allowed += 1
            continue

        blocked += 1
        violations_raw = metadata.get("violations", ())
        violations: list[Dict[str, Any]] = []
        for violation in violations_raw:
            if not isinstance(violation, Mapping):
                continue
            metric = violation.get("metric")
            if metric:
                metrics_counter[str(metric)] += 1
            violations.append(dict(violation))

        blocked_payload.append(
            {
                "scenario": name,
                "reason": result.reason,
                "violations": violations,
            }
        )

    return GuardrailSummary(
        total=total,
        allowed=allowed,
        blocked=blocked,
        blocked_scenarios=blocked_payload,
        metrics_violations=dict(metrics_counter),
        warnings=dict(warning_counter),
    )


__all__ = ["evaluate_backtest_guardrails", "GuardrailSummary", "summarize_guardrail_results"]
