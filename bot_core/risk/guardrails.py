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
    warning_scenarios: Dict[str, list[str]] = field(default_factory=dict)
    reason_counts: Dict[str, int] = field(default_factory=dict)
    reason_scenarios: Dict[str, list[str]] = field(default_factory=dict)
    profile_counts: Dict[str, int] = field(default_factory=dict)
    profile_block_counts: Dict[str, int] = field(default_factory=dict)
    profile_pass_counts: Dict[str, int] = field(default_factory=dict)
    profile_warning_counts: Dict[str, int] = field(default_factory=dict)
    profile_violation_counts: Dict[str, int] = field(default_factory=dict)
    profile_reason_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    profile_reason_scenarios: Dict[str, Dict[str, list[str]]] = field(default_factory=dict)
    profile_metric_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    profile_metric_scenarios: Dict[str, Dict[str, list[str]]] = field(default_factory=dict)
    metric_violation_scenarios: Dict[str, list[str]] = field(default_factory=dict)
    metric_violation_details: Dict[str, list[Dict[str, Any]]] = field(default_factory=dict)
    profile_metric_details: Dict[str, Dict[str, list[Dict[str, Any]]]] = field(
        default_factory=dict
    )
    profile_warning_messages: Dict[str, Dict[str, int]] = field(default_factory=dict)
    profile_warning_message_scenarios: Dict[str, Dict[str, list[str]]] = field(
        default_factory=dict
    )
    profile_warning_scenarios: Dict[str, list[str]] = field(default_factory=dict)
    profile_pass_scenarios: Dict[str, list[str]] = field(default_factory=dict)
    profile_block_scenarios: Dict[str, list[str]] = field(default_factory=dict)
    profile_violation_scenarios: Dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the summary to a JSON-friendly mapping."""

        return {
            "total": self.total,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "blocked_scenarios": list(self.blocked_scenarios),
            "metrics_violations": dict(self.metrics_violations),
            "warnings": dict(self.warnings),
            "warning_scenarios": {
                key: list(value) for key, value in self.warning_scenarios.items()
            },
            "reason_counts": dict(self.reason_counts),
            "reason_scenarios": {
                key: list(value) for key, value in self.reason_scenarios.items()
            },
            "profile_counts": dict(self.profile_counts),
            "profile_block_counts": dict(self.profile_block_counts),
            "profile_pass_counts": dict(self.profile_pass_counts),
            "profile_warning_counts": dict(self.profile_warning_counts),
            "profile_violation_counts": dict(self.profile_violation_counts),
            "profile_reason_counts": {
                key: dict(value) for key, value in self.profile_reason_counts.items()
            },
            "profile_reason_scenarios": {
                profile: {
                    reason: list(scenarios)
                    for reason, scenarios in reason_map.items()
                }
                for profile, reason_map in self.profile_reason_scenarios.items()
            },
            "profile_metric_counts": {
                key: dict(value) for key, value in self.profile_metric_counts.items()
            },
            "profile_metric_scenarios": {
                profile: {
                    metric: list(scenarios)
                    for metric, scenarios in metric_map.items()
                }
                for profile, metric_map in self.profile_metric_scenarios.items()
            },
            "metric_violation_scenarios": {
                key: list(value) for key, value in self.metric_violation_scenarios.items()
            },
            "metric_violation_details": {
                key: [dict(item) for item in value]
                for key, value in self.metric_violation_details.items()
            },
            "profile_metric_details": {
                profile: {
                    metric: [dict(item) for item in details]
                    for metric, details in metric_map.items()
                }
                for profile, metric_map in self.profile_metric_details.items()
            },
            "profile_warning_messages": {
                key: dict(value)
                for key, value in self.profile_warning_messages.items()
            },
            "profile_warning_message_scenarios": {
                profile: {
                    message: list(scenarios)
                    for message, scenarios in message_map.items()
                }
                for profile, message_map in self.profile_warning_message_scenarios.items()
            },
            "profile_warning_scenarios": {
                key: list(value)
                for key, value in self.profile_warning_scenarios.items()
            },
            "profile_pass_scenarios": {
                key: list(value)
                for key, value in self.profile_pass_scenarios.items()
            },
            "profile_block_scenarios": {
                key: list(value)
                for key, value in self.profile_block_scenarios.items()
            },
            "profile_violation_scenarios": {
                key: list(value)
                for key, value in self.profile_violation_scenarios.items()
            },
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
    warning_scenario_map: Dict[str, list[str]] = {}
    reason_counter: Counter[str] = Counter()
    reason_scenario_map: Dict[str, list[str]] = {}
    profile_counter: Counter[str] = Counter()
    profile_block_counter: Counter[str] = Counter()
    profile_pass_counter: Counter[str] = Counter()
    profile_warning_counter: Counter[str] = Counter()
    profile_violation_counter: Counter[str] = Counter()
    profile_reason_counter: Dict[str, Counter[str]] = {}
    profile_reason_scenario_map: Dict[str, Dict[str, list[str]]] = {}
    profile_metric_counter: Dict[str, Counter[str]] = {}
    profile_metric_scenario_map: Dict[str, Dict[str, list[str]]] = {}
    metric_violation_detail_map: Dict[str, list[Dict[str, Any]]] = {}
    profile_metric_detail_map: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
    profile_warning_message_counter: Dict[str, Counter[str]] = {}
    profile_warning_scenario_map: Dict[str, list[str]] = {}
    profile_warning_message_scenario_map: Dict[str, Dict[str, list[str]]] = {}
    profile_pass_scenario_map: Dict[str, list[str]] = {}
    profile_block_scenario_map: Dict[str, list[str]] = {}
    profile_violation_scenario_map: Dict[str, list[str]] = {}
    metric_violation_scenario_map: Dict[str, list[str]] = {}

    for name, result in items:
        total += 1
        metadata = result.metadata or {}
        warnings = metadata.get("warnings", ())
        warning_messages: list[str] = []
        for warning in warnings:
            message = str(warning)
            warning_counter[message] += 1
            scenario_bucket = warning_scenario_map.setdefault(message, [])
            if name not in scenario_bucket:
                scenario_bucket.append(name)
            warning_messages.append(message)
        warning_count = len(warning_messages)

        profile_name = metadata.get("risk_profile")
        normalized_profile: str | None = None
        if profile_name:
            normalized_profile = str(profile_name)
            profile_counter[normalized_profile] += 1
            profile_warning_counter.setdefault(normalized_profile, 0)
            profile_violation_counter.setdefault(normalized_profile, 0)
            if warning_count:
                profile_warning_counter[normalized_profile] += warning_count
        elif warning_count:
            profile_warning_counter["unassigned"] += warning_count
        if warning_count:
            bucket = profile_warning_message_counter.setdefault(
                normalized_profile or "unassigned", Counter()
            )
            for message in warning_messages:
                bucket[message] += 1
                message_scenario_bucket = profile_warning_message_scenario_map.setdefault(
                    normalized_profile or "unassigned", {}
                )
                scenario_bucket = message_scenario_bucket.setdefault(message, [])
                if name not in scenario_bucket:
                    scenario_bucket.append(name)
            scenario_bucket = profile_warning_scenario_map.setdefault(
                normalized_profile or "unassigned", []
            )
            if name not in scenario_bucket:
                scenario_bucket.append(name)

        if result.allowed:
            allowed += 1
            if normalized_profile is not None:
                profile_pass_counter[normalized_profile] += 1
                pass_bucket = profile_pass_scenario_map.setdefault(
                    normalized_profile, []
                )
                if name not in pass_bucket:
                    pass_bucket.append(name)
            else:
                pass_bucket = profile_pass_scenario_map.setdefault("unassigned", [])
                if name not in pass_bucket:
                    pass_bucket.append(name)
            continue

        blocked += 1
        if normalized_profile is not None:
            profile_block_counter[normalized_profile] += 1
            block_bucket = profile_block_scenario_map.setdefault(
                normalized_profile, []
            )
            if name not in block_bucket:
                block_bucket.append(name)
        if result.reason:
            reason = str(result.reason)
            reason_counter[reason] += 1
            scenario_bucket = reason_scenario_map.setdefault(reason, [])
            if name not in scenario_bucket:
                scenario_bucket.append(name)
            bucket = profile_reason_counter.setdefault(
                normalized_profile or "unassigned", Counter()
            )
            bucket[reason] += 1
            profile_reason_bucket = profile_reason_scenario_map.setdefault(
                normalized_profile or "unassigned", {}
            )
            reason_profile_bucket = profile_reason_bucket.setdefault(reason, [])
            if name not in reason_profile_bucket:
                reason_profile_bucket.append(name)
        violations_raw = metadata.get("violations", ())
        violations: list[Dict[str, Any]] = []
        violation_count = 0
        for violation in violations_raw:
            if not isinstance(violation, Mapping):
                continue
            metric = violation.get("metric")
            if metric:
                metrics_counter[str(metric)] += 1
                bucket = profile_metric_counter.setdefault(
                    normalized_profile or "unassigned", Counter()
                )
                bucket[str(metric)] += 1
                scenario_bucket = metric_violation_scenario_map.setdefault(
                    str(metric), []
                )
                if name not in scenario_bucket:
                    scenario_bucket.append(name)
                profile_metric_bucket = profile_metric_scenario_map.setdefault(
                    normalized_profile or "unassigned", {}
                )
                metric_profile_bucket = profile_metric_bucket.setdefault(str(metric), [])
                if name not in metric_profile_bucket:
                    metric_profile_bucket.append(name)
                detail_entry = dict(violation)
                detail_entry.setdefault("scenario", name)
                details = metric_violation_detail_map.setdefault(str(metric), [])
                details.append(detail_entry)
                profile_detail_bucket = profile_metric_detail_map.setdefault(
                    normalized_profile or "unassigned", {}
                )
                metric_detail_bucket = profile_detail_bucket.setdefault(str(metric), [])
                metric_detail_bucket.append(detail_entry)
            violations.append(dict(violation))
            violation_count += 1

        if normalized_profile is not None:
            profile_violation_counter[normalized_profile] += violation_count
        elif violation_count:
            profile_violation_counter["unassigned"] += violation_count

        if violation_count:
            violation_bucket = profile_violation_scenario_map.setdefault(
                normalized_profile or "unassigned", []
            )
            if name not in violation_bucket:
                violation_bucket.append(name)

        if normalized_profile is None:
            block_bucket = profile_block_scenario_map.setdefault("unassigned", [])
            if name not in block_bucket:
                block_bucket.append(name)

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
        warning_scenarios={
            key: list(scenarios) for key, scenarios in warning_scenario_map.items()
        },
        reason_counts=dict(reason_counter),
        reason_scenarios={
            key: list(scenarios) for key, scenarios in reason_scenario_map.items()
        },
        profile_counts=dict(profile_counter),
        profile_block_counts=dict(profile_block_counter),
        profile_pass_counts=dict(profile_pass_counter),
        profile_warning_counts=dict(profile_warning_counter),
        profile_violation_counts=dict(profile_violation_counter),
        profile_reason_counts={
            key: dict(counter) for key, counter in profile_reason_counter.items()
        },
        profile_reason_scenarios={
            profile: {
                reason: list(scenarios)
                for reason, scenarios in reason_map.items()
            }
            for profile, reason_map in profile_reason_scenario_map.items()
        },
        profile_metric_counts={
            key: dict(counter) for key, counter in profile_metric_counter.items()
        },
        profile_metric_scenarios={
            profile: {
                metric: list(scenarios)
                for metric, scenarios in metric_map.items()
            }
            for profile, metric_map in profile_metric_scenario_map.items()
        },
        metric_violation_scenarios={
            key: list(scenarios)
            for key, scenarios in metric_violation_scenario_map.items()
        },
        metric_violation_details={
            key: [dict(item) for item in details]
            for key, details in metric_violation_detail_map.items()
        },
        profile_metric_details={
            profile: {
                metric: [dict(item) for item in details]
                for metric, details in detail_map.items()
            }
            for profile, detail_map in profile_metric_detail_map.items()
        },
        profile_warning_messages={
            key: dict(counter)
            for key, counter in profile_warning_message_counter.items()
        },
        profile_warning_message_scenarios={
            profile: {
                message: list(scenarios)
                for message, scenarios in message_map.items()
            }
            for profile, message_map in profile_warning_message_scenario_map.items()
        },
        profile_warning_scenarios={
            key: list(scenarios)
            for key, scenarios in profile_warning_scenario_map.items()
        },
        profile_pass_scenarios={
            key: list(scenarios)
            for key, scenarios in profile_pass_scenario_map.items()
        },
        profile_block_scenarios={
            key: list(scenarios)
            for key, scenarios in profile_block_scenario_map.items()
        },
        profile_violation_scenarios={
            key: list(scenarios)
            for key, scenarios in profile_violation_scenario_map.items()
        },
    )


__all__ = ["evaluate_backtest_guardrails", "GuardrailSummary", "summarize_guardrail_results"]
