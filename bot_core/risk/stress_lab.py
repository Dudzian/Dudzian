"""Stage6 Stress Lab – ewaluacja symulacji i adaptacyjne overridy portfela."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.config.models import PortfolioAssetConfig
from bot_core.risk.simulation import ProfileSimulationResult, RiskSimulationReport, StressTestResult
from bot_core.security.signing import build_hmac_signature


_REPORT_SCHEMA = "stage6.risk.stress_lab.report"
_REPORT_SCHEMA_VERSION = 1
_SIGNATURE_SCHEMA = "stage6.risk.stress_lab.report.signature"
_SEVERITY_ORDER = {"critical": 3, "warning": 2, "notice": 1, "info": 0}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class StressLabSeverityPolicy:
    """Polityka adaptacyjna dla określonego poziomu severity."""

    severity: str
    weight_multiplier: float | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    force_rebalance: bool = False


def _default_policies() -> Mapping[str, StressLabSeverityPolicy]:
    return {
        "warning": StressLabSeverityPolicy(
            severity="warning",
            weight_multiplier=0.6,
            min_weight=0.0,
        ),
        "critical": StressLabSeverityPolicy(
            severity="critical",
            weight_multiplier=0.0,
            min_weight=0.0,
            max_weight=0.0,
            force_rebalance=True,
        ),
        "notice": StressLabSeverityPolicy(severity="notice"),
        "info": StressLabSeverityPolicy(severity="info"),
    }


@dataclass(slots=True)
class StressLabConfig:
    """Ustawienia oceny Stress Lab Stage6."""

    drawdown_warning_threshold: float = 0.08
    drawdown_critical_threshold: float = 0.12
    liquidity_warning_threshold_usd: float | None = 200_000.0
    liquidity_critical_threshold_usd: float | None = 100_000.0
    latency_warning_threshold_ms: float | None = 250.0
    latency_critical_threshold_ms: float | None = 500.0
    failure_default_severity: str = "critical"
    degradation_default_severity: str = "warning"
    default_tags: Sequence[str] = field(default_factory=lambda: ("stress_lab",))
    severity_policies: Mapping[str, StressLabSeverityPolicy] = field(default_factory=_default_policies)


@dataclass(slots=True)
class StressOverrideRecommendation:
    """Rekomendacja override'u dla PortfolioGovernora."""

    severity: str
    reason: str
    symbol: str | None = None
    risk_budget: str | None = None
    weight_multiplier: float | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    force_rebalance: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "severity": self.severity,
            "reason": self.reason,
            "tags": list(self.tags),
        }
        if self.symbol is not None:
            payload["symbol"] = self.symbol
        if self.risk_budget is not None:
            payload["risk_budget"] = self.risk_budget
        if self.weight_multiplier is not None:
            payload["weight_multiplier"] = self.weight_multiplier
        if self.min_weight is not None:
            payload["min_weight"] = self.min_weight
        if self.max_weight is not None:
            payload["max_weight"] = self.max_weight
        if self.force_rebalance:
            payload["force_rebalance"] = True
        return payload


@dataclass(slots=True)
class StressScenarioInsight:
    """Wniosek z pojedynczego scenariusza Stress Lab."""

    profile: str
    scenario: str
    severity: str
    message: str
    metrics: Mapping[str, object]
    targets: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "profile": self.profile,
            "scenario": self.scenario,
            "severity": self.severity,
            "message": self.message,
            "metrics": dict(self.metrics),
        }
        if self.targets:
            payload["targets"] = list(self.targets)
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(slots=True)
class StressLabReport:
    """Zbiorczy raport Stress Lab obejmujący wszystkie scenariusze."""

    generated_at: datetime
    source_report_at: str
    insights: Sequence[StressScenarioInsight]
    overrides: Sequence[StressOverrideRecommendation]
    counts: Mapping[str, int]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": _REPORT_SCHEMA,
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "source_report_at": self.source_report_at,
            "counts": dict(self.counts),
            "overrides_total": len(self.overrides),
            "insights": [insight.to_dict() for insight in self.insights],
            "overrides": [override.to_dict() for override in self.overrides],
        }


class StressLabEvaluator:
    """Przetwarza raporty symulacji i buduje adaptacyjne rekomendacje."""

    def __init__(
        self,
        config: StressLabConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config or StressLabConfig()
        self._clock = clock or _now_utc

    def evaluate(
        self,
        risk_report: RiskSimulationReport,
        *,
        portfolio: Mapping[str, PortfolioAssetConfig] | Sequence[PortfolioAssetConfig] | None = None,
    ) -> StressLabReport:
        if portfolio is None:
            asset_map: Mapping[str, PortfolioAssetConfig] = {}
        elif isinstance(portfolio, Mapping):
            asset_map = portfolio
        else:
            asset_map = {asset.symbol: asset for asset in portfolio}

        insights: list[StressScenarioInsight] = []
        overrides: list[StressOverrideRecommendation] = []
        counts: MutableMapping[str, int] = {"total": 0}

        for profile in risk_report.profiles:
            drawdown_insight = self._evaluate_drawdown(profile)
            if drawdown_insight is not None:
                counts["total"] = counts.get("total", 0) + 1
                counts[drawdown_insight.severity] = counts.get(drawdown_insight.severity, 0) + 1
                insights.append(drawdown_insight)

            for stress in profile.stress_tests:
                insight, new_overrides = self._evaluate_stress(profile, stress, asset_map)
                if insight is None:
                    continue
                counts["total"] = counts.get("total", 0) + 1
                counts[insight.severity] = counts.get(insight.severity, 0) + 1
                insights.append(insight)
                overrides.extend(new_overrides)

        return StressLabReport(
            generated_at=self._clock(),
            source_report_at=risk_report.generated_at,
            insights=tuple(insights),
            overrides=tuple(overrides),
            counts=dict(counts),
        )

    def _evaluate_drawdown(self, profile: ProfileSimulationResult) -> StressScenarioInsight | None:
        drawdown = float(profile.max_drawdown_pct)
        if drawdown >= self._config.drawdown_critical_threshold:
            severity = "critical"
            message = (
                f"Max drawdown {drawdown:.2%} przekracza próg {self._config.drawdown_critical_threshold:.2%}"
            )
        elif drawdown >= self._config.drawdown_warning_threshold:
            severity = "warning"
            message = (
                f"Max drawdown {drawdown:.2%} przekracza próg {self._config.drawdown_warning_threshold:.2%}"
            )
        else:
            return None

        tags = tuple(dict.fromkeys((*self._config.default_tags, f"profile:{profile.profile}")))
        metrics = {
            "max_drawdown_pct": drawdown,
            "total_return_pct": float(profile.total_return_pct),
            "worst_daily_loss_pct": float(profile.worst_daily_loss_pct),
        }
        return StressScenarioInsight(
            profile=profile.profile,
            scenario="max_drawdown",
            severity=severity,
            message=message,
            metrics=metrics,
            tags=tags,
        )

    def _evaluate_stress(
        self,
        profile: ProfileSimulationResult,
        stress: StressTestResult,
        asset_map: Mapping[str, PortfolioAssetConfig],
    ) -> tuple[StressScenarioInsight | None, Sequence[StressOverrideRecommendation]]:
        metrics: MutableMapping[str, object] = dict(self._normalize_metrics(stress.metrics))
        severity = self._resolve_severity(stress, metrics)
        derived_severity, derived_messages, derived_tags = self._derive_metric_alerts(metrics)
        severity = self._merge_severity(severity, derived_severity)
        if severity is None:
            return None, ()

        base_reason = self._resolve_reason(stress)
        message_parts = [base_reason]
        if derived_messages:
            message_parts.extend(derived_messages)
        message = "; ".join(part for part in message_parts if part)
        reason = "; ".join(
            part
            for part in (
                base_reason,
                derived_messages[0] if derived_messages else None,
            )
            if part
        )

        tags = self._resolve_tags(metrics)
        tags = tuple(
            dict.fromkeys(
                (
                    *self._config.default_tags,
                    *tags,
                    *derived_tags,
                    f"profile:{profile.profile}",
                )
            )
        )
        targets = self._resolve_targets(metrics, asset_map)

        insight = StressScenarioInsight(
            profile=profile.profile,
            scenario=stress.name,
            severity=severity,
            message=message,
            metrics=metrics,
            targets=targets,
            tags=tags,
        )

        policy = self._config.severity_policies.get(severity)
        if policy is None:
            return insight, ()

        overrides: list[StressOverrideRecommendation] = []
        risk_budget = str(metrics.get("risk_budget")) if metrics.get("risk_budget") else None

        if targets:
            for symbol in targets:
                overrides.append(
                    StressOverrideRecommendation(
                        severity=severity,
                        reason=f"{stress.name}: {reason}",
                        symbol=symbol,
                        risk_budget=risk_budget,
                        weight_multiplier=policy.weight_multiplier,
                        min_weight=policy.min_weight,
                        max_weight=policy.max_weight,
                        tags=tags,
                        force_rebalance=policy.force_rebalance,
                    )
                )
        else:
            overrides.append(
                StressOverrideRecommendation(
                    severity=severity,
                    reason=f"{stress.name}: {reason}",
                    risk_budget=risk_budget,
                    weight_multiplier=policy.weight_multiplier,
                    min_weight=policy.min_weight,
                    max_weight=policy.max_weight,
                    tags=tags,
                    force_rebalance=policy.force_rebalance,
                )
            )

        return insight, overrides

    def _resolve_severity(
        self,
        stress: StressTestResult,
        metrics: Mapping[str, object],
    ) -> str | None:
        metrics_severity = metrics.get("severity") if isinstance(metrics, Mapping) else None
        if isinstance(metrics_severity, str) and metrics_severity.strip():
            return metrics_severity.strip().lower()

        normalized_status = stress.status.strip().lower()
        if normalized_status in {"failed", "error", "breach"}:
            return self._config.failure_default_severity
        if normalized_status in {"warning", "degraded"}:
            return self._config.degradation_default_severity
        if normalized_status in {"ok", "passed", "success"}:
            return "info"
        return None

    @staticmethod
    def _resolve_reason(stress: StressTestResult) -> str:
        note = stress.notes.strip() if isinstance(stress.notes, str) else ""
        if note:
            return note
        return f"Status {stress.status}".strip()

    @staticmethod
    def _normalize_metrics(metrics: Mapping[str, object]) -> Mapping[str, object]:
        normalized: dict[str, object] = {}
        for key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[str(key)] = value
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                normalized[str(key)] = [str(item) for item in value]
            else:
                normalized[str(key)] = str(value)
        return normalized

    def _merge_severity(self, base: str | None, candidate: str | None) -> str | None:
        if base is None:
            return candidate
        if candidate is None:
            return base
        base_rank = _SEVERITY_ORDER.get(base, -1)
        candidate_rank = _SEVERITY_ORDER.get(candidate, -1)
        return base if base_rank >= candidate_rank else candidate

    def _derive_metric_alerts(
        self,
        metrics: MutableMapping[str, object],
    ) -> tuple[str | None, list[str], list[str]]:
        severity: str | None = None
        messages: list[str] = []
        tags: list[str] = []

        liquidity = self._try_parse_float(metrics.get("liquidity_usd"))
        if liquidity is None:
            liquidity = self._try_parse_float(metrics.get("available_liquidity_usd"))
        if liquidity is not None:
            level = self._assess_liquidity(liquidity)
            if level is not None:
                severity = self._merge_severity(severity, level)
                qualifier = "krytycznego" if level == "critical" else "ostrzegawczego"
                threshold = (
                    self._config.liquidity_critical_threshold_usd
                    if level == "critical"
                    else self._config.liquidity_warning_threshold_usd
                )
                messages.append(
                    (
                        f"Płynność {liquidity:,.2f} USD poniżej progu {qualifier} "
                        f"{threshold:,.2f} USD"
                    ).replace(",", " ")
                )
                tags.append("metric:liquidity")
                metrics.setdefault("liquidity_alert", level)
                metrics.setdefault("liquidity_threshold_usd", threshold)

        latency_values = [
            self._try_parse_float(metrics.get(key))
            for key in (
                "avg_order_latency_ms",
                "p95_order_latency_ms",
                "max_order_latency_ms",
                "latency_ms",
            )
        ]
        latency_candidates = [value for value in latency_values if value is not None]
        if latency_candidates:
            worst_latency = max(latency_candidates)
            level = self._assess_latency(worst_latency)
            if level is not None:
                severity = self._merge_severity(severity, level)
                qualifier = "krytycznego" if level == "critical" else "ostrzegawczego"
                threshold = (
                    self._config.latency_critical_threshold_ms
                    if level == "critical"
                    else self._config.latency_warning_threshold_ms
                )
                messages.append(
                    (
                        f"Latencja zleceń {worst_latency:.1f} ms przekracza próg {qualifier} "
                        f"{threshold:.1f} ms"
                    ).replace(",", " ")
                )
                tags.append("metric:latency")
                metrics.setdefault("latency_alert", level)
                metrics.setdefault("latency_threshold_ms", threshold)
                metrics.setdefault("latency_peak_ms", worst_latency)

        return severity, messages, tags

    def _assess_liquidity(self, liquidity: float) -> str | None:
        critical = self._config.liquidity_critical_threshold_usd
        warning = self._config.liquidity_warning_threshold_usd
        if critical is not None and liquidity < critical:
            return "critical"
        if warning is not None and liquidity < warning:
            return "warning"
        return None

    def _assess_latency(self, latency_ms: float) -> str | None:
        critical = self._config.latency_critical_threshold_ms
        warning = self._config.latency_warning_threshold_ms
        if critical is not None and latency_ms > critical:
            return "critical"
        if warning is not None and latency_ms > warning:
            return "warning"
        return None

    @staticmethod
    def _try_parse_float(value: object) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _resolve_tags(metrics: Mapping[str, object]) -> Sequence[str]:
        raw = metrics.get("tags")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return tuple(str(item) for item in raw)
        if isinstance(raw, str) and raw.strip():
            return (raw.strip(),)
        return ()

    @staticmethod
    def _resolve_targets(
        metrics: Mapping[str, object],
        asset_map: Mapping[str, PortfolioAssetConfig],
    ) -> Sequence[str]:
        assets_raw = metrics.get("assets")
        targets: list[str] = []
        if isinstance(assets_raw, Sequence) and not isinstance(assets_raw, (str, bytes)):
            targets.extend(str(item) for item in assets_raw)
        elif isinstance(assets_raw, str) and assets_raw.strip():
            targets.append(assets_raw.strip())

        tags = metrics.get("tags")
        tag_values: tuple[str, ...]
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            tag_values = tuple(str(item) for item in tags)
        elif isinstance(tags, str) and tags.strip():
            tag_values = (tags.strip(),)
        else:
            tag_values = ()

        if tag_values and asset_map:
            for symbol, asset in asset_map.items():
                if symbol in targets:
                    continue
                if any(tag in asset.tags for tag in tag_values):
                    targets.append(symbol)

        if not asset_map:
            return tuple(dict.fromkeys(targets))

        filtered = [symbol for symbol in targets if symbol in asset_map]
        return tuple(dict.fromkeys(filtered))


def write_report_json(report: StressLabReport, output_path: Path) -> dict[str, object]:
    payload = report.to_payload()
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_report_csv(report: StressLabReport, output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "profile",
            "scenario",
            "severity",
            "message",
            "targets",
            "tags",
        ])
        for insight in report.insights:
            writer.writerow(
                [
                    insight.profile,
                    insight.scenario,
                    insight.severity,
                    insight.message,
                    "|".join(insight.targets),
                    "|".join(insight.tags),
                ]
            )
    return output_path


def write_overrides_csv(report: StressLabReport, output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "risk_budget",
                "severity",
                "weight_multiplier",
                "min_weight",
                "max_weight",
                "force_rebalance",
                "reason",
                "tags",
            ]
        )
        for override in report.overrides:
            writer.writerow(
                [
                    override.symbol or "",
                    override.risk_budget or "",
                    override.severity,
                    "" if override.weight_multiplier is None else f"{override.weight_multiplier:.6f}",
                    "" if override.min_weight is None else f"{override.min_weight:.6f}",
                    "" if override.max_weight is None else f"{override.max_weight:.6f}",
                    "yes" if override.force_rebalance else "no",
                    override.reason,
                    "|".join(override.tags),
                ]
            )
    return output_path


def write_report_signature(
    payload: Mapping[str, object],
    output_path: Path,
    *,
    key: bytes,
    key_id: str | None = None,
    target: str | None = None,
) -> Mapping[str, object]:
    document = {
        "schema": _SIGNATURE_SCHEMA,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "signed_at": _now_utc().isoformat(),
        "target": target or output_path.name,
        "signature": build_hmac_signature(
            payload,
            key=key,
            algorithm="HMAC-SHA256",
            key_id=key_id,
        ),
    }
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


__all__ = [
    "StressLabConfig",
    "StressLabEvaluator",
    "StressLabReport",
    "StressLabSeverityPolicy",
    "StressOverrideRecommendation",
    "StressScenarioInsight",
    "write_overrides_csv",
    "write_report_csv",
    "write_report_json",
    "write_report_signature",
]
