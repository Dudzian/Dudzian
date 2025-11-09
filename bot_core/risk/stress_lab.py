"""Stage6 Stress Lab — scalony moduł (HEAD + main).

Zawiera DWIE warstwy:
1) Evaluator (HEAD): z wnioskami/scenariuszami i adaptacyjnymi override'ami dla PortfolioGovernora
   - StressLabPolicyConfig, StressLabSeverityPolicy
   - StressOverrideRecommendation, StressScenarioInsight
   - StressLabReportHead (alias: StressLabReport) + writer-y (JSON/CSV/podpis)
   - StressLabEvaluator

2) Symulator (main): wykonuje scenariusze na bazie baseline'ów rynku (Market Intelligence)
   - MarketBaseline, MarketStressMetrics
   - StressScenarioResult
   - StressLabReportMain (alias: StressLabReportV2)
   - StressLab (runner)
"""

from __future__ import annotations

# ------------------------------ imports wspólne ------------------------------
import csv
import json
import logging
import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.config.models import (
    PortfolioAssetConfig,
    StressLabConfig,            # to jest konfiguracja z config.models (main)
    StressLabDatasetConfig,
    StressLabScenarioConfig,
    StressLabShockConfig,
    StressLabThresholdsConfig,
)
from bot_core.risk._time import now_utc
from bot_core.risk.simulation import (
    ProfileSimulationResult,
    RiskSimulationReport,
    StressTestResult,
)
from bot_core.market_intel.models import MarketIntelBaseline
from bot_core.security.signing import HmacSignedReportMixin, build_hmac_signature


# =============================================================================
#                           CZĘŚĆ 1 — EVALUATOR (HEAD)
# =============================================================================

_REPORT_SCHEMA = "stage6.risk.stress_lab.report"
_REPORT_SCHEMA_VERSION = 1
_SIGNATURE_SCHEMA = "stage6.risk.stress_lab.report.signature"
_SEVERITY_ORDER = {"critical": 3, "warning": 2, "notice": 1, "info": 0}


_DEFAULT_RUNTIME_SCENARIOS: tuple[StressLabScenarioConfig, ...] = (
    StressLabScenarioConfig(
        name="flash_crash_core",
        severity="high",
        markets=("btc_usdt", "eth_usdt"),
        shocks=(
            StressLabShockConfig(type="liquidity_crunch", intensity=0.85),
            StressLabShockConfig(type="volatility_spike", intensity=1.2),
            StressLabShockConfig(type="price_gap", intensity=1.0),
            StressLabShockConfig(type="latency_spike", intensity=0.6),
        ),
        description="Nagła przecena na głównych parach z jednoczesnym pogorszeniem płynności i infrastruktury giełdowej.",
    ),
    StressLabScenarioConfig(
        name="liquidity_drain_altcoins",
        severity="medium",
        markets=("ada_usdt", "sol_usdt", "dot_usdt"),
        shocks=(
            StressLabShockConfig(type="liquidity", intensity=0.65),
            StressLabShockConfig(type="volume_surge", intensity=0.5),
            StressLabShockConfig(type="dispersion", intensity=0.7),
        ),
        description="Ucieczka płynności z alternatywnych rynków i zwiększenie zmienności względnej.",
    ),
    StressLabScenarioConfig(
        name="infra_blackout_and_funding",
        severity="critical",
        markets=("btc_usdt",),
        shocks=(
            StressLabShockConfig(type="infrastructure_blackout", intensity=1.0, duration_minutes=90.0),
            StressLabShockConfig(type="funding_shock", intensity=1.1),
            StressLabShockConfig(type="sentiment_crash", intensity=0.8),
        ),
        description="Symulacja awarii infrastruktury połączonej z gwałtownymi zmianami stawek fundingowych.",
    ),
)


def build_default_runtime_scenarios() -> tuple[StressLabScenarioConfig, ...]:
    """Zwraca zestaw rekomendowanych scenariuszy Stress Lab dla runtime."""

    return _DEFAULT_RUNTIME_SCENARIOS


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
class StressLabPolicyConfig:
    """Polityka oceny Stress Lab (HEAD).

    Uwaga: to NIE jest `config.models.StressLabConfig` z main.
    Ten typ konfiguruje sam evaluator (progi drawdown/liquidity/latency).
    """
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
    """Wniosek z pojedynczego scenariusza Stress Lab (HEAD)."""
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
class StressLabReportHead:
    """Zbiorczy raport Stress Lab (HEAD) obejmujący wnioski i overridy."""
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
    """Przetwarza raporty symulacji (RiskSimulationReport) i buduje overridy (HEAD)."""

    def __init__(
        self,
        config: StressLabPolicyConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config or StressLabPolicyConfig()
        self._clock = clock or now_utc

    def evaluate(
        self,
        risk_report: RiskSimulationReport,
        *,
        portfolio: Mapping[str, PortfolioAssetConfig] | Sequence[PortfolioAssetConfig] | None = None,
    ) -> StressLabReportHead:
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

        return StressLabReportHead(
            generated_at=self._clock(),
            source_report_at=risk_report.generated_at,
            insights=tuple(insights),
            overrides=tuple(overrides),
            counts=dict(counts),
        )

    # --- pomocnicze (HEAD) -------------------------------------------------

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


# --- writer-y (HEAD) ---------------------------------------------------------

def write_report_json(report: StressLabReportHead, output_path: Path) -> dict[str, object]:
    payload = report.to_payload()
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_report_csv(report: StressLabReportHead, output_path: Path) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile", "scenario", "severity", "message", "targets", "tags"])
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


def write_overrides_csv(report: StressLabReportHead, output_path: Path) -> Path:
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
        "signed_at": now_utc().isoformat(),
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


# =============================================================================
#                      CZĘŚĆ 2 — SYMULATOR / RUNNER (main)
# =============================================================================

_LOGGER = logging.getLogger(__name__)


def _is_synthetic_override_enabled() -> bool:
    """Sprawdź, czy wymuszono syntetyczne baseline'y poprzez zmienną środowiskową."""

    flag = os.getenv("STRESS_LAB_ALLOW_SYNTHETIC_FALLBACK")
    if not flag:
        return False
    return flag.strip().lower() in {"1", "true", "yes", "y", "on"}

_SEVERITY_FACTORS: Mapping[str, float] = {
    "low": 0.4,
    "medium": 0.7,
    "high": 1.0,
    "extreme": 1.25,
}


# Bazowy zestaw metryk rynku współdzielony z agregatorem Market Intelligence.
# Alias zachowuje dotychczasową nazwę eksportowaną przez moduł Stress Lab.
MarketBaseline = MarketIntelBaseline


@dataclass(slots=True)
class MarketStressMetrics:
    """Metryki wynikowe pojedynczego rynku po zasymulowaniu szoków (main)."""
    symbol: str
    baseline: MarketBaseline
    liquidity_loss_pct: float
    spread_increase_bps: float
    volatility_increase_pct: float
    sentiment_drawdown: float
    funding_shift_bps: float
    latency_spike_ms: float
    blackout_minutes: float
    dispersion_bps: float
    notes: Sequence[str] = field(default_factory=tuple)

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "symbol": self.symbol,
            "baseline": dict(self.baseline.to_mapping()),
            "liquidity_loss_pct": self.liquidity_loss_pct,
            "spread_increase_bps": self.spread_increase_bps,
            "volatility_increase_pct": self.volatility_increase_pct,
            "sentiment_drawdown": self.sentiment_drawdown,
            "funding_shift_bps": self.funding_shift_bps,
            "latency_spike_ms": self.latency_spike_ms,
            "blackout_minutes": self.blackout_minutes,
            "dispersion_bps": self.dispersion_bps,
        }
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass(slots=True)
class StressScenarioResult:
    """Zbiorczy wynik scenariusza Stress Lab (main)."""
    name: str
    severity: str
    status: str
    metrics: Mapping[str, float]
    markets: Sequence[MarketStressMetrics]
    failures: Sequence[str] = field(default_factory=tuple)
    description: str | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "name": self.name,
            "severity": self.severity,
            "status": self.status,
            "metrics": dict(self.metrics),
            "markets": [market.to_mapping() for market in self.markets],
        }
        if self.failures:
            payload["failures"] = list(self.failures)
        if self.description:
            payload["description"] = self.description
        return payload

    def has_failures(self) -> bool:
        return self.status.lower() not in {"passed", "ok", "success"}


@dataclass(slots=True)
class StressLabReportMain(HmacSignedReportMixin):
    """Raport zbiorczy Stress Lab (main)."""
    generated_at: str
    thresholds: StressLabThresholdsConfig
    scenarios: Sequence[StressScenarioResult]

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at,
            "thresholds": asdict(self.thresholds),
            "scenarios": [scenario.to_mapping() for scenario in self.scenarios],
            "failure_count": sum(1 for scenario in self.scenarios if scenario.has_failures()),
        }

    def has_failures(self) -> bool:
        return any(scenario.has_failures() for scenario in self.scenarios)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path

class StressLab:
    """Wykonuje scenariusze Stress Lab na podstawie konfiguracji Stage6 (main)."""

    def __init__(self, config: StressLabConfig) -> None:
        self._config = config
        self._datasets: dict[str, MarketBaseline] = {}
        self._datasets_by_symbol: dict[str, MarketBaseline] = {}
        for name, dataset in config.datasets.items():
            baseline = self._load_dataset(name, dataset)
            self._datasets[name] = baseline
            self._datasets_by_symbol[baseline.symbol] = baseline

    def run(self) -> StressLabReportMain:
        generated_at = datetime.now(timezone.utc).isoformat()
        scenario_results: list[StressScenarioResult] = []
        for scenario in self._config.scenarios:
            scenario_results.append(self._run_scenario(scenario))
        return StressLabReportMain(
            generated_at=generated_at,
            thresholds=self._config.thresholds,
            scenarios=tuple(scenario_results),
        )

    def _run_scenario(self, scenario: StressLabScenarioConfig) -> StressScenarioResult:
        severity = scenario.severity.lower()
        severity_factor = _SEVERITY_FACTORS.get(severity, _SEVERITY_FACTORS["medium"])
        market_results: list[MarketStressMetrics] = []
        for market in scenario.markets:
            baseline = self._resolve_baseline(market)
            market_results.append(
                self._apply_shocks(
                    baseline=baseline,
                    shocks=scenario.shocks,
                    severity_factor=severity_factor,
                )
            )

        aggregated = self._aggregate_metrics(market_results)
        thresholds = scenario.threshold_overrides or self._config.thresholds
        failures: list[str] = []

        if aggregated["liquidity_loss_pct"] > thresholds.max_liquidity_loss_pct:
            failures.append(
                f"liquidity_loss_pct={aggregated['liquidity_loss_pct']:.3f}>{thresholds.max_liquidity_loss_pct:.3f}"
            )
        if aggregated["spread_increase_bps"] > thresholds.max_spread_increase_bps:
            failures.append(
                f"spread_increase_bps={aggregated['spread_increase_bps']:.2f}>{thresholds.max_spread_increase_bps:.2f}"
            )
        if aggregated["volatility_increase_pct"] > thresholds.max_volatility_increase_pct:
            failures.append(
                f"volatility_increase_pct={aggregated['volatility_increase_pct']:.3f}>{thresholds.max_volatility_increase_pct:.3f}"
            )
        if aggregated["sentiment_drawdown"] > thresholds.max_sentiment_drawdown:
            failures.append(
                f"sentiment_drawdown={aggregated['sentiment_drawdown']:.3f}>{thresholds.max_sentiment_drawdown:.3f}"
            )
        if aggregated["funding_shift_bps"] > thresholds.max_funding_change_bps:
            failures.append(
                f"funding_shift_bps={aggregated['funding_shift_bps']:.2f}>{thresholds.max_funding_change_bps:.2f}"
            )
        if aggregated["latency_spike_ms"] > thresholds.max_latency_spike_ms:
            failures.append(
                f"latency_spike_ms={aggregated['latency_spike_ms']:.2f}>{thresholds.max_latency_spike_ms:.2f}"
            )
        if aggregated["blackout_minutes"] > thresholds.max_blackout_minutes:
            failures.append(
                f"blackout_minutes={aggregated['blackout_minutes']:.1f}>{thresholds.max_blackout_minutes:.1f}"
            )
        if aggregated["dispersion_bps"] > thresholds.max_dispersion_bps:
            failures.append(
                f"dispersion_bps={aggregated['dispersion_bps']:.2f}>{thresholds.max_dispersion_bps:.2f}"
            )

        status = "failed" if failures else "passed"
        metrics_payload = {key: float(value) for key, value in aggregated.items()}
        return StressScenarioResult(
            name=scenario.name,
            severity=severity,
            status=status,
            metrics=metrics_payload,
            markets=tuple(market_results),
            failures=tuple(failures),
            description=scenario.description,
        )

    def _aggregate_metrics(self, market_results: Sequence[MarketStressMetrics]) -> Mapping[str, float]:
        if not market_results:
            return {
                "liquidity_loss_pct": 0.0,
                "spread_increase_bps": 0.0,
                "volatility_increase_pct": 0.0,
                "sentiment_drawdown": 0.0,
                "funding_shift_bps": 0.0,
                "latency_spike_ms": 0.0,
                "blackout_minutes": 0.0,
                "dispersion_bps": 0.0,
            }
        weights = [max(result.baseline.weight, 0.0) for result in market_results]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            weights = [1.0 for _ in market_results]
            total_weight = float(len(market_results))

        def _weighted_average(attr: str) -> float:
            return sum(
                getattr(result, attr) * weight
                for result, weight in zip(market_results, weights)
            ) / total_weight

        return {
            "liquidity_loss_pct": max(0.0, _weighted_average("liquidity_loss_pct")),
            "spread_increase_bps": max(0.0, _weighted_average("spread_increase_bps")),
            "volatility_increase_pct": max(0.0, _weighted_average("volatility_increase_pct")),
            "sentiment_drawdown": max(0.0, _weighted_average("sentiment_drawdown")),
            "funding_shift_bps": max(0.0, _weighted_average("funding_shift_bps")),
            "latency_spike_ms": max(result.latency_spike_ms for result in market_results),
            "blackout_minutes": max(result.blackout_minutes for result in market_results),
            "dispersion_bps": max(result.dispersion_bps for result in market_results),
        }

    def _apply_shocks(
        self,
        *,
        baseline: MarketBaseline,
        shocks: Sequence[StressLabShockConfig],
        severity_factor: float,
    ) -> MarketStressMetrics:
        depth = max(baseline.avg_depth_usd, 1.0)
        spread = max(baseline.avg_spread_bps, 0.01)
        volatility = max(baseline.realized_volatility, 0.001)
        sentiment = baseline.sentiment_score
        funding = baseline.funding_rate_bps
        latency_spike_ms = 0.0
        blackout_minutes = 0.0
        dispersion_bps = 0.0
        notes: list[str] = []

        for shock in shocks:
            intensity = max(0.0, float(shock.intensity))
            shock_type = shock.type.lower()
            if shock_type in {"liquidity", "liquidity_crunch"}:
                loss = min(0.99, severity_factor * intensity)
                depth *= 1.0 - loss
                spread += baseline.avg_spread_bps * (0.4 + 0.6 * loss)
                notes.append(f"liquidity_loss={loss:.3f}")
            elif shock_type in {"volatility", "volatility_spike"}:
                multiplier = 1.0 + severity_factor * intensity
                volatility *= multiplier
                spread += baseline.avg_spread_bps * 0.2 * intensity * severity_factor
            elif shock_type in {"sentiment", "sentiment_crash"}:
                sentiment -= min(1.5, severity_factor * intensity)
            elif shock_type in {"funding", "funding_shock"}:
                funding += severity_factor * intensity * 25.0
            elif shock_type in {"latency", "latency_spike", "infrastructure"}:
                latency_spike_ms = max(
                    latency_spike_ms,
                    25.0 + severity_factor * intensity * 220.0,
                )
            elif shock_type in {"blackout", "exchange_outage", "infrastructure_blackout"}:
                duration = shock.duration_minutes if shock.duration_minutes is not None else severity_factor * intensity * 90.0
                blackout_minutes = max(blackout_minutes, max(0.0, duration))
            elif shock_type in {"price_gap", "divergence", "dispersion"}:
                dispersion_bps = max(dispersion_bps, severity_factor * intensity * 80.0)
            elif shock_type in {"volume_surge", "volume"}:
                depth *= 1.0 - min(0.5, 0.2 * intensity * severity_factor)
                volatility *= 1.0 + 0.12 * intensity * severity_factor
            else:
                notes.append(f"unknown_shock={shock.type}")

        liquidity_loss_pct = min(1.0, max(0.0, 1.0 - (depth / max(baseline.avg_depth_usd, 1.0))))
        spread_increase_bps = max(0.0, spread - baseline.avg_spread_bps)
        volatility_increase_pct = max(
            0.0, (volatility - baseline.realized_volatility) / max(baseline.realized_volatility, 1e-6)
        )
        sentiment_drawdown = max(0.0, baseline.sentiment_score - sentiment)
        funding_shift_bps = abs(funding - baseline.funding_rate_bps)

        return MarketStressMetrics(
            symbol=baseline.symbol,
            baseline=baseline,
            liquidity_loss_pct=liquidity_loss_pct,
            spread_increase_bps=spread_increase_bps,
            volatility_increase_pct=volatility_increase_pct,
            sentiment_drawdown=sentiment_drawdown,
            funding_shift_bps=funding_shift_bps,
            latency_spike_ms=latency_spike_ms,
            blackout_minutes=blackout_minutes,
            dispersion_bps=dispersion_bps,
            notes=tuple(notes),
        )

    def _resolve_baseline(self, market: str) -> MarketBaseline:
        if market in self._datasets:
            return self._datasets[market]
        if market in self._datasets_by_symbol:
            return self._datasets_by_symbol[market]
        _LOGGER.warning("Stress Lab: brak datasetu dla rynku %s – generuję syntetyczny baseline", market)
        baseline = self._build_synthetic_baseline(market, weight=1.0)
        self._datasets_by_symbol[baseline.symbol] = baseline
        return baseline

    def _load_dataset(self, name: str, dataset: StressLabDatasetConfig) -> MarketBaseline:
        metrics_path = Path(dataset.metrics_path)
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Nie udało się odczytać pliku metryk Stress Lab {metrics_path}: {exc}"
                ) from exc
        else:
            override_enabled = _is_synthetic_override_enabled()
            if not dataset.allow_synthetic and not override_enabled:
                _LOGGER.error(
                    "Stress Lab: brak metryk Stage6 dla rynku %s – oczekiwano pliku %s",
                    dataset.symbol,
                    metrics_path,
                )
                raise FileNotFoundError(
                    f"Dataset Stress Lab {metrics_path} nie istnieje, a allow_synthetic jest False"
                )
            if dataset.allow_synthetic:
                _LOGGER.warning(
                    "Stress Lab: używam syntetycznych danych dla rynku %s (plik %s)",
                    dataset.symbol,
                    metrics_path,
                )
            else:
                _LOGGER.warning(
                    "Stress Lab: brak metryk Stage6 dla rynku %s – STRESS_LAB_ALLOW_SYNTHETIC_FALLBACK"
                    " aktywny, generuję syntetyczne baseline",
                    dataset.symbol,
                )
            data = self._build_synthetic_payload(dataset.symbol)

        baseline_payload = data.get("baseline", data)
        return MarketBaseline(
            symbol=str(baseline_payload.get("symbol", dataset.symbol)),
            mid_price=float(baseline_payload.get("mid_price", 25_000.0)),
            avg_depth_usd=float(baseline_payload.get("avg_depth_usd", 1_500_000.0)),
            avg_spread_bps=float(baseline_payload.get("avg_spread_bps", 6.0)),
            funding_rate_bps=float(baseline_payload.get("funding_rate_bps", 12.0)),
            sentiment_score=float(baseline_payload.get("sentiment_score", 0.4)),
            realized_volatility=float(baseline_payload.get("realized_volatility", 0.35)),
            weight=max(0.0, dataset.weight),
        )

    def _build_synthetic_payload(self, symbol: str) -> Mapping[str, object]:
        baseline = self._build_synthetic_baseline(symbol, weight=1.0)
        payload = baseline.to_mapping()
        payload["baseline"] = dict(payload)
        return payload

    def _build_synthetic_baseline(self, symbol: str, *, weight: float) -> MarketBaseline:
        digest = hashlib.sha256(symbol.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        mid_price = 10_000.0 + (seed % 50_000)
        depth = 1_000_000.0 + float(seed % 750_000)
        spread = 4.0 + (seed % 120) / 4.0
        funding = ((seed % 400) - 200) / 10.0
        sentiment = max(-1.0, min(1.0, ((seed % 200) - 100) / 90.0))
        volatility = 0.25 + ((seed % 120) / 200.0)
        return MarketBaseline(
            symbol=symbol,
            mid_price=mid_price,
            avg_depth_usd=depth,
            avg_spread_bps=spread,
            funding_rate_bps=funding,
            sentiment_score=sentiment,
            realized_volatility=volatility,
            weight=max(weight, 0.0),
        )


# =============================================================================
#                                 EXPORTY
# =============================================================================

# Alias zgodności: w większości kodu HEAD nazwała się po prostu StressLabReport.
StressLabReport = StressLabReportHead
# A raport z main udostępniamy zarówno jako StressLabReportMain, jak i StressLabReportV2.
StressLabReportV2 = StressLabReportMain

__all__ = [
    # Evaluator (HEAD)
    "StressLabPolicyConfig",
    "StressLabSeverityPolicy",
    "StressOverrideRecommendation",
    "StressScenarioInsight",
    "StressLabReportHead",
    "StressLabReport",          # alias -> Head
    "StressLabEvaluator",
    "write_overrides_csv",
    "write_report_csv",
    "write_report_json",
    "write_report_signature",
    # Runner (main)
    "MarketBaseline",
    "MarketStressMetrics",
    "StressScenarioResult",
    "StressLabReportMain",
    "StressLabReportV2",
    "StressLab",
    "build_default_runtime_scenarios",
]
