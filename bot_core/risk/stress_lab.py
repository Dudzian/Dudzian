"""Symulator Stress Lab dla Etapu 6."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.config.models import (
    StressLabConfig,
    StressLabDatasetConfig,
    StressLabScenarioConfig,
    StressLabShockConfig,
    StressLabThresholdsConfig,
)
from bot_core.security.signing import build_hmac_signature

_LOGGER = logging.getLogger(__name__)

_SEVERITY_FACTORS: Mapping[str, float] = {
    "low": 0.4,
    "medium": 0.7,
    "high": 1.0,
    "extreme": 1.25,
}


@dataclass(slots=True)
class MarketBaseline:
    """Bazowe metryki rynku wykorzystywane w Stress Lab."""

    symbol: str
    mid_price: float
    avg_depth_usd: float
    avg_spread_bps: float
    funding_rate_bps: float
    sentiment_score: float
    realized_volatility: float
    weight: float = 1.0

    def to_mapping(self) -> Mapping[str, float | str]:
        return {
            "symbol": self.symbol,
            "mid_price": self.mid_price,
            "avg_depth_usd": self.avg_depth_usd,
            "avg_spread_bps": self.avg_spread_bps,
            "funding_rate_bps": self.funding_rate_bps,
            "sentiment_score": self.sentiment_score,
            "realized_volatility": self.realized_volatility,
            "weight": self.weight,
        }


@dataclass(slots=True)
class MarketStressMetrics:
    """Metryki wynikowe pojedynczego rynku po zasymulowaniu szoków."""

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
    """Zbiorczy wynik scenariusza Stress Lab."""

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
class StressLabReport:
    """Raport zbiorczy Stress Lab."""

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

    def build_signature(
        self,
        *,
        key: bytes,
        algorithm: str = "HMAC-SHA256",
        key_id: str | None = None,
    ) -> Mapping[str, str]:
        return build_hmac_signature(self.to_mapping(), key=key, algorithm=algorithm, key_id=key_id)

    def write_signature(
        self,
        path: Path,
        *,
        key: bytes,
        algorithm: str = "HMAC-SHA256",
        key_id: str | None = None,
    ) -> Path:
        signature = self.build_signature(key=key, algorithm=algorithm, key_id=key_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(signature, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path


class StressLab:
    """Wykonuje scenariusze Stress Lab na podstawie konfiguracji Stage6."""

    def __init__(self, config: StressLabConfig) -> None:
        self._config = config
        self._datasets: dict[str, MarketBaseline] = {}
        self._datasets_by_symbol: dict[str, MarketBaseline] = {}
        for name, dataset in config.datasets.items():
            baseline = self._load_dataset(name, dataset)
            self._datasets[name] = baseline
            self._datasets_by_symbol[baseline.symbol] = baseline

    def run(self) -> StressLabReport:
        generated_at = datetime.now(timezone.utc).isoformat()
        scenario_results: list[StressScenarioResult] = []
        for scenario in self._config.scenarios:
            scenario_results.append(self._run_scenario(scenario))
        return StressLabReport(
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
                f"liquidity_loss_pct={aggregated['liquidity_loss_pct']:.3f}>"
                f"{thresholds.max_liquidity_loss_pct:.3f}"
            )
        if aggregated["spread_increase_bps"] > thresholds.max_spread_increase_bps:
            failures.append(
                f"spread_increase_bps={aggregated['spread_increase_bps']:.2f}>"
                f"{thresholds.max_spread_increase_bps:.2f}"
            )
        if aggregated["volatility_increase_pct"] > thresholds.max_volatility_increase_pct:
            failures.append(
                f"volatility_increase_pct={aggregated['volatility_increase_pct']:.3f}>"
                f"{thresholds.max_volatility_increase_pct:.3f}"
            )
        if aggregated["sentiment_drawdown"] > thresholds.max_sentiment_drawdown:
            failures.append(
                f"sentiment_drawdown={aggregated['sentiment_drawdown']:.3f}>"
                f"{thresholds.max_sentiment_drawdown:.3f}"
            )
        if aggregated["funding_shift_bps"] > thresholds.max_funding_change_bps:
            failures.append(
                f"funding_shift_bps={aggregated['funding_shift_bps']:.2f}>"
                f"{thresholds.max_funding_change_bps:.2f}"
            )
        if aggregated["latency_spike_ms"] > thresholds.max_latency_spike_ms:
            failures.append(
                f"latency_spike_ms={aggregated['latency_spike_ms']:.2f}>"
                f"{thresholds.max_latency_spike_ms:.2f}"
            )
        if aggregated["blackout_minutes"] > thresholds.max_blackout_minutes:
            failures.append(
                f"blackout_minutes={aggregated['blackout_minutes']:.1f}>"
                f"{thresholds.max_blackout_minutes:.1f}"
            )
        if aggregated["dispersion_bps"] > thresholds.max_dispersion_bps:
            failures.append(
                f"dispersion_bps={aggregated['dispersion_bps']:.2f}>"
                f"{thresholds.max_dispersion_bps:.2f}"
            )

        status = "failed" if failures else "passed"
        metrics_payload = {
            key: float(value)
            for key, value in aggregated.items()
        }
        return StressScenarioResult(
            name=scenario.name,
            severity=severity,
            status=status,
            metrics=metrics_payload,
            markets=tuple(market_results),
            failures=tuple(failures),
            description=scenario.description,
        )

    def _aggregate_metrics(
        self, market_results: Sequence[MarketStressMetrics]
    ) -> Mapping[str, float]:
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
                duration = (
                    shock.duration_minutes
                    if shock.duration_minutes is not None
                    else severity_factor * intensity * 90.0
                )
                blackout_minutes = max(blackout_minutes, max(0.0, duration))
            elif shock_type in {"price_gap", "divergence", "dispersion"}:
                dispersion_bps = max(
                    dispersion_bps,
                    severity_factor * intensity * 80.0,
                )
            elif shock_type in {"volume_surge", "volume"}:
                depth *= 1.0 - min(0.5, 0.2 * intensity * severity_factor)
                volatility *= 1.0 + 0.12 * intensity * severity_factor
            else:
                notes.append(f"unknown_shock={shock.type}")

        liquidity_loss_pct = min(1.0, max(0.0, 1.0 - (depth / max(baseline.avg_depth_usd, 1.0))))
        spread_increase_bps = max(0.0, spread - baseline.avg_spread_bps)
        volatility_increase_pct = max(
            0.0,
            (volatility - baseline.realized_volatility)
            / max(baseline.realized_volatility, 1e-6),
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
        _LOGGER.warning(
            "Stress Lab: brak datasetu dla rynku %s – generuję syntetyczny baseline", market
        )
        baseline = self._build_synthetic_baseline(market, weight=1.0)
        self._datasets_by_symbol[baseline.symbol] = baseline
        return baseline

    def _load_dataset(
        self, name: str, dataset: StressLabDatasetConfig
    ) -> MarketBaseline:
        metrics_path = Path(dataset.metrics_path)
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Nie udało się odczytać pliku metryk Stress Lab {metrics_path}: {exc}"
                ) from exc
        else:
            if not dataset.allow_synthetic:
                raise FileNotFoundError(
                    f"Dataset Stress Lab {metrics_path} nie istnieje, a allow_synthetic jest False"
                )
            _LOGGER.warning(
                "Stress Lab: używam syntetycznych danych dla rynku %s (plik %s)",
                dataset.symbol,
                metrics_path,
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


__all__ = ["StressLab", "StressLabReport", "StressScenarioResult", "MarketStressMetrics", "MarketBaseline"]
