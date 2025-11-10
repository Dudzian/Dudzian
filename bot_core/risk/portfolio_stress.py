"""Symulacja makroekonomicznych scenariuszy stresowych portfela Stage6."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bot_core.config.models import (
    PortfolioStressAssetShockConfig,
    PortfolioStressFactorShockConfig,
    PortfolioStressScenarioConfig,
)
from bot_core.security.signing import HmacSignedReportMixin

__all__ = [
    "PortfolioStressPosition",
    "PortfolioStressBaseline",
    "PortfolioStressPositionResult",
    "PortfolioStressScenarioResult",
    "PortfolioStressReport",
    "load_portfolio_stress_baseline",
    "baseline_from_mapping",
    "run_portfolio_stress",
]


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float | None:
    """Zwraca kwantyl ważony dla przekazanych wartości."""

    if not values or not weights or len(values) != len(weights):
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Kwantyl musi mieścić się w przedziale <0, 1>")

    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return None

    cumulative = 0.0
    target = quantile * total_weight
    for value, weight in pairs:
        cumulative += max(weight, 0.0)
        if cumulative >= target:
            return value
    return pairs[-1][0]


def _weighted_expected_shortfall(
    values: Sequence[float],
    weights: Sequence[float],
    threshold: float,
) -> float | None:
    """Średnia ważona wartości poniżej zadanego progu (CVaR)."""

    if not values or not weights or len(values) != len(weights):
        return None

    loss_sum = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        if value <= threshold:
            w = max(weight, 0.0)
            loss_sum += value * w
            weight_sum += w
    if weight_sum <= 0:
        return None
    return loss_sum / weight_sum


@dataclass(slots=True)
class PortfolioStressPosition:
    """Stan pozycji wejściowej do symulacji stresowej."""

    symbol: str
    value_usd: float
    weight: float
    factor_betas: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioStressBaseline:
    """Reprezentuje portfel w chwili odniesienia."""

    portfolio_id: str
    total_value_usd: float
    cash_usd: float
    positions: Sequence[PortfolioStressPosition]
    as_of: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioStressPositionResult:
    """Wynik pojedynczej pozycji po zastosowaniu scenariusza."""

    symbol: str
    base_value_usd: float
    shocked_value_usd: float
    pnl_usd: float
    return_pct: float
    drawdown_pct: float
    weight: float
    factor_contributions: Mapping[str, float] = field(default_factory=dict)
    asset_contribution_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "base_value_usd": self.base_value_usd,
            "shocked_value_usd": self.shocked_value_usd,
            "pnl_usd": self.pnl_usd,
            "return_pct": self.return_pct,
            "drawdown_pct": self.drawdown_pct,
            "weight": self.weight,
        }
        if self.factor_contributions:
            payload["factor_contributions"] = dict(self.factor_contributions)
        if self.asset_contribution_usd is not None:
            payload["asset_contribution_usd"] = self.asset_contribution_usd
        return payload


@dataclass(slots=True)
class PortfolioStressScenarioResult:
    """Podsumowanie pojedynczego scenariusza makro."""

    scenario: PortfolioStressScenarioConfig
    total_pnl_usd: float
    total_return_pct: float
    drawdown_pct: float
    shocked_value_usd: float
    cash_pnl_usd: float
    liquidity_impact_usd: float
    factor_contributions: Mapping[str, float]
    positions: Sequence[PortfolioStressPositionResult]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    worst_position_symbol: str | None = None
    worst_position_return_pct: float | None = None
    worst_position_pnl_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        scenario = self.scenario
        payload: dict[str, Any] = {
            "name": scenario.name,
            "title": getattr(scenario, "title", None),
            "description": getattr(scenario, "description", None),
            "horizon_days": getattr(scenario, "horizon_days", None),
            "probability": getattr(scenario, "probability", None),
            "tags": list(getattr(scenario, "tags", ())),
            "total_pnl_usd": self.total_pnl_usd,
            "total_return_pct": self.total_return_pct,
            "drawdown_pct": self.drawdown_pct,
            "shocked_value_usd": self.shocked_value_usd,
            "cash_pnl_usd": self.cash_pnl_usd,
            "liquidity_impact_usd": self.liquidity_impact_usd,
            "factor_contributions": dict(self.factor_contributions),
            "positions": [position.to_dict() for position in self.positions],
        }
        if getattr(scenario, "metadata", None):
            payload["scenario_metadata"] = dict(scenario.metadata)  # type: ignore[arg-type]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.worst_position_symbol is not None:
            payload["worst_position"] = {
                "symbol": self.worst_position_symbol,
                "return_pct": self.worst_position_return_pct,
                "pnl_usd": self.worst_position_pnl_usd,
            }
        return payload


@dataclass(slots=True)
class PortfolioStressReport(HmacSignedReportMixin):
    """Zbiorczy raport z symulacji scenariuszy stresowych portfela."""

    baseline: PortfolioStressBaseline
    scenarios: Sequence[PortfolioStressScenarioResult]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "stage6.risk.portfolio_stress.report"
    schema_version: int = 1
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def _build_summary(self) -> Mapping[str, Any]:
        scenario_count = len(self.scenarios)
        summary: dict[str, Any] = {"scenario_count": scenario_count}
        if scenario_count == 0:
            summary["total_probability"] = 0.0
            summary["expected_pnl_usd"] = 0.0
            return summary

        max_drawdown = float("-inf")
        min_total_return = float("inf")
        worst: PortfolioStressScenarioResult | None = None
        max_liquidity_impact = 0.0
        total_probability = 0.0
        expected_pnl = 0.0
        probability_weights: list[float] = []
        return_values: list[float] = []
        pnl_values: list[float] = []
        tag_stats: dict[str, dict[str, Any]] = {}

        for result in self.scenarios:
            drawdown = result.drawdown_pct
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                worst = result
            total_return = result.total_return_pct
            if total_return < min_total_return:
                min_total_return = total_return
            liquidity = result.liquidity_impact_usd
            if liquidity > max_liquidity_impact:
                max_liquidity_impact = liquidity

            probability = getattr(result.scenario, "probability", None)
            if probability is not None:
                probability_value = max(float(probability), 0.0)
                total_probability += probability_value
                expected_pnl += probability_value * result.total_pnl_usd
            else:
                probability_value = 0.0

            probability_weights.append(
                probability_value if probability_value > 0 else 1.0 / scenario_count
            )
            return_values.append(result.total_return_pct)
            pnl_values.append(result.total_pnl_usd)

            tags = getattr(result.scenario, "tags", None)
            if tags:
                for tag in dict.fromkeys(tags):
                    entry = tag_stats.setdefault(
                        str(tag),
                        {
                            "scenario_count": 0,
                            "total_probability": 0.0,
                            "expected_pnl_usd": 0.0,
                            "max_drawdown_pct": float("-inf"),
                            "worst_return_pct": float("inf"),
                            "worst_scenario": None,
                        },
                    )
                    entry["scenario_count"] += 1
                    entry["total_probability"] += probability_value
                    entry["expected_pnl_usd"] += probability_value * result.total_pnl_usd
                    if drawdown > entry["max_drawdown_pct"]:
                        entry["max_drawdown_pct"] = drawdown
                    if total_return < entry["worst_return_pct"]:
                        entry["worst_return_pct"] = total_return
                        entry["worst_scenario"] = {
                            "name": result.scenario.name,
                            "title": getattr(result.scenario, "title", None),
                            "drawdown_pct": drawdown,
                            "total_return_pct": total_return,
                            "total_pnl_usd": result.total_pnl_usd,
                        }

        if max_drawdown != float("-inf"):
            summary["max_drawdown_pct"] = max_drawdown
        if min_total_return != float("inf"):
            summary["min_total_return_pct"] = min_total_return
        summary["max_liquidity_impact_usd"] = max_liquidity_impact

        if worst is not None:
            summary["worst_scenario"] = {
                "name": worst.scenario.name,
                "title": getattr(worst.scenario, "title", None),
                "drawdown_pct": worst.drawdown_pct,
                "total_return_pct": worst.total_return_pct,
                "total_pnl_usd": worst.total_pnl_usd,
                "liquidity_impact_usd": worst.liquidity_impact_usd,
            }

        summary["total_probability"] = total_probability
        summary["expected_pnl_usd"] = expected_pnl
        if total_probability > 0 and self.baseline.total_value_usd:
            summary["expected_return_pct"] = expected_pnl / self.baseline.total_value_usd

        quantile = _weighted_quantile(return_values, probability_weights, 0.05)
        if quantile is not None:
            summary["var_95_return_pct"] = quantile
            cvar_return = _weighted_expected_shortfall(
                return_values, probability_weights, quantile
            )
            if cvar_return is not None:
                summary["cvar_95_return_pct"] = cvar_return

            pnl_quantile = _weighted_quantile(pnl_values, probability_weights, 0.05)
            if pnl_quantile is not None:
                summary["var_95_pnl_usd"] = pnl_quantile
                cvar_pnl = _weighted_expected_shortfall(
                    pnl_values, probability_weights, pnl_quantile
                )
                if cvar_pnl is not None:
                    summary["cvar_95_pnl_usd"] = cvar_pnl

        if tag_stats:
            aggregates: list[dict[str, Any]] = []
            for tag, stats in sorted(tag_stats.items()):
                aggregate: dict[str, Any] = {
                    "tag": tag,
                    "scenario_count": stats["scenario_count"],
                    "total_probability": stats["total_probability"],
                    "expected_pnl_usd": stats["expected_pnl_usd"],
                }
                if stats["max_drawdown_pct"] != float("-inf"):
                    aggregate["max_drawdown_pct"] = stats["max_drawdown_pct"]
                if stats["worst_return_pct"] != float("inf"):
                    aggregate["worst_return_pct"] = stats["worst_return_pct"]
                worst_scenario = stats.get("worst_scenario")
                if worst_scenario is not None:
                    aggregate["worst_scenario"] = worst_scenario
                aggregates.append(aggregate)
            summary["tag_aggregates"] = aggregates
        return summary

    @property
    def summary(self) -> Mapping[str, Any]:
        """Udostępnia zagregowane metryki raportu."""

        return self._build_summary()

    def to_mapping(self) -> Mapping[str, Any]:
        baseline = self.baseline
        payload: dict[str, Any] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.isoformat(),
            "portfolio_id": baseline.portfolio_id,
            "baseline": {
                "portfolio_id": baseline.portfolio_id,
                "total_value_usd": baseline.total_value_usd,
                "cash_usd": baseline.cash_usd,
                "positions": [
                    {
                        "symbol": position.symbol,
                        "value_usd": position.value_usd,
                        "weight": position.weight,
                        "factor_betas": dict(position.factor_betas),
                    }
                    for position in baseline.positions
                ],
            },
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }
        summary = self._build_summary()
        if summary:
            payload["summary"] = summary
        if baseline.as_of is not None:
            payload["baseline"]["as_of"] = baseline.as_of.isoformat()
        if baseline.metadata:
            payload["baseline"]["metadata"] = dict(baseline.metadata)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def write_json(self, path: Path, *, pretty: bool = True) -> Path:
        payload = self.to_mapping()
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            if pretty:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            else:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        return path

    def write_csv(self, path: Path) -> Path:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "scenario",
            "title",
            "horizon_days",
            "total_return_pct",
            "drawdown_pct",
            "total_pnl_usd",
            "cash_pnl_usd",
            "liquidity_impact_usd",
            "worst_position",
            "worst_position_return_pct",
            "worst_position_pnl_usd",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for scenario in self.scenarios:
                writer.writerow(
                    {
                        "scenario": scenario.scenario.name,
                        "title": getattr(scenario.scenario, "title", None),
                        "horizon_days": getattr(scenario.scenario, "horizon_days", None),
                        "total_return_pct": scenario.total_return_pct,
                        "drawdown_pct": scenario.drawdown_pct,
                        "total_pnl_usd": scenario.total_pnl_usd,
                        "cash_pnl_usd": scenario.cash_pnl_usd,
                        "liquidity_impact_usd": scenario.liquidity_impact_usd,
                        "worst_position": scenario.worst_position_symbol,
                        "worst_position_return_pct": scenario.worst_position_return_pct,
                        "worst_position_pnl_usd": scenario.worst_position_pnl_usd,
                    }
                )
        return path


def _parse_datetime(value: object | None) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def baseline_from_mapping(payload: Mapping[str, Any]) -> PortfolioStressBaseline:
    portfolio_id = str(payload.get("portfolio_id", "stage6_portfolio"))
    cash_usd = float(payload.get("cash_usd", 0.0) or 0.0)
    positions_raw = payload.get("positions") or []
    if not isinstance(positions_raw, Sequence):
        raise ValueError("Sekcja 'positions' w baseline musi być listą")

    parsed_positions: list[PortfolioStressPosition] = []
    position_values: list[float] = []
    for entry in positions_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Pozycja w baseline musi być mapą")
        symbol_raw = entry.get("symbol")
        if symbol_raw in (None, ""):
            raise ValueError("Pozycja w baseline wymaga pola 'symbol'")
        symbol = str(symbol_raw)
        value_raw = entry.get("value_usd", entry.get("notional_usd"))
        try:
            value_usd = float(value_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pozycja {symbol} posiada niepoprawną wartość 'value_usd'") from exc
        weight_raw = entry.get("weight")
        weight = float(weight_raw) if weight_raw not in (None, "") else 0.0
        betas_raw = entry.get("factor_betas") or {}
        if not isinstance(betas_raw, Mapping):
            raise ValueError(f"Pozycja {symbol} posiada niepoprawną sekcję 'factor_betas'")
        betas = {str(key): float(value) for key, value in betas_raw.items()}
        metadata_raw = entry.get("metadata")
        metadata: Mapping[str, Any]
        if isinstance(metadata_raw, Mapping):
            metadata = dict(metadata_raw)
        else:
            metadata = {}
        parsed_positions.append(
            PortfolioStressPosition(
                symbol=symbol,
                value_usd=value_usd,
                weight=weight,
                factor_betas=betas,
                metadata=metadata,
            )
        )
        position_values.append(value_usd)

    positions_total = sum(position_values)
    if positions_total <= 0 and parsed_positions:
        raise ValueError("Suma wartości pozycji w baseline musi być dodatnia")

    normalized_positions: list[PortfolioStressPosition] = []
    for position, value_usd in zip(parsed_positions, position_values):
        weight = position.weight
        if weight <= 0 and positions_total > 0:
            weight = value_usd / positions_total
        normalized_positions.append(
            PortfolioStressPosition(
                symbol=position.symbol,
                value_usd=value_usd,
                weight=weight,
                factor_betas=position.factor_betas,
                metadata=position.metadata,
            )
        )

    total_value_raw = payload.get("total_value_usd")
    if total_value_raw in (None, ""):
        total_value_usd = cash_usd + positions_total
    else:
        total_value_usd = float(total_value_raw)
    if total_value_usd <= 0:
        raise ValueError("Pole 'total_value_usd' w baseline musi być dodatnie")

    metadata_raw = payload.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

    return PortfolioStressBaseline(
        portfolio_id=portfolio_id,
        total_value_usd=total_value_usd,
        cash_usd=cash_usd,
        positions=tuple(normalized_positions),
        as_of=_parse_datetime(payload.get("as_of")),
        metadata=metadata,
    )


def load_portfolio_stress_baseline(path: Path | str) -> PortfolioStressBaseline:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Plik baseline musi zawierać obiekt JSON")
    return baseline_from_mapping(payload)


def _factor_map(scenario: PortfolioStressScenarioConfig) -> Mapping[str, PortfolioStressFactorShockConfig]:
    return {shock.factor: shock for shock in getattr(scenario, "factors", ())}


def _asset_map(scenario: PortfolioStressScenarioConfig) -> Mapping[str, PortfolioStressAssetShockConfig]:
    return {shock.symbol.lower(): shock for shock in getattr(scenario, "assets", ())}


def run_portfolio_stress(
    baseline: PortfolioStressBaseline,
    scenarios: Sequence[PortfolioStressScenarioConfig],
    *,
    report_metadata: Mapping[str, Any] | None = None,
) -> PortfolioStressReport:
    results = [
        _evaluate_scenario(baseline, scenario)
        for scenario in scenarios
    ]
    return PortfolioStressReport(
        baseline=baseline,
        scenarios=tuple(results),
        metadata=dict(report_metadata or {}),
    )


def _evaluate_scenario(
    baseline: PortfolioStressBaseline,
    scenario: PortfolioStressScenarioConfig,
) -> PortfolioStressScenarioResult:
    factor_shocks = _factor_map(scenario)
    asset_shocks = _asset_map(scenario)
    total_value = baseline.total_value_usd

    position_results: list[PortfolioStressPositionResult] = []
    factor_contributions: dict[str, float] = {}
    liquidity_impact = 0.0
    worst_symbol: str | None = None
    worst_return: float | None = None
    worst_pnl: float | None = None

    for position in baseline.positions:
        value = position.value_usd
        factor_return = 0.0
        contributions: dict[str, float] = {}
        for factor, beta in position.factor_betas.items():
            shock = factor_shocks.get(factor)
            if shock is None:
                continue
            pnl_component = value * beta * shock.return_pct
            factor_return += beta * shock.return_pct
            contributions[factor] = pnl_component
            factor_contributions[factor] = factor_contributions.get(factor, 0.0) + pnl_component
            if shock.liquidity_haircut_pct is not None:
                liquidity_impact += value * max(shock.liquidity_haircut_pct, 0.0)
        asset_shock = asset_shocks.get(position.symbol.lower())
        asset_return = asset_shock.return_pct if asset_shock else 0.0
        asset_contribution = value * asset_return if asset_shock else None
        if asset_shock and asset_shock.liquidity_haircut_pct is not None:
            liquidity_impact += value * max(asset_shock.liquidity_haircut_pct, 0.0)
        total_return = factor_return + asset_return
        shocked_value = value * (1.0 + total_return)
        pnl = shocked_value - value
        drawdown_pct = max(0.0, -total_return)
        position_results.append(
            PortfolioStressPositionResult(
                symbol=position.symbol,
                base_value_usd=value,
                shocked_value_usd=shocked_value,
                pnl_usd=pnl,
                return_pct=total_return,
                drawdown_pct=drawdown_pct,
                weight=position.weight,
                factor_contributions=contributions,
                asset_contribution_usd=asset_contribution,
            )
        )
        if worst_return is None or total_return < worst_return:
            worst_return = total_return
            worst_symbol = position.symbol
            worst_pnl = pnl

    cash_return_pct = getattr(scenario, "cash_return_pct", None) or 0.0
    cash_pnl = baseline.cash_usd * cash_return_pct
    total_pnl = sum(result.pnl_usd for result in position_results) + cash_pnl
    shocked_value_usd = total_value + total_pnl
    total_return_pct = total_pnl / total_value if total_value else 0.0
    drawdown_pct = max(0.0, -total_return_pct)

    scenario_metadata: dict[str, Any] = {"cash_return_pct": cash_return_pct}
    liquidity_metadata = getattr(scenario, "metadata", None)
    if isinstance(liquidity_metadata, Mapping):
        scenario_metadata.update(dict(liquidity_metadata))
    scenario_metadata["liquidity_impact_usd"] = liquidity_impact

    return PortfolioStressScenarioResult(
        scenario=scenario,
        total_pnl_usd=total_pnl,
        total_return_pct=total_return_pct,
        drawdown_pct=drawdown_pct,
        shocked_value_usd=shocked_value_usd,
        cash_pnl_usd=cash_pnl,
        liquidity_impact_usd=liquidity_impact,
        factor_contributions=factor_contributions,
        positions=tuple(position_results),
        metadata=scenario_metadata,
        worst_position_symbol=worst_symbol,
        worst_position_return_pct=worst_return,
        worst_position_pnl_usd=worst_pnl,
    )
