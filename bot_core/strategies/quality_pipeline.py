"""Pipelines wspierające walidację jakości nowych strategii."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from .catalog import DEFAULT_STRATEGY_CATALOG, StrategyCatalog


@dataclass(slots=True)
class StrategyQualityScenario:
    """Opis scenariusza walidacyjnego z sandboxu backtestowego."""

    name: str
    engine: str
    parameters: Mapping[str, Any]
    calibration_metrics: Mapping[str, float]
    acceptance_criteria: Mapping[str, Mapping[str, float]]
    notes: Sequence[str]
    source_path: Path


@dataclass(slots=True)
class StrategyQualityReport:
    """Rezultat walidacji strategii w sandboxie."""

    scenario: StrategyQualityScenario
    passed: bool
    findings: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "scenario": {
                "name": self.scenario.name,
                "engine": self.scenario.engine,
                "parameters": dict(self.scenario.parameters),
                "notes": list(self.scenario.notes),
                "source_path": str(self.scenario.source_path),
            },
            "passed": self.passed,
            "findings": dict(self.findings),
        }


class StrategyQualityPipeline:
    """Ładuje scenariusze z sandboxu i generuje raporty jakości."""

    def __init__(
        self,
        catalog: StrategyCatalog | None = None,
        *,
        simulations_dir: str | Path = "data/simulations",
        report_dir: str | Path = "reports/strategies",
    ) -> None:
        self._catalog = catalog or DEFAULT_STRATEGY_CATALOG
        self._simulations_dir = Path(simulations_dir)
        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Sequence[StrategyQualityReport]:
        reports: list[StrategyQualityReport] = []
        for scenario in self._load_scenarios():
            report = self._evaluate_scenario(scenario)
            self._save_report(report)
            reports.append(report)
        self._save_summary(reports)
        return tuple(reports)

    def _load_scenarios(self) -> Sequence[StrategyQualityScenario]:
        scenarios: list[StrategyQualityScenario] = []
        if not self._simulations_dir.exists():
            return ()

        for candidate in sorted(self._simulations_dir.glob("*.yaml")):
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            for entry in raw.get("scenarios", []):
                scenario = self._parse_scenario(entry, source_path=candidate)
                scenarios.append(scenario)
        return tuple(scenarios)

    def _parse_scenario(self, payload: Mapping[str, Any], *, source_path: Path) -> StrategyQualityScenario:
        engine = str(payload.get("engine") or "").strip()
        if not engine:
            raise ValueError(f"Scenario in {source_path} is missing engine key")

        try:
            self._catalog.get(engine)
        except KeyError as exc:
            raise ValueError(
                f"Engine '{engine}' referenced in {source_path} is not registered in catalog"
            ) from exc

        name = str(payload.get("name") or engine).strip()
        parameters = dict(payload.get("parameters") or {})
        calibration_metrics = {
            key: float(value)
            for key, value in (payload.get("calibration_metrics") or {}).items()
        }

        acceptance_payload: MutableMapping[str, Mapping[str, float]] = {}
        for metric_name, rule in (payload.get("acceptance_criteria") or {}).items():
            if not isinstance(rule, Mapping):
                continue
            normalized_rule: dict[str, float] = {}
            if "min" in rule:
                normalized_rule["min"] = float(rule["min"])
            if "max" in rule:
                normalized_rule["max"] = float(rule["max"])
            if normalized_rule:
                acceptance_payload[metric_name] = normalized_rule

        notes = tuple(str(item).strip() for item in payload.get("notes", []) if str(item).strip())

        return StrategyQualityScenario(
            name=name,
            engine=engine,
            parameters=parameters,
            calibration_metrics=calibration_metrics,
            acceptance_criteria=dict(acceptance_payload),
            notes=notes,
            source_path=source_path,
        )

    def _evaluate_scenario(self, scenario: StrategyQualityScenario) -> StrategyQualityReport:
        findings: dict[str, Any] = {
            "calibration_metrics": dict(scenario.calibration_metrics),
            "acceptance_criteria": dict(scenario.acceptance_criteria),
        }

        failed_metrics: dict[str, dict[str, float]] = {}
        for metric, bounds in scenario.acceptance_criteria.items():
            observed = scenario.calibration_metrics.get(metric)
            if observed is None:
                failed_metrics[metric] = {"reason": "missing"}
                continue
            minimum = bounds.get("min")
            maximum = bounds.get("max")
            if minimum is not None and observed < minimum:
                failed_metrics[metric] = {"observed": observed, "min": minimum}
            if maximum is not None and observed > maximum:
                failed_metrics[metric] = {"observed": observed, "max": maximum}

        passed = not failed_metrics
        if failed_metrics:
            findings["failed_metrics"] = failed_metrics
        else:
            findings["status"] = "accepted"

        findings["scenario_name"] = scenario.name

        return StrategyQualityReport(scenario=scenario, passed=passed, findings=findings)

    def _save_report(self, report: StrategyQualityReport) -> None:
        destination = self._report_dir / f"{report.scenario.name}.json"
        destination.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _save_summary(self, reports: Sequence[StrategyQualityReport]) -> None:
        summary = {
            "total": len(reports),
            "passed": sum(1 for report in reports if report.passed),
            "failed": sum(1 for report in reports if not report.passed),
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "scenarios": [
                {
                    "name": report.scenario.name,
                    "engine": report.scenario.engine,
                    "passed": report.passed,
                }
                for report in reports
            ],
        }
        summary_path = self._report_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "StrategyQualityScenario",
    "StrategyQualityReport",
    "StrategyQualityPipeline",
]


def main() -> None:
    """Uruchamia pipeline walidacji po stronie CLI."""

    pipeline = StrategyQualityPipeline()
    reports = pipeline.run()
    summary = {
        "total": len(reports),
        "passed": sum(1 for report in reports if report.passed),
        "failed": sum(1 for report in reports if not report.passed),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

