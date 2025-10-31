"""Reguły walidacji zbiorów danych ML wykorzystywanych przez pipeline treningowy."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector

LOGGER = logging.getLogger(__name__)

Severity = str


@dataclass(slots=True)
class ValidationIssue:
    """Pojedynczy problem wykryty podczas walidacji."""

    rule: str
    severity: Severity
    message: str
    details: Mapping[str, object] | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(slots=True)
class ValidationReport:
    """Rezultat działania walidatora datasetu."""

    issues: tuple[ValidationIssue, ...]
    dataset_metadata: Mapping[str, object]
    generated_at: datetime

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "issues": [issue.to_mapping() for issue in self.issues],
            "dataset_metadata": dict(self.dataset_metadata),
            "status": "failed" if self.has_errors else "passed",
        }


class DatasetValidationError(RuntimeError):
    """Błąd walidacji datasetu powodujący przerwanie retreningu."""

    def __init__(self, report: ValidationReport, log_path: Path | None) -> None:
        issues = ", ".join(f"{item.rule}: {item.message}" for item in report.issues)
        super().__init__(f"Walidacja datasetu zakończyła się niepowodzeniem: {issues}")
        self.report = report
        self.log_path = log_path


@runtime_checkable
class ValidationRule(Protocol):
    """Interfejs implementowany przez reguły walidacyjne."""

    name: str

    def evaluate(self, dataset: FeatureDataset) -> Iterable[ValidationIssue]:
        """Zwraca listę problemów zidentyfikowanych w datasetcie."""


class MissingDataRule:
    """Weryfikuje obecność braków danych i wartości nie-finitycznych."""

    name = "missing_data"

    def evaluate(self, dataset: FeatureDataset) -> Iterable[ValidationIssue]:
        if not dataset.vectors:
            yield ValidationIssue(
                rule=self.name,
                severity="error",
                message="Dataset nie zawiera żadnych wektorów treningowych.",
            )
            return

        non_finite_targets: list[int] = []
        for idx, vector in enumerate(dataset.vectors):
            try:
                target = float(vector.target_bps)
            except (TypeError, ValueError):
                non_finite_targets.append(idx)
                continue
            if math.isnan(target) or not math.isfinite(target):
                non_finite_targets.append(idx)

        if non_finite_targets:
            yield ValidationIssue(
                rule=self.name,
                severity="error",
                message="Wykryto nieprawidłowe wartości targetów (NaN/inf).",
                details={"rows": non_finite_targets},
            )

        problematic_features: list[Mapping[str, object]] = []
        for idx, vector in enumerate(dataset.vectors):
            for feature, value in vector.features.items():
                if value is None:
                    problematic_features.append({"row": idx, "feature": feature, "reason": "null"})
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    problematic_features.append(
                        {"row": idx, "feature": feature, "reason": "non_numeric"}
                    )
                    continue
                if math.isnan(numeric) or not math.isfinite(numeric):
                    problematic_features.append(
                        {"row": idx, "feature": feature, "reason": "non_finite"}
                    )

        if problematic_features:
            yield ValidationIssue(
                rule=self.name,
                severity="error",
                message="Niektóre cechy zawierają brakujące lub nie-finityczne wartości.",
                details={"features": problematic_features},
            )


class AnomalyDetectionRule:
    """Wykrywa proste anomalie statystyczne w cechach."""

    name = "feature_anomalies"

    def evaluate(self, dataset: FeatureDataset) -> Iterable[ValidationIssue]:
        if len(dataset.vectors) < 2:
            return []

        feature_stats = dataset.feature_stats
        constant_features = []
        high_spread_features = []
        for name, stats in feature_stats.items():
            stdev = float(stats.get("stdev", 0.0))
            data_range = float(stats.get("max", 0.0)) - float(stats.get("min", 0.0))
            if stdev == 0.0:
                constant_features.append(name)
            elif stdev > 0 and data_range > max(stdev * 120.0, 1e-9):
                high_spread_features.append({"feature": name, "range": data_range, "stdev": stdev})

        issues: list[ValidationIssue] = []
        if constant_features:
            issues.append(
                ValidationIssue(
                    rule=self.name,
                    severity="warning",
                    message="Część cech ma zerową zmienność.",
                    details={"features": sorted(constant_features)},
                )
            )
        if high_spread_features:
            issues.append(
                ValidationIssue(
                    rule=self.name,
                    severity="warning",
                    message="Zidentyfikowano cechy o bardzo szerokim zakresie wartości względem odchylenia.",
                    details={"features": high_spread_features},
                )
            )

        return tuple(issues)


class DatasetValidator:
    """Walidator datasetów wejściowych dla pipeline'u treningowego."""

    def __init__(
        self,
        rules: Sequence[ValidationRule] | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._rules: tuple[ValidationRule, ...] = tuple(rules or (MissingDataRule(), AnomalyDetectionRule()))
        self._logger = logger or LOGGER

    def validate(self, dataset: FeatureDataset) -> ValidationReport:
        if not isinstance(dataset, FeatureDataset):
            raise TypeError("Do walidacji należy przekazać instancję FeatureDataset")

        issues: list[ValidationIssue] = []
        for rule in self._rules:
            try:
                issues.extend(rule.evaluate(dataset))
            except Exception as exc:  # pragma: no cover - zabezpieczenie przed błędami reguł
                self._logger.exception("Reguła walidacji %s zgłosiła wyjątek: %s", getattr(rule, "name", rule), exc)
                issues.append(
                    ValidationIssue(
                        rule=getattr(rule, "name", rule.__class__.__name__),
                        severity="error",
                        message=f"Reguła walidacji uległa awarii: {exc}",
                    )
                )

        metadata = dict(dataset.metadata)
        metadata.setdefault("row_count", len(dataset.vectors))
        report = ValidationReport(
            issues=tuple(issues),
            dataset_metadata=metadata,
            generated_at=datetime.now(timezone.utc),
        )
        return report

    def log_report(self, report: ValidationReport, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = report.generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"dataset_validation_{timestamp}.json"
        path.write_text(json.dumps(report.to_mapping(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def summarize_vectors(vectors: Sequence[FeatureVector]) -> Mapping[str, object]:
    """Pomocnicza funkcja wykorzystywana w raportach testowych."""

    return {
        "row_count": len(vectors),
        "symbols": sorted({vector.symbol for vector in vectors}),
    }


__all__ = [
    "AnomalyDetectionRule",
    "DatasetValidationError",
    "DatasetValidator",
    "MissingDataRule",
    "ValidationIssue",
    "ValidationReport",
]
