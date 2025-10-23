"""Monitorowanie jakości danych i dryfu dla pipeline'u AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


def _to_seconds(value: timedelta | pd.Timedelta | float | int | str) -> float:
    """Konwertuje różne reprezentacje okresu na liczbę sekund."""

    if isinstance(value, timedelta):
        return float(value.total_seconds())
    if isinstance(value, pd.Timedelta):  # pragma: no cover - zależne od opcjonalnej instalacji pandas
        return float(value.total_seconds())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            delta = pd.to_timedelta(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensywne
            raise ValueError(f"Nieprawidłowa wartość okresu: {value!r}") from exc
        return float(delta.total_seconds())
    raise TypeError(f"Nieobsługiwany typ okresu: {type(value)!r}")


def _isoformat(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


@dataclass(slots=True)
class DataQualityIssue:
    """Pojedynczy problem jakości danych wykryty przez monitor."""

    code: str
    message: str
    severity: str = "warning"
    details: Mapping[str, object] | None = None

    def as_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "code": str(self.code),
            "message": str(self.message),
            "severity": str(self.severity),
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(slots=True)
class DataQualityAssessment:
    """Zbiorczy wynik monitoringu jakości danych."""

    issues: Sequence[DataQualityIssue]
    summary: Mapping[str, object]
    status: str = "ok"

    def issues_payload(self) -> list[Mapping[str, object]]:
        return [issue.as_dict() for issue in self.issues]


@dataclass(slots=True)
class DataCompletenessWatcher:
    """Monitoruje luki czasowe w szeregach OHLCV lub cechach."""

    frequency: timedelta | pd.Timedelta | float | int | str
    warning_gap_ratio: float = 0.01
    critical_gap_ratio: float = 0.05
    timestamp_field: str = "timestamp"

    _frequency_seconds: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._frequency_seconds = max(_to_seconds(self.frequency), 1.0)
        self.warning_gap_ratio = max(float(self.warning_gap_ratio), 0.0)
        self.critical_gap_ratio = max(float(self.critical_gap_ratio), self.warning_gap_ratio)

    def _extract_timestamps(self, frame: pd.DataFrame, *, timestamp_field: str | None) -> pd.Series:
        if frame is None or frame.empty:
            return pd.Series(dtype="datetime64[ns]")
        if timestamp_field and timestamp_field in frame.columns:
            series = pd.to_datetime(frame[timestamp_field], utc=True, errors="coerce")
            return series.dropna()
        if isinstance(frame.index, pd.DatetimeIndex):
            return frame.index.to_series().dt.tz_convert("UTC") if frame.index.tz is not None else frame.index.to_series().dt.tz_localize("UTC")
        raise KeyError(
            f"Brak kolumny {timestamp_field!r} oraz indeks nie jest DatetimeIndex"
        )

    def assess(self, frame: pd.DataFrame, *, timestamp_field: str | None = None) -> DataQualityAssessment:
        """Analizuje DataFrame i raportuje luki względem oczekiwanej częstotliwości."""

        ts = self._extract_timestamps(frame, timestamp_field=timestamp_field or self.timestamp_field)
        if ts.empty:
            issue = DataQualityIssue(
                code="no_data",
                message="Brak obserwacji do oceny kompletności",
                severity="warning",
            )
            summary = {
                "status": "no_data",
                "observed_rows": 0,
                "expected_rows": 0,
                "frequency_seconds": self._frequency_seconds,
            }
            return DataQualityAssessment(issues=(issue,), summary=summary, status="warning")

        ordered = ts.sort_values()
        timestamps = ordered.to_numpy(dtype="datetime64[ns]")
        deltas = np.diff(timestamps.astype("datetime64[ns]"))
        gap_seconds = deltas.astype("timedelta64[s]").astype(float)
        expected = self._frequency_seconds

        issues: list[DataQualityIssue] = []
        missing_total = 0
        expected_intervals = len(gap_seconds)

        for idx, gap in enumerate(gap_seconds):
            if gap <= expected:
                continue
            missing = int(max(round(gap / expected) - 1, 0))
            if missing <= 0:
                continue
            missing_total += missing
            start = ordered.iloc[idx]
            end = ordered.iloc[idx + 1]
            start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
            end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
            gap_ratio = missing / max(len(ordered), 1)
            severity = "critical" if gap_ratio >= self.critical_gap_ratio else "warning"
            issues.append(
                DataQualityIssue(
                    code="missing_bars",
                    message=f"Brak {missing} obserwacji pomiędzy {start.isoformat()} a {end.isoformat()}",
                    severity=severity,
                    details={
                        "start": _isoformat(start),
                        "end": _isoformat(end),
                        "missing_bars": missing,
                        "gap_seconds": float(gap),
                    },
                )
            )

        expected_rows = expected_intervals + missing_total + 1
        missing_ratio = (missing_total / expected_rows) if expected_rows else 0.0
        ok_ratio = max(0.0, 1.0 - missing_ratio)

        status = "ok"
        if missing_ratio > 0:
            status = "warning"
        if missing_ratio >= self.critical_gap_ratio:
            status = "critical"

        summary = {
            "status": status,
            "observed_rows": int(len(timestamps)),
            "expected_rows": int(expected_rows),
            "total_gaps": int(missing_total),
            "missing_ratio": float(missing_ratio),
            "ok_ratio": float(ok_ratio),
            "frequency_seconds": float(expected),
            "first_timestamp": _isoformat(ordered.iloc[0]),
            "last_timestamp": _isoformat(ordered.iloc[-1]),
        }
        return DataQualityAssessment(issues=issues, summary=summary, status=status)


def _compute_psi(
    baseline: np.ndarray,
    production: np.ndarray,
    *,
    bins: int,
    epsilon: float,
) -> float:
    if baseline.size == 0 or production.size == 0:
        return 0.0
    percentile_count = min(bins, baseline.size)
    if percentile_count < 2:
        return 0.0
    percentiles = np.linspace(0, 100, percentile_count + 1)
    edges = np.percentile(baseline, percentiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0
    counts_baseline, edges = np.histogram(baseline, bins=edges)
    counts_production, _ = np.histogram(production, bins=edges)
    baseline_dist = counts_baseline / max(counts_baseline.sum(), 1)
    production_dist = counts_production / max(counts_production.sum(), 1)
    baseline_dist = np.clip(baseline_dist, epsilon, None)
    production_dist = np.clip(production_dist, epsilon, None)
    return float(np.sum((production_dist - baseline_dist) * np.log(production_dist / baseline_dist)))


def _compute_ks(baseline: np.ndarray, production: np.ndarray) -> float:
    if baseline.size == 0 or production.size == 0:
        return 0.0
    combined = np.sort(np.unique(np.concatenate([baseline, production])))
    if combined.size == 0:
        return 0.0
    baseline_sorted = np.sort(baseline)
    production_sorted = np.sort(production)
    baseline_cdf = np.searchsorted(baseline_sorted, combined, side="right") / baseline_sorted.size
    production_cdf = np.searchsorted(production_sorted, combined, side="right") / production_sorted.size
    return float(np.max(np.abs(baseline_cdf - production_cdf)))


@dataclass(slots=True)
class FeatureDriftAssessment:
    """Szczegółowy raport dryfu cech."""

    metrics: Mapping[str, Mapping[str, float]]
    summary: Mapping[str, object]
    triggered: bool
    issues: Sequence[DataQualityIssue] = ()

    def metrics_payload(self) -> Mapping[str, Mapping[str, float]]:
        return {name: dict(values) for name, values in self.metrics.items()}

    def issues_payload(self) -> list[Mapping[str, object]]:
        return [issue.as_dict() for issue in self.issues]


@dataclass(slots=True)
class FeatureDriftAnalyzer:
    """Porównuje rozkłady cech pomiędzy zbiorem bazowym a produkcyjnym."""

    psi_threshold: float = 0.2
    ks_threshold: float = 0.1
    min_samples: int = 30
    bins: int = 10
    epsilon: float = 1e-6

    def compare(
        self,
        baseline: pd.DataFrame,
        production: pd.DataFrame,
        *,
        features: Iterable[str] | None = None,
    ) -> FeatureDriftAssessment:
        if baseline is None or production is None:
            raise ValueError("Porównywane ramki danych nie mogą być None")

        if features is None:
            numeric_baseline = baseline.select_dtypes(include=["number"]).columns
            numeric_production = production.select_dtypes(include=["number"]).columns
            columns = sorted(set(numeric_baseline) & set(numeric_production))
        else:
            columns = [str(col) for col in features]

        metrics: dict[str, dict[str, float]] = {}
        issues: list[DataQualityIssue] = []
        triggered_features: list[str] = []
        insufficient: list[str] = []
        max_psi = 0.0
        max_ks = 0.0

        for column in columns:
            base_series = pd.to_numeric(baseline.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
            prod_series = pd.to_numeric(production.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
            base_values = base_series.to_numpy(dtype=float)
            prod_values = prod_series.to_numpy(dtype=float)

            sample_baseline = base_values.size
            sample_production = prod_values.size
            metric: dict[str, float] = {
                "baseline_mean": float(np.nanmean(base_values)) if sample_baseline else 0.0,
                "production_mean": float(np.nanmean(prod_values)) if sample_production else 0.0,
                "baseline_std": float(np.nanstd(base_values)) if sample_baseline else 0.0,
                "production_std": float(np.nanstd(prod_values)) if sample_production else 0.0,
                "baseline_count": float(sample_baseline),
                "production_count": float(sample_production),
            }

            if sample_baseline < self.min_samples or sample_production < self.min_samples:
                insufficient.append(column)
                issues.append(
                    DataQualityIssue(
                        code="insufficient_samples",
                        message=f"Za mało obserwacji do oceny dryfu dla cechy {column}",
                        severity="warning",
                        details={
                            "feature": column,
                            "baseline_count": sample_baseline,
                            "production_count": sample_production,
                        },
                    )
                )
                metrics[column] = metric
                continue

            psi = abs(_compute_psi(base_values, prod_values, bins=self.bins, epsilon=self.epsilon))
            ks = _compute_ks(base_values, prod_values)
            mean_diff = abs(metric["production_mean"] - metric["baseline_mean"])
            metric.update({"psi": float(psi), "ks": float(ks), "mean_abs_diff": float(mean_diff)})

            max_psi = max(max_psi, psi)
            max_ks = max(max_ks, ks)

            if psi >= self.psi_threshold or ks >= self.ks_threshold:
                triggered_features.append(column)

            metrics[column] = metric

        summary: dict[str, object] = {
            "features_evaluated": len(columns),
            "insufficient_features": insufficient,
            "psi_threshold": float(self.psi_threshold),
            "ks_threshold": float(self.ks_threshold),
            "max_psi": float(max_psi),
            "max_ks": float(max_ks),
            "triggered_features": triggered_features,
        }
        triggered = bool(triggered_features)

        if not columns:
            issues.append(
                DataQualityIssue(
                    code="no_numeric_features",
                    message="Brak wspólnych cech numerycznych do porównania",
                    severity="warning",
                )
            )
            summary["max_psi"] = 0.0
            summary["max_ks"] = 0.0

        return FeatureDriftAssessment(metrics=metrics, summary=summary, triggered=triggered, issues=issues)


@dataclass(slots=True)
class FeatureBoundsValidator:
    """Waliduje, czy obserwacje mieszczą się w zakresie średnia ± n * sigma."""

    sigma_multiplier: float = 3.0
    report_missing: bool = False

    def validate(
        self,
        features: Mapping[str, float],
        scalers: Mapping[str, tuple[float, float]],
    ) -> Sequence[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for name, params in scalers.items():
            if not isinstance(params, Sequence) or len(params) < 2:
                continue
            mean = float(params[0])
            stdev = float(params[1])
            if stdev <= 0:
                continue
            value = features.get(name)
            if value is None:
                if self.report_missing:
                    issues.append(
                        DataQualityIssue(
                            code="missing_feature",
                            message=f"Brak wartości cechy {name}",
                            severity="warning",
                            details={"feature": name},
                        )
                    )
                continue
            deviation = float(value) - mean
            limit = float(self.sigma_multiplier * stdev)
            if abs(deviation) > limit:
                issues.append(
                    DataQualityIssue(
                        code="feature_out_of_bounds",
                        message=f"Cechy {name} wykracza poza {self.sigma_multiplier}σ",
                        severity="critical",
                        details={
                            "feature": name,
                            "value": float(value),
                            "mean": mean,
                            "stdev": stdev,
                            "limit": limit,
                            "deviation": deviation,
                        },
                    )
                )
        return issues

    def is_within_bounds(
        self,
        features: Mapping[str, float],
        scalers: Mapping[str, tuple[float, float]],
    ) -> bool:
        return not self.validate(features, scalers)


__all__ = [
    "DataQualityIssue",
    "DataQualityAssessment",
    "DataCompletenessWatcher",
    "FeatureDriftAnalyzer",
    "FeatureDriftAssessment",
    "FeatureBoundsValidator",
]

