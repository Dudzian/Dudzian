"""Eksport metryk ryzyka do rejestru Prometheusa."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from bot_core.observability.metrics import GaugeMetric, MetricsRegistry
from bot_core.runtime.risk_service import RiskSnapshot


_SEVERITY_CODES: Mapping[str, int] = {
    "debug": 0,
    "info": 1,
    "notice": 2,
    "warning": 3,
    "error": 4,
    "critical": 5,
}


def _severity_to_code(value: object) -> int:
    if isinstance(value, str):
        code = _SEVERITY_CODES.get(value.strip().lower())
        if code is not None:
            return code
    return 0


def _sorted_labels(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


@dataclass(slots=True)
class RiskMetricsExporter:
    """Aktualizuje metryki Prometheusa na podstawie snapshotów ryzyka."""

    registry: MetricsRegistry
    environment: str
    stage: str | None = None
    _portfolio_gauge: GaugeMetric = field(init=False)
    _drawdown_gauge: GaugeMetric = field(init=False)
    _daily_loss_gauge: GaugeMetric = field(init=False)
    _leverage_gauge: GaugeMetric = field(init=False)
    _force_liquidation_gauge: GaugeMetric = field(init=False)
    _snapshot_timestamp_gauge: GaugeMetric = field(init=False)
    _severity_gauge: GaugeMetric = field(init=False)
    _chain_length_gauge: GaugeMetric = field(init=False)
    _exposure_current_gauge: GaugeMetric = field(init=False)
    _exposure_max_gauge: GaugeMetric = field(init=False)
    _exposure_threshold_gauge: GaugeMetric = field(init=False)
    _exposure_ratio_gauge: GaugeMetric = field(init=False)
    _known_exposures: set[tuple[tuple[str, str], ...]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._portfolio_gauge = self.registry.gauge(
            "risk_portfolio_value",
            "Aktualna wartość portfela profilu ryzyka.",
        )
        self._drawdown_gauge = self.registry.gauge(
            "risk_drawdown_pct",
            "Bieżące obsunięcie portfela (0-1).",
        )
        self._daily_loss_gauge = self.registry.gauge(
            "risk_daily_loss_pct",
            "Dzienne straty w relacji do kapitału początkowego (0-1).",
        )
        self._leverage_gauge = self.registry.gauge(
            "risk_used_leverage",
            "Wykorzystana dźwignia liczona jako suma notional / kapitał.",
        )
        self._force_liquidation_gauge = self.registry.gauge(
            "risk_force_liquidation",
            "Flaga awaryjnej likwidacji pozycji (0/1).",
        )
        self._snapshot_timestamp_gauge = self.registry.gauge(
            "risk_snapshot_generated_at",
            "Znacznik czasu snapshotu ryzyka (sekundy UNIX).",
        )
        self._severity_gauge = self.registry.gauge(
            "risk_profile_min_severity",
            "Minimalna kategoria alertów wymagana przez profil ryzyka (kod numeryczny).",
        )
        self._chain_length_gauge = self.registry.gauge(
            "risk_profile_extends_chain_length",
            "Długość łańcucha dziedziczenia profilu ryzyka.",
        )
        self._exposure_current_gauge = self.registry.gauge(
            "risk_exposure_current",
            "Bieżąca wartość limitu ekspozycji.",
        )
        self._exposure_max_gauge = self.registry.gauge(
            "risk_exposure_max",
            "Maksymalna dopuszczalna wartość limitu ekspozycji.",
        )
        self._exposure_threshold_gauge = self.registry.gauge(
            "risk_exposure_threshold",
            "Próg ostrzegawczy limitu ekspozycji.",
        )
        self._exposure_ratio_gauge = self.registry.gauge(
            "risk_exposure_ratio",
            "Wykorzystanie limitu ekspozycji (current/max).",
        )

    # ------------------------------------------------------------------
    def __call__(self, snapshot: RiskSnapshot) -> None:
        self.observe(snapshot)

    # ------------------------------------------------------------------
    def observe(self, snapshot: RiskSnapshot) -> None:
        base_labels = self._base_labels(snapshot)
        self._portfolio_gauge.set(float(snapshot.portfolio_value), labels=base_labels)
        self._drawdown_gauge.set(float(snapshot.current_drawdown), labels=base_labels)
        self._daily_loss_gauge.set(float(snapshot.daily_loss), labels=base_labels)
        self._leverage_gauge.set(float(snapshot.used_leverage), labels=base_labels)
        self._force_liquidation_gauge.set(
            1.0 if snapshot.force_liquidation else 0.0, labels=base_labels
        )
        self._snapshot_timestamp_gauge.set(
            float(_to_timestamp(snapshot.generated_at)), labels=base_labels
        )

        summary = snapshot.profile_summary() or {}
        severity_code = _severity_to_code(summary.get("severity_min"))
        extends_chain = summary.get("extends_chain")
        chain_length = 0
        if isinstance(extends_chain, Sequence):
            chain_length = sum(1 for item in extends_chain if item)
        self._severity_gauge.set(float(severity_code), labels=base_labels)
        self._chain_length_gauge.set(float(chain_length), labels=base_labels)

        active_labels: set[tuple[tuple[str, str], ...]] = set()
        for exposure in snapshot.exposures:
            labels = dict(base_labels)
            labels["limit"] = exposure.code
            label_key = _sorted_labels(labels)
            active_labels.add(label_key)

            current = float(exposure.current)
            maximum = float(exposure.maximum) if exposure.maximum is not None else 0.0
            threshold = float(exposure.threshold) if exposure.threshold is not None else 0.0
            ratio = current / maximum if maximum > 0 else 0.0

            self._exposure_current_gauge.set(current, labels=labels)
            self._exposure_max_gauge.set(maximum, labels=labels)
            self._exposure_threshold_gauge.set(threshold, labels=labels)
            self._exposure_ratio_gauge.set(ratio, labels=labels)

        self._reset_missing_exposures(active_labels)

    # ------------------------------------------------------------------
    def _base_labels(self, snapshot: RiskSnapshot) -> dict[str, str]:
        labels: dict[str, str] = {
            "environment": str(self.environment),
            "profile": str(snapshot.profile_name),
        }
        if self.stage:
            labels["stage"] = str(self.stage)
        return labels

    def _reset_missing_exposures(
        self, active_labels: set[tuple[tuple[str, str], ...]]
    ) -> None:
        stale = self._known_exposures - active_labels
        for label_tuple in stale:
            labels = dict(label_tuple)
            self._exposure_current_gauge.set(0.0, labels=labels)
            self._exposure_max_gauge.set(0.0, labels=labels)
            self._exposure_threshold_gauge.set(0.0, labels=labels)
            self._exposure_ratio_gauge.set(0.0, labels=labels)
        self._known_exposures = active_labels


def _to_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        return value.timestamp()
    return value.timestamp()


__all__ = ["RiskMetricsExporter"]
