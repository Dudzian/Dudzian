"""Eksporter metryk manifestu OHLCV dla Prometheusa."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Sequence

from bot_core.data.ohlcv.manifest_report import ManifestEntry
from bot_core.observability.metrics import GaugeMetric, MetricsRegistry


# Mapowanie statusów manifestu na skalę liczbową (0 = brak problemów).
STATUS_SEVERITY: Mapping[str, int] = {
    "ok": 0,
    "warning": 1,
    "missing_metadata": 2,
    "invalid_metadata": 3,
    "unknown": 4,
}


def status_to_severity(status: str) -> int:
    """Konwertuje status wpisu manifestu na wartość liczbową."""

    return STATUS_SEVERITY.get(status, STATUS_SEVERITY["unknown"])


def _sorted_labels(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


@dataclass(slots=True)
class ManifestMetricsExporter:
    """Aktualizuje rejestr metryk na podstawie wpisów z manifestu SQLite."""

    registry: MetricsRegistry
    environment: str
    exchange: str
    stage: str | None = None
    risk_profile: str | None = None
    _gap_gauge: GaugeMetric = field(init=False)
    _row_gauge: GaugeMetric = field(init=False)
    _threshold_gauge: GaugeMetric = field(init=False)
    _status_gauge: GaugeMetric = field(init=False)
    _status_totals: GaugeMetric = field(init=False)
    _known_entry_labels: set[tuple[tuple[str, str], ...]] = field(init=False, default_factory=set)
    _known_status_labels: set[tuple[tuple[str, str], ...]] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._gap_gauge = self.registry.gauge(
            "ohlcv_manifest_gap_minutes",
            "Aktualna luka czasowa (w minutach) pomiędzy manifestem a referencją.",
        )
        self._row_gauge = self.registry.gauge(
            "ohlcv_manifest_rows_total",
            "Przybliżona liczba świec zarejestrowana w manifeście OHLCV.",
        )
        self._threshold_gauge = self.registry.gauge(
            "ohlcv_manifest_gap_threshold_minutes",
            "Próg (w minutach) po którym manifest zgłasza ostrzeżenie.",
        )
        self._status_gauge = self.registry.gauge(
            "ohlcv_manifest_status_code",
            "Status wpisu manifestu w formie numerycznej (0=ok,1=warning,2=missing,3=invalid,4=unknown).",
        )
        self._status_totals = self.registry.gauge(
            "ohlcv_manifest_entries_total",
            "Łączna liczba wpisów manifestu pogrupowana według statusu.",
        )

    # ---------------------------------------------------------------------
    # Pomocnicze metody
    # ------------------------------------------------------------------
    def _base_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {
            "environment": str(self.environment),
            "exchange": str(self.exchange),
        }
        if self.stage:
            labels["stage"] = str(self.stage)
        if self.risk_profile:
            labels["risk_profile"] = str(self.risk_profile)
        return labels

    def _entry_labels(self, entry: ManifestEntry) -> dict[str, str]:
        labels = self._base_labels()
        labels["symbol"] = entry.symbol
        labels["interval"] = entry.interval
        return labels

    def _reset_missing_entries(self, active_labels: set[tuple[tuple[str, str], ...]]) -> None:
        stale = self._known_entry_labels - active_labels
        for label_tuple in stale:
            labels = dict(label_tuple)
            self._gap_gauge.set(0.0, labels=labels)
            self._row_gauge.set(0.0, labels=labels)
            self._threshold_gauge.set(0.0, labels=labels)
            self._status_gauge.set(float(status_to_severity("unknown")), labels=labels)
        self._known_entry_labels = set(active_labels)

    def _reset_missing_statuses(self, active_labels: set[tuple[tuple[str, str], ...]]) -> None:
        stale = self._known_status_labels - active_labels
        for label_tuple in stale:
            labels = dict(label_tuple)
            self._status_totals.set(0.0, labels=labels)
        self._known_status_labels = set(active_labels)

    # ------------------------------------------------------------------
    # Główna logika eksportu
    # ------------------------------------------------------------------
    def observe(self, entries: Sequence[ManifestEntry]) -> Mapping[str, object]:
        """Aktualizuje metryki i zwraca podsumowanie statusów."""

        active_entry_labels: set[tuple[tuple[str, str], ...]] = set()
        status_counts: MutableMapping[str, int] = {}

        for entry in entries:
            labels = self._entry_labels(entry)
            label_key = _sorted_labels(labels)
            active_entry_labels.add(label_key)

            gap = float(entry.gap_minutes or 0.0)
            rows = float(entry.row_count or 0)
            threshold = float(entry.threshold_minutes or 0)
            severity = float(status_to_severity(entry.status))

            self._gap_gauge.set(gap, labels=labels)
            self._row_gauge.set(rows, labels=labels)
            self._threshold_gauge.set(threshold, labels=labels)
            self._status_gauge.set(severity, labels=labels)

            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

        self._reset_missing_entries(active_entry_labels)

        base_labels = self._base_labels()
        active_status_labels: set[tuple[tuple[str, str], ...]] = set()
        for status, count in status_counts.items():
            labels = dict(base_labels)
            labels["status"] = status
            label_key = _sorted_labels(labels)
            active_status_labels.add(label_key)
            self._status_totals.set(float(count), labels=labels)

        self._reset_missing_statuses(active_status_labels)

        total_entries = sum(status_counts.values())
        worst_status = "unknown"
        if status_counts:
            worst_status = max(
                status_counts.keys(),
                key=lambda candidate: (status_to_severity(candidate), candidate),
            )

        return {
            "status_counts": dict(status_counts),
            "total_entries": total_entries,
            "worst_status": worst_status,
        }


__all__ = [
    "ManifestMetricsExporter",
    "STATUS_SEVERITY",
    "status_to_severity",
]
