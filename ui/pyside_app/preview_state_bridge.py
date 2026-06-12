"""Typed, local-only preview snapshot bridge for PySide/QML smoke checks."""

from __future__ import annotations

from typing import Any, Final

from PySide6.QtCore import QObject, Property, Signal, Slot

Snapshot = dict[str, object]

PAPER_SESSION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "state",
        "active",
        "normalizedState",
        "simulationStatusLabel",
        "ticks",
        "simulatedCount",
        "blockedCount",
        "noOrderCount",
        "orderRows",
        "latestOrderAction",
        "latestOrderStatus",
    }
)
SCANNER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "active",
        "tickCount",
        "rows",
        "candidates",
        "rejected",
        "bestOpportunity",
        "selectedPair",
    }
)
GOVERNOR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "lastDecision",
        "latestAction",
        "latestSymbol",
        "latestReason",
        "riskProfile",
        "riskState",
        "riskBlockReason",
        "decisionRows",
    }
)
PORTFOLIO_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "equity",
        "pnl",
        "startingEquity",
        "orders",
        "simulatedCount",
        "blockedCount",
        "openPositions",
        "closedTrades",
    }
)
ALERT_TELEMETRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "alertRows",
        "unreadAlerts",
        "criticalAlerts",
        "warningAlerts",
        "infoAlerts",
        "latestAlertTitle",
        "telemetryRows",
        "latestTelemetry",
        "telemetryTick",
    }
)
RUNTIME_BOUNDARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "liveTradingDisabled",
        "exchangeIoDisabled",
        "orderSubmissionDisabled",
        "apiKeysRequired",
        "runtimeLoopStarted",
    }
)


class LocalPreviewStateBridge(QObject):
    """Read-only mirror contract for local preview snapshots.

    The bridge does not start loops, read secrets, submit orders, or touch exchange/network
    adapters. Smoke code may refresh it from already-loaded QML helper snapshots to prove the
    Python/PySide contract matches the existing local preview state shape.
    """

    snapshotsChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._paper_session_snapshot: Snapshot = {}
        self._scanner_snapshot: Snapshot = {}
        self._governor_snapshot: Snapshot = {}
        self._portfolio_snapshot: Snapshot = {}
        self._alert_telemetry_snapshot: Snapshot = {}
        self._runtime_boundary_status: Snapshot = {
            "liveTradingDisabled": True,
            "exchangeIoDisabled": True,
            "orderSubmissionDisabled": True,
            "apiKeysRequired": False,
            "runtimeLoopStarted": False,
        }

    @Property("QVariantMap", notify=snapshotsChanged)
    def paperSessionSnapshot(self) -> Snapshot:  # type: ignore[override]
        return dict(self._paper_session_snapshot)

    @Property("QVariantMap", notify=snapshotsChanged)
    def scannerSnapshot(self) -> Snapshot:  # type: ignore[override]
        return dict(self._scanner_snapshot)

    @Property("QVariantMap", notify=snapshotsChanged)
    def governorSnapshot(self) -> Snapshot:  # type: ignore[override]
        return dict(self._governor_snapshot)

    @Property("QVariantMap", notify=snapshotsChanged)
    def portfolioSnapshot(self) -> Snapshot:  # type: ignore[override]
        return dict(self._portfolio_snapshot)

    @Property("QVariantMap", notify=snapshotsChanged)
    def alertTelemetrySnapshot(self) -> Snapshot:  # type: ignore[override]
        return dict(self._alert_telemetry_snapshot)

    @Property("QVariantMap", notify=snapshotsChanged)
    def runtimeBoundaryStatus(self) -> Snapshot:  # type: ignore[override]
        return dict(self._runtime_boundary_status)

    @Property("QVariantMap", notify=snapshotsChanged)
    def schemaContract(self) -> Snapshot:  # type: ignore[override]
        return {
            "paperSession": sorted(PAPER_SESSION_FIELDS),
            "scanner": sorted(SCANNER_FIELDS),
            "governor": sorted(GOVERNOR_FIELDS),
            "portfolio": sorted(PORTFOLIO_FIELDS),
            "alertTelemetry": sorted(ALERT_TELEMETRY_FIELDS),
            "runtimeBoundary": sorted(RUNTIME_BOUNDARY_FIELDS),
        }

    @Property(bool, notify=snapshotsChanged)
    def schemaContractValid(self) -> bool:  # type: ignore[override]
        return self.validate_current_schema()

    @Property(bool, notify=snapshotsChanged)
    def runtimeBoundaryLocalOnly(self) -> bool:  # type: ignore[override]
        return self.validate_runtime_boundary()

    @Slot("QVariantMap", "QVariantMap", "QVariantMap", "QVariantMap", "QVariantMap")
    def updateSnapshots(  # noqa: N802 (Qt naming)
        self,
        paper_session: dict[str, Any],
        scanner: dict[str, Any],
        governor: dict[str, Any],
        portfolio: dict[str, Any],
        alert_telemetry: dict[str, Any],
    ) -> None:
        """Refresh the Python-side mirror from already-local QML snapshot helpers."""

        self.refresh_from_snapshots(
            paper_session=paper_session,
            scanner=scanner,
            governor=governor,
            portfolio=portfolio,
            alert_telemetry=alert_telemetry,
        )

    def refresh_from_snapshots(
        self,
        *,
        paper_session: dict[str, Any],
        scanner: dict[str, Any],
        governor: dict[str, Any],
        portfolio: dict[str, Any],
        alert_telemetry: dict[str, Any],
    ) -> None:
        """Refresh snapshots without any runtime/live side effects."""

        self._paper_session_snapshot = dict(paper_session)
        self._scanner_snapshot = dict(scanner)
        self._governor_snapshot = dict(governor)
        self._portfolio_snapshot = dict(portfolio)
        self._alert_telemetry_snapshot = dict(alert_telemetry)
        self.snapshotsChanged.emit()

    @Slot("QVariantMap")
    def updateRuntimeBoundary(self, boundary: dict[str, Any]) -> None:  # noqa: N802
        """Mirror local-only boundary flags supplied by smoke/root QML properties."""

        self._runtime_boundary_status = dict(boundary)
        self.snapshotsChanged.emit()

    @Slot(result=bool)
    def validateCurrentSchema(self) -> bool:  # noqa: N802
        return self.validate_current_schema()

    @Slot(result=bool)
    def validateRuntimeBoundary(self) -> bool:  # noqa: N802
        return self.validate_runtime_boundary()

    def validate_current_schema(self) -> bool:
        return (
            PAPER_SESSION_FIELDS.issubset(self._paper_session_snapshot)
            and SCANNER_FIELDS.issubset(self._scanner_snapshot)
            and GOVERNOR_FIELDS.issubset(self._governor_snapshot)
            and PORTFOLIO_FIELDS.issubset(self._portfolio_snapshot)
            and ALERT_TELEMETRY_FIELDS.issubset(self._alert_telemetry_snapshot)
        )

    def validate_runtime_boundary(self) -> bool:
        return (
            RUNTIME_BOUNDARY_FIELDS.issubset(self._runtime_boundary_status)
            and self._runtime_boundary_status.get("liveTradingDisabled") is True
            and self._runtime_boundary_status.get("exchangeIoDisabled") is True
            and self._runtime_boundary_status.get("orderSubmissionDisabled") is True
            and self._runtime_boundary_status.get("apiKeysRequired") is False
            and self._runtime_boundary_status.get("runtimeLoopStarted") is False
        )
