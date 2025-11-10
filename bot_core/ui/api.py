"""Zestaw helperów wspierających integrację backendu z aplikacją desktopową UI."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from packaging.version import InvalidVersion, Version

from bot_core.ai import parse_explainability_payload
from bot_core.marketplace import (
    MarketplaceIndex,
    MarketplacePlan,
    PresetDocument,
    PresetRepository,
    build_marketplace_preset,
)
from bot_core.marketplace.assignments import PresetAssignmentStore
from bot_core.runtime.journal import TradingDecisionJournal
from bot_core.strategies.installer import (
    MarketplaceInstallResult,
    MarketplacePresetInstaller,
)
from bot_core.security.license import summarize_license_payload

try:  # pragma: no cover - środowiska bez pełnej warstwy strategii
    from bot_core.strategies.catalog import StrategyCatalog, StrategyDefinition
except Exception:  # pragma: no cover
    StrategyCatalog = Any  # type: ignore
    StrategyDefinition = Any  # type: ignore

try:  # pragma: no cover - import tylko do adnotacji typów
    from bot_core.runtime.controller import TradingController
except Exception:  # pragma: no cover
    TradingController = Any  # type: ignore


@dataclass(slots=True)
class ExplainabilityEntry:
    """Pojedynczy wpis raportowany do UI."""

    event: str
    timestamp: str
    symbol: str | None
    side: str | None
    model: str | None
    method: str
    top_features: Sequence[str]
    summary: str | None


@dataclass(slots=True)
class StrategySummary:
    """Opis pojedynczej strategii prezentowany w panelu zarządzania."""

    name: str
    engine: str
    license_tier: str
    risk_classes: Sequence[str]
    required_data: Sequence[str]
    tags: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyRuntimeState:
    """Bieżący stan strategii uruchomionej w kontrolerze."""

    name: str | None
    mode: str
    ai_failover_active: bool
    ai_health: str | None
    last_signal_at: str | None
    last_event: Mapping[str, str] | None


@dataclass(slots=True)
class PortfolioStatus:
    """Migawka portfela przekazywana do UI."""

    portfolio_id: str
    environment: str
    risk_profile: str
    total_equity: float
    available_margin: float
    maintenance_margin: float
    balances: Mapping[str, float]
    risk_snapshot: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ExchangeStatus:
    """Stan giełdy/adaptatora widoczny w UI."""

    name: str
    status: str
    breaker_state: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlertEntry:
    """Alert zapisany w audycie, wykorzystywany przez UI."""

    title: str
    category: str
    severity: str
    timestamp: str
    channel: str


@dataclass(slots=True)
class RuntimeSnapshot:
    """Komplet danych synchronizowanych pomiędzy backendem a UI."""

    portfolio: PortfolioStatus
    strategies: Sequence[StrategySummary]
    runtime_state: StrategyRuntimeState
    exchanges: Sequence[ExchangeStatus]
    explainability: Sequence[ExplainabilityEntry]
    alerts: Sequence[AlertEntry]


@dataclass(slots=True)
class MarketplacePresetView:
    """Reprezentacja presetu Marketplace dla warstwy UI."""

    preset_id: str
    name: str
    version: str
    summary: str | None
    required_exchanges: Sequence[str]
    tags: Sequence[str]
    license_tier: str | None
    artifact_path: Path
    signature_verified: bool
    fingerprint_verified: bool | None
    issues: Sequence[str]
    warnings: Sequence[str] = field(default_factory=tuple)
    warning_messages: Sequence[str] = field(default_factory=tuple)
    installed_version: str | None = None
    dependencies: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    update_channels: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    preferred_channel: str | None = None
    assigned_portfolios: Sequence[str] = field(default_factory=tuple)
    upgrade_available: bool = False
    upgrade_version: str | None = None
    license: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "presetId": self.preset_id,
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
            "requiredExchanges": list(self.required_exchanges),
            "tags": list(self.tags),
            "licenseTier": self.license_tier,
            "artifactPath": str(self.artifact_path),
            "signatureVerified": self.signature_verified,
            "fingerprintVerified": self.fingerprint_verified,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "warningMessages": list(self.warning_messages),
            "installedVersion": self.installed_version,
            "installed": self.installed_version is not None,
            "dependencies": [dict(entry) for entry in self.dependencies],
            "updateChannels": [dict(entry) for entry in self.update_channels],
            "preferredChannel": self.preferred_channel,
            "assignedPortfolios": list(self.assigned_portfolios),
            "upgradeAvailable": self.upgrade_available,
            "upgradeVersion": self.upgrade_version,
            "license": _to_json_compatible(self.license) if self.license is not None else None,
        }


def _to_json_compatible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_json_compatible(item) for item in value]
    return value


def _extract_payload(record: Mapping[str, str]) -> str | None:
    for key in (
        "ai_explainability_json",
        "signal_ai_explainability_json",
        "ai_explainability",
        "signal_ai_explainability",
    ):
        payload = record.get(key)
        if payload:
            return payload
    return None


def build_explainability_feed(
    journal: TradingDecisionJournal,
    *,
    limit: int = 20,
) -> list[ExplainabilityEntry]:
    """Przygotowuje listę wpisów explainability do prezentacji w UI."""

    exported = list(journal.export())
    feed: list[ExplainabilityEntry] = []
    for record in reversed(exported):
        payload = _extract_payload(record)
        if not payload:
            continue
        report = parse_explainability_payload(payload)
        if report is None:
            continue
        entry = ExplainabilityEntry(
            event=record.get("event", ""),
            timestamp=record.get("timestamp", ""),
            symbol=record.get("symbol"),
            side=record.get("side"),
            model=report.model_name,
            method=report.method,
            top_features=[attr.name for attr in report.attributions[:5]],
            summary=report.summary,
        )
        feed.append(entry)
        if len(feed) >= limit:
            break
    return feed


def describe_strategy_catalog(
    catalog: StrategyCatalog,
    definitions: Mapping[str, StrategyDefinition],
    *,
    include_metadata: bool = True,
) -> list[StrategySummary]:
    """Buduje listę strategii z katalogu wraz z metadanymi wymaganymi w UI."""

    try:
        described = catalog.describe_definitions(
            definitions,
            include_metadata=include_metadata,
        )
    except Exception as exc:  # pragma: no cover - defensywny fallback
        raise RuntimeError("Nie udało się opisać katalogu strategii") from exc

    entries: list[StrategySummary] = []
    for item in described:
        metadata = dict(item.get("metadata", {})) if include_metadata else {}
        tags = tuple(item.get("tags", ()) or ())
        entry = StrategySummary(
            name=str(item.get("name", "")),
            engine=str(item.get("engine", "")),
            license_tier=str(item.get("license_tier", "")),
            risk_classes=tuple(item.get("risk_classes", ()) or ()),
            required_data=tuple(item.get("required_data", ()) or ()),
            tags=tags,
            metadata=metadata,
        )
        entries.append(entry)
    return entries


def _to_iso(timestamp: str | datetime | None) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, str):
        text = timestamp.strip()
        return text or None
    try:
        return timestamp.astimezone().isoformat()
    except Exception:  # pragma: no cover - defensywne
        return str(timestamp)


def _collect_latest_event(journal: TradingDecisionJournal) -> Mapping[str, str] | None:
    exported = list(journal.export())
    if not exported:
        return None
    return dict(exported[-1])


def _collect_last_signal(journal: TradingDecisionJournal) -> str | None:
    exported = list(journal.export())
    for record in reversed(exported):
        if record.get("event") == "signal_received":
            return record.get("timestamp")
    return None


def _runtime_state_from_controller(controller: TradingController) -> StrategyRuntimeState:
    journal = getattr(controller, "_decision_journal", None) or controller.decision_journal
    if journal is None:
        last_event: Mapping[str, str] | None = None
        last_signal_at: str | None = None
    else:
        last_event = _collect_latest_event(journal)
        last_signal_at = _collect_last_signal(journal)

    ai_health = None
    if hasattr(controller, "_ai_health_status"):
        status = getattr(controller, "_ai_health_status")
        ai_health = getattr(status, "value", None) or str(status) if status is not None else None

    mode = "rules"
    if getattr(controller, "_ai_failover_active", False):
        mode = "rules-fallback"
    elif getattr(controller, "_ai_signal_modes", ()):  # preferujemy AI, jeśli jest aktywne
        mode = "ai"

    return StrategyRuntimeState(
        name=getattr(controller, "_strategy_name", None) or controller.strategy_name,
        mode=mode,
        ai_failover_active=bool(getattr(controller, "_ai_failover_active", False)),
        ai_health=ai_health,
        last_signal_at=last_signal_at,
        last_event=last_event,
    )


def _portfolio_status_from_controller(controller: TradingController) -> PortfolioStatus:
    snapshot = controller.account_snapshot_provider()
    balances = dict(getattr(snapshot, "balances", {}) or {})
    risk_snapshot: Mapping[str, Any] | None = None
    engine = getattr(controller, "risk_engine", None)
    if engine is not None and hasattr(engine, "snapshot_state"):
        try:
            risk_snapshot = engine.snapshot_state(controller.risk_profile)
        except NotImplementedError:  # pragma: no cover - silnik nie wspiera
            risk_snapshot = None
        except Exception:  # pragma: no cover - defensywne
            risk_snapshot = None

    return PortfolioStatus(
        portfolio_id=controller.portfolio_id,
        environment=controller.environment,
        risk_profile=controller.risk_profile,
        total_equity=float(getattr(snapshot, "total_equity", 0.0)),
        available_margin=float(getattr(snapshot, "available_margin", 0.0)),
        maintenance_margin=float(getattr(snapshot, "maintenance_margin", 0.0)),
        balances=balances,
        risk_snapshot=risk_snapshot,
    )


def _breaker_state_name(breaker: Any) -> str | None:
    state = getattr(breaker, "state", None)
    if state is None:
        return None
    if isinstance(state, str):
        return state
    return getattr(state, "value", None) or str(state)


def _status_from_breaker(breaker_state: str | None) -> str:
    if breaker_state is None:
        return "unknown"
    normalized = breaker_state.lower()
    if normalized == "open":
        return "unavailable"
    if normalized == "half_open" or normalized == "half-open":
        return "degraded"
    return "healthy"


def _exchange_statuses_from_execution(execution_service: Any) -> list[ExchangeStatus]:
    if execution_service is None:
        return []

    adapter_names: Sequence[str] = ()
    if hasattr(execution_service, "list_adapters"):
        try:
            adapter_names = tuple(execution_service.list_adapters())
        except Exception:  # pragma: no cover - fallback
            adapter_names = ()
    elif hasattr(execution_service, "adapters"):
        adapters = getattr(execution_service, "adapters")
        if isinstance(adapters, Mapping):
            adapter_names = tuple(adapters.keys())
    elif hasattr(execution_service, "_adapters"):
        adapters = getattr(execution_service, "_adapters")
        if isinstance(adapters, Mapping):
            adapter_names = tuple(adapters.keys())

    breakers: Mapping[str, Any] | None = None
    if hasattr(execution_service, "breakers"):
        candidate = getattr(execution_service, "breakers")
        if isinstance(candidate, Mapping):
            breakers = candidate
    elif hasattr(execution_service, "_breakers"):
        candidate = getattr(execution_service, "_breakers")
        if isinstance(candidate, Mapping):
            breakers = candidate

    statuses: list[ExchangeStatus] = []
    for name in adapter_names:
        breaker_state = None
        details: MutableMapping[str, Any] = {}
        if breakers is not None:
            breaker = breakers.get(name)
            breaker_state = _breaker_state_name(breaker)
            if breaker is not None:
                details["failure_count"] = getattr(breaker, "failure_count", None)
        status = _status_from_breaker(breaker_state)
        statuses.append(
            ExchangeStatus(name=str(name), status=status, breaker_state=breaker_state, details=details)
        )
    return statuses


def _collect_alerts(controller: TradingController, *, limit: int = 50) -> list[AlertEntry]:
    router = getattr(controller, "alert_router", None)
    audit_log = getattr(router, "audit_log", None)
    if audit_log is None:
        return []
    entries: list[AlertEntry] = []
    try:
        exported = list(audit_log.export())
    except Exception:  # pragma: no cover - defensywny fallback
        return []
    for record in reversed(exported):
        entry = AlertEntry(
            title=str(record.get("title", "")),
            category=str(record.get("category", "")),
            severity=str(record.get("severity", "")),
            timestamp=record.get("timestamp", ""),
            channel=str(record.get("channel", "")),
        )
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def build_runtime_snapshot(
    controller: TradingController,
    *,
    catalog: StrategyCatalog | None = None,
    strategies: Mapping[str, StrategyDefinition] | None = None,
    explainability_limit: int = 20,
) -> RuntimeSnapshot:
    """Zbiera dane o stanie runtime i przygotowuje payload dla UI."""

    if catalog is not None and strategies is not None:
        catalog_entries = describe_strategy_catalog(catalog, strategies)
    else:
        catalog_entries = []

    portfolio = _portfolio_status_from_controller(controller)
    runtime_state = _runtime_state_from_controller(controller)
    explainability: Sequence[ExplainabilityEntry] = ()
    journal = getattr(controller, "_decision_journal", None) or controller.decision_journal
    if journal is not None:
        explainability = build_explainability_feed(journal, limit=explainability_limit)

    exchanges = _exchange_statuses_from_execution(getattr(controller, "execution_service", None))
    alerts = _collect_alerts(controller)

    return RuntimeSnapshot(
        portfolio=portfolio,
        strategies=catalog_entries,
        runtime_state=runtime_state,
        exchanges=exchanges,
        explainability=explainability,
        alerts=alerts,
    )


class RuntimeStateSync:
    """Prosty mechanizm synchronizacji stanu pomiędzy backendem a UI bez WebSocketów."""

    def __init__(
        self,
        controller: TradingController,
        *,
        poll_interval: float = 2.0,
        catalog: StrategyCatalog | None = None,
        strategies: Mapping[str, StrategyDefinition] | None = None,
        explainability_limit: int = 20,
    ) -> None:
        self._controller = controller
        self._poll_interval = max(0.25, float(poll_interval))
        self._catalog = catalog
        self._strategies = strategies
        self._explainability_limit = explainability_limit
        self._listeners: list[Callable[[RuntimeSnapshot], None]] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def add_listener(self, callback: Callable[[RuntimeSnapshot], None]) -> None:
        """Rejestruje obserwatora otrzymującego kolejne snapshoty."""

        if not callable(callback):
            raise TypeError("Listener musi być wywoływalny")
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[RuntimeSnapshot], None]) -> None:
        """Usuwa wcześniej zarejestrowanego słuchacza."""

        with self._lock:
            self._listeners = [listener for listener in self._listeners if listener is not callback]

    def _notify(self, snapshot: RuntimeSnapshot) -> None:
        listeners: Sequence[Callable[[RuntimeSnapshot], None]]
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:  # pragma: no cover - słuchacze nie powinni zatrzymywać pętli
                continue

    def poll_once(self) -> RuntimeSnapshot:
        """Wymusza pojedynczy odczyt stanu i wysyła go do słuchaczy."""

        snapshot = build_runtime_snapshot(
            self._controller,
            catalog=self._catalog,
            strategies=self._strategies,
            explainability_limit=self._explainability_limit,
        )
        self._notify(snapshot)
        return snapshot

    def start(self) -> None:
        """Uruchamia pętlę pollingową w tle."""

        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _runner() -> None:
            while not self._stop_event.is_set():
                self.poll_once()
                self._stop_event.wait(self._poll_interval)

        self._thread = threading.Thread(target=_runner, name="RuntimeStateSync", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float | None = 5.0) -> None:
        """Zatrzymuje wątek pollingowy."""

        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout)
        self._thread = None


class MarketplaceService:
    """Udostępnia operacje Marketplace wykorzystywane przez UI."""

    def __init__(self, installer: MarketplacePresetInstaller, repository: PresetRepository) -> None:
        self._installer = installer
        self._repository = repository
        meta_root = repository.root / ".meta"
        self._assignments = PresetAssignmentStore(meta_root / "assignments.json")

    @staticmethod
    def _normalize_selection(preset_ids: Sequence[object]) -> list[str]:
        normalized: list[str] = []
        for value in preset_ids:
            text = str(value).strip()
            if text:
                normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_message_list(messages: Any) -> list[str]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return []
        normalized: list[str] = []
        for item in messages:
            if isinstance(item, str):
                text = item.strip()
                if text and text not in normalized:
                    normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_portfolio_list(values: Any) -> list[str]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return []
        normalized: list[str] = []
        for value in values:
            if isinstance(value, str):
                text = value.strip()
                if text and text not in normalized:
                    normalized.append(text)
        normalized.sort()
        return normalized

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(float(text))
            except ValueError:
                return None
        return None

    @classmethod
    def _warning_messages_from_license(cls, payload: Mapping[str, Any] | None) -> tuple[str, ...]:
        if not isinstance(payload, Mapping):
            return tuple()
        summary = summarize_license_payload(payload)
        messages = cls._normalize_message_list(summary.get("warningMessages"))
        return tuple(messages)

    @classmethod
    def _plan_license_entry(
        cls, preset_id: str, preview: MarketplaceInstallResult
    ) -> Mapping[str, Any]:
        license_payload = preview.license_payload if isinstance(preview.license_payload, Mapping) else None
        summary = {
            "presetId": preset_id,
            "success": preview.success,
            "signatureVerified": preview.signature_verified,
            "fingerprintVerified": preview.fingerprint_verified,
            "issues": list(preview.issues),
            "warnings": list(preview.warnings),
        }
        if license_payload is None:
            summary["licenseMissing"] = True
            summary["warningMessages"] = []
        else:
            summary["license"] = _to_json_compatible(license_payload)
            license_summary = summarize_license_payload(license_payload)
            for key, value in license_summary.items():
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    summary[key] = list(value)
                else:
                    summary[key] = value
            summary.setdefault("warningMessages", [])
            summary["warningMessages"] = cls._normalize_message_list(summary["warningMessages"])
        return summary

    def _plan_assignment_entry(
        self, preset_id: str, license_entry: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        assigned = list(self._assignments.assigned_portfolios(preset_id))
        summary: dict[str, Any] = {
            "presetId": preset_id,
            "assignedPortfolios": assigned,
            "assignedCount": len(assigned),
        }

        seat_summary: Mapping[str, Any] | None = None
        if isinstance(license_entry, Mapping):
            seat_candidate = license_entry.get("seatSummary")
            if isinstance(seat_candidate, Mapping):
                seat_summary = seat_candidate

        licensed_assignments = self._normalize_portfolio_list(
            seat_summary.get("assignments") if seat_summary else None
        )
        if licensed_assignments:
            summary["licensedAssignments"] = licensed_assignments

        assigned_set = set(assigned)
        licensed_set = set(licensed_assignments)
        unlicensed = sorted(assigned_set - licensed_set)
        if unlicensed:
            summary["unlicensedAssignments"] = unlicensed
        orphaned = sorted(licensed_set - assigned_set)
        if orphaned:
            summary["orphanedAssignments"] = orphaned

        pending_assignments = self._normalize_portfolio_list(
            seat_summary.get("pending") if seat_summary else None
        )
        if pending_assignments:
            summary["pendingAssignments"] = pending_assignments

        seat_limit = self._coerce_int(seat_summary.get("total") if seat_summary else None)
        in_use = self._coerce_int(seat_summary.get("inUse") if seat_summary else None)
        available = self._coerce_int(seat_summary.get("available") if seat_summary else None)

        if seat_limit is not None:
            summary["seatLimit"] = seat_limit
        if in_use is not None:
            summary["inUseSeats"] = in_use
        if available is not None:
            summary["availableSeats"] = available

        projected_candidates = [len(assigned)]
        if in_use is not None:
            projected_candidates.append(in_use)
        if licensed_assignments:
            projected_candidates.append(len(licensed_assignments))
        projected_in_use = max(projected_candidates) if projected_candidates else None
        if projected_in_use is not None:
            summary["projectedInUseSeats"] = projected_in_use

        seat_shortfall: int | None = None
        if seat_limit is not None and projected_in_use is not None:
            remaining = seat_limit - projected_in_use
            summary["projectedRemainingSeats"] = max(remaining, 0)
            if remaining < 0:
                seat_shortfall = abs(remaining)
                summary["seatShortfall"] = seat_shortfall
        elif seat_limit is not None:
            remaining = seat_limit - len(assigned)
            summary["projectedRemainingSeats"] = max(remaining, 0)
            if remaining < 0:
                seat_shortfall = abs(remaining)
                summary["seatShortfall"] = seat_shortfall

        warning_codes: list[str] = []
        warning_messages: list[str] = []
        if unlicensed:
            warning_codes.append("assignment-unlicensed")
            warning_messages.append(
                "Brak licencji dla portfeli: " + ", ".join(unlicensed)
            )
        if pending_assignments:
            warning_codes.append("assignment-pending")
            warning_messages.append(
                "Portfele oczekują na zatwierdzenie w licencji: "
                + ", ".join(pending_assignments)
            )
        if seat_shortfall:
            warning_codes.append("assignment-seat-shortfall")
            warning_messages.append(
                f"Brakuje {seat_shortfall} miejsc licencyjnych dla przypisanych portfeli."
            )

        if warning_codes:
            summary["warningCodes"] = warning_codes
        if warning_messages:
            summary["warningMessages"] = warning_messages

        return summary

    def _plan_portfolio_summaries(
        self, assignment_summaries: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Mapping[str, Any]]:
        def _append(entry: dict[str, Any], key: str, preset_id: str) -> None:
            bucket = entry.setdefault(key, [])
            if preset_id not in bucket:
                bucket.append(preset_id)

        def _append_warning(entry: dict[str, Any], code: str, message: str) -> None:
            if code:
                codes = entry.setdefault("warningCodes", [])
                if code not in codes:
                    codes.append(code)
            if message:
                messages = entry.setdefault("warningMessages", [])
                if message not in messages:
                    messages.append(message)

        portfolio_summaries: dict[str, dict[str, Any]] = {}

        for preset_id, summary in assignment_summaries.items():
            assigned = self._normalize_portfolio_list(summary.get("assignedPortfolios"))
            licensed = self._normalize_portfolio_list(summary.get("licensedAssignments"))
            unlicensed = set(self._normalize_portfolio_list(summary.get("unlicensedAssignments")))
            orphaned = set(self._normalize_portfolio_list(summary.get("orphanedAssignments")))
            pending = set(self._normalize_portfolio_list(summary.get("pendingAssignments")))

            participants = set(assigned) | set(licensed) | orphaned | pending
            if not participants:
                continue

            seat_shortfall = self._coerce_int(summary.get("seatShortfall"))

            for portfolio_id in sorted(participants):
                entry = portfolio_summaries.setdefault(
                    portfolio_id,
                    {"portfolioId": portfolio_id},
                )

                if portfolio_id in assigned:
                    _append(entry, "assignedPresets", preset_id)
                if portfolio_id in licensed:
                    _append(entry, "licensedPresets", preset_id)
                if portfolio_id in unlicensed:
                    _append(entry, "unlicensedPresets", preset_id)
                    _append_warning(
                        entry,
                        "portfolio-assignment-unlicensed",
                        f"Portfel {portfolio_id} nie ma licencji na preset {preset_id}.",
                    )
                if portfolio_id in orphaned:
                    _append(entry, "orphanedPresets", preset_id)
                    _append_warning(
                        entry,
                        "portfolio-license-orphaned",
                        (
                            f"Portfel {portfolio_id} jest przypisany w licencji {preset_id}, "
                            "ale nie ma lokalnego przydziału."
                        ),
                    )
                if portfolio_id in pending:
                    _append(entry, "pendingPresets", preset_id)
                    _append_warning(
                        entry,
                        "portfolio-assignment-pending",
                        f"Portfel {portfolio_id} oczekuje na zatwierdzenie w licencji {preset_id}.",
                    )
                if seat_shortfall and (portfolio_id in assigned or portfolio_id in licensed):
                    _append(entry, "seatShortfallPresets", preset_id)
                    _append_warning(
                        entry,
                        "portfolio-seat-shortfall",
                        (
                            f"Preset {preset_id} wymaga dodatkowych {seat_shortfall} miejsc licencyjnych "
                            "dla przypisanych portfeli."
                        ),
                    )

        for entry in portfolio_summaries.values():
            for key in (
                "assignedPresets",
                "licensedPresets",
                "unlicensedPresets",
                "orphanedPresets",
                "pendingPresets",
                "seatShortfallPresets",
            ):
                if key in entry:
                    entry[key] = sorted(entry[key])
            if "warningCodes" in entry:
                entry["warningCodes"] = self._normalize_message_list(entry["warningCodes"])
            if "warningMessages" in entry:
                entry["warningMessages"] = self._normalize_message_list(entry["warningMessages"])

        return portfolio_summaries

    def list_presets(self) -> list[MarketplacePresetView]:
        installed_docs = {doc.preset_id: doc for doc in self._repository.load_all()}
        views: list[MarketplacePresetView] = []
        for preset in self._installer.list_available():
            preview = self._installer.preview_installation(preset.preset_id)
            installed_doc = installed_docs.get(preset.preset_id)
            try:
                catalog_doc = self._installer.load_catalog_document(preset.preset_id)
            except Exception:  # pragma: no cover - katalog może nie zawierać presetu
                catalog_doc = installed_doc

            marketplace_entry = None
            if catalog_doc is not None:
                marketplace_entry = build_marketplace_preset(catalog_doc)
            elif installed_doc is not None:
                marketplace_entry = build_marketplace_preset(installed_doc)

            dependencies: Sequence[Mapping[str, object]] = ()
            update_channels: Sequence[Mapping[str, object]] = ()
            preferred_channel: str | None = None
            available_version = preset.version
            if marketplace_entry is not None:
                dependencies = [dep.to_payload() for dep in marketplace_entry.dependencies]
                update_channels = [channel.to_payload() for channel in marketplace_entry.update_channels]
                preferred_channel = marketplace_entry.preferred_channel
                available_version = marketplace_entry.version or available_version

            installed_version = installed_doc.version if installed_doc else None
            upgrade_available = False
            upgrade_version = None
            if available_version and installed_version:
                try:
                    if Version(installed_version) < Version(available_version):
                        upgrade_available = True
                        upgrade_version = available_version
                except InvalidVersion:
                    upgrade_available = False

            assigned = self._assignments.assigned_portfolios(preset.preset_id)

            license_payload: Mapping[str, Any] | None
            if isinstance(preview.license_payload, Mapping):
                license_payload = dict(preview.license_payload)
            else:
                license_payload = None

            warning_messages = self._warning_messages_from_license(license_payload)

            views.append(
                MarketplacePresetView(
                    preset_id=preset.preset_id,
                    name=preset.name,
                    version=preset.version,
                    summary=preset.summary,
                    required_exchanges=preset.required_exchanges,
                    tags=preset.tags,
                    license_tier=preset.license_tier,
                    artifact_path=preset.artifact_path,
                    signature_verified=preview.signature_verified,
                    fingerprint_verified=preview.fingerprint_verified,
                    issues=preview.issues,
                    warnings=preview.warnings,
                    warning_messages=warning_messages,
                    installed_version=installed_doc.version if installed_doc else None,
                    dependencies=dependencies,
                    update_channels=update_channels,
                    preferred_channel=preferred_channel,
                    assigned_portfolios=assigned,
                    upgrade_available=upgrade_available,
                    upgrade_version=upgrade_version,
                    license=license_payload,
                )
            )
        return views

    def list_presets_payload(self) -> list[dict[str, Any]]:
        return [view.to_payload() for view in self.list_presets()]

    def install_from_catalog(self, preset_id: str) -> MarketplaceInstallResult:
        return self._installer.install_from_catalog(preset_id)

    def install_from_file(self, path: str | Path) -> MarketplaceInstallResult:
        return self._installer.install_from_path(path)

    def remove_preset(self, preset_id: str) -> bool:
        return self._repository.remove(preset_id)

    def export_preset(self, preset_id: str, *, format: str = "json") -> tuple[dict[str, Any], bytes]:
        document, payload = self._repository.export_preset(preset_id, format=format)
        return document.payload, payload

    def assign_to_portfolio(self, preset_id: str, portfolio_id: str) -> tuple[str, ...]:
        return self._assignments.assign(preset_id, portfolio_id)

    def unassign_from_portfolio(self, preset_id: str, portfolio_id: str) -> tuple[str, ...]:
        return self._assignments.unassign(preset_id, portfolio_id)

    def assignments_payload(self) -> Mapping[str, tuple[str, ...]]:
        return self._assignments.all_assignments()

    def plan_installation(self, preset_ids: Sequence[str]) -> MarketplacePlan:
        selection = self._normalize_selection(preset_ids)
        documents: list[PresetDocument] = []
        seen: set[str] = set()

        for descriptor in self._installer.list_available():
            preset_id = getattr(descriptor, "preset_id", None)
            if not isinstance(preset_id, str) or not preset_id.strip():
                continue
            try:
                document = self._installer.load_catalog_document(preset_id)
            except Exception:
                continue
            if not document.preset_id or document.preset_id in seen:
                continue
            documents.append(document)
            seen.add(document.preset_id)

        installed_docs = self._repository.load_all()
        installed_versions: dict[str, str] = {}
        for document in installed_docs:
            if document.preset_id and document.preset_id not in seen:
                documents.append(document)
                seen.add(document.preset_id)
            if document.preset_id and document.version:
                installed_versions[document.preset_id] = document.version

        index = MarketplaceIndex.from_documents(documents)
        return index.plan_installation(selection, installed_versions=installed_versions)

    def plan_installation_payload(self, preset_ids: Sequence[str]) -> Mapping[str, object]:
        selection = self._normalize_selection(preset_ids)
        plan = self.plan_installation(selection)
        payload = dict(plan.to_payload())
        payload["selection"] = selection
        license_summaries: dict[str, Mapping[str, Any]] = {}
        for preset_id in plan.install_order:
            try:
                preview = self._installer.preview_installation(preset_id)
            except Exception:
                license_summaries[preset_id] = {
                    "presetId": preset_id,
                    "success": False,
                    "licenseMissing": True,
                    "issues": [],
                    "warnings": [],
                    "warningMessages": [],
                }
            else:
                license_summaries[preset_id] = dict(self._plan_license_entry(preset_id, preview))

        for preset_id in selection:
            license_summaries.setdefault(
                preset_id,
                {
                    "presetId": preset_id,
                    "success": False,
                    "licenseMissing": True,
                    "issues": [],
                    "warnings": [],
                    "warningMessages": [],
                },
            )

        assignment_summaries: dict[str, Mapping[str, Any]] = {}
        for preset_id, license_entry in license_summaries.items():
            assignment_summary = self._plan_assignment_entry(preset_id, license_entry)
            assignment_summaries[preset_id] = assignment_summary

            warning_messages = assignment_summary.get("warningMessages")
            if warning_messages:
                existing_messages = self._normalize_message_list(license_entry.get("warningMessages"))
                for message in warning_messages:
                    if message not in existing_messages:
                        existing_messages.append(message)
                license_entry["warningMessages"] = existing_messages

            warning_codes = assignment_summary.get("warningCodes")
            if warning_codes:
                existing_codes = self._normalize_message_list(license_entry.get("warningCodes"))
                for code in warning_codes:
                    if code not in existing_codes:
                        existing_codes.append(code)
                license_entry["warningCodes"] = existing_codes

        payload["licenseSummaries"] = license_summaries
        payload["assignmentSummaries"] = assignment_summaries
        portfolio_summaries = self._plan_portfolio_summaries(assignment_summaries)
        if portfolio_summaries:
            payload["portfolioSummaries"] = portfolio_summaries
        return payload


__all__ = [
    "AlertEntry",
    "ExplainabilityEntry",
    "ExchangeStatus",
    "PortfolioStatus",
    "MarketplacePresetView",
    "MarketplaceService",
    "RuntimeSnapshot",
    "RuntimeStateSync",
    "StrategyRuntimeState",
    "StrategySummary",
    "build_explainability_feed",
    "build_runtime_snapshot",
    "describe_strategy_catalog",
]

