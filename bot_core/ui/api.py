"""Zestaw helperów wspierających integrację backendu z aplikacją desktopową UI."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from bot_core.ai import parse_explainability_payload
from bot_core.marketplace import PresetRepository
from bot_core.runtime.journal import TradingDecisionJournal
from bot_core.strategies.installer import (
    MarketplaceInstallResult,
    MarketplacePresetInstaller,
)

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
    installed_version: str | None = None

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
            "installedVersion": self.installed_version,
            "installed": self.installed_version is not None,
        }


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

    def list_presets(self) -> list[MarketplacePresetView]:
        installed_docs = {doc.preset_id: doc for doc in self._repository.load_all()}
        views: list[MarketplacePresetView] = []
        for preset in self._installer.list_available():
            preview = self._installer.preview_installation(preset.preset_id)
            installed_doc = installed_docs.get(preset.preset_id)
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
                    installed_version=installed_doc.version if installed_doc else None,
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

