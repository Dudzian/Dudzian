"""Procedury rozruchowe spinające konfigurację z modułami runtime."""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import stat
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)

from bot_core.alerts import (
    AlertSeverity,
    AlertThrottle,
    DefaultAlertRouter,
    EmailChannel,
    FileAlertAuditLog,
    InMemoryAlertAuditLog,
    MessengerChannel,
    SMSChannel,
    SignalChannel,
    TelegramChannel,
    WhatsAppChannel,
    emit_alert,
    get_sms_provider,
)
from bot_core.alerts.base import AlertAuditLog, AlertChannel
from bot_core.alerts.channels.providers import SmsProviderConfig
from bot_core.config.loader import load_core_config
from bot_core.config.models import (
    CoreConfig,
    DecisionJournalConfig,
    EmailChannelSettings,
    EnvironmentAIConfig,
    EnvironmentAIEnsembleConfig,
    EnvironmentAIModelConfig,
    EnvironmentAIPipelineScheduleConfig,
    EnvironmentConfig,
    LicenseValidationConfig,
    MessengerChannelSettings,
    RiskProfileConfig,
    SMSProviderSettings,
    SignalChannelSettings,
    TelegramChannelSettings,
    WhatsAppChannelSettings,
)
from bot_core.config.validation import assert_core_config_valid
from bot_core.exchanges.base import (
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeCredentials,
)
from bot_core.exchanges.binance import BinanceFuturesAdapter, BinanceSpotAdapter
from bot_core.exchanges.bitfinex import BitfinexSpotAdapter
from bot_core.exchanges.bybit import BybitSpotAdapter
from bot_core.exchanges.coinbase import CoinbaseSpotAdapter
from bot_core.exchanges.kraken import KrakenFuturesAdapter, KrakenSpotAdapter
from bot_core.exchanges.nowa_gielda import NowaGieldaSpotAdapter
from bot_core.exchanges.kucoin import KuCoinSpotAdapter
from bot_core.exchanges.okx import OKXSpotAdapter
from bot_core.exchanges.zonda import ZondaSpotAdapter
from bot_core.risk.base import RiskRepository
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.factory import build_risk_profile_from_config
from bot_core.risk.repository import FileRiskRepository
from bot_core.security import SecretManager, SecretStorageError, build_service_token_validator
from bot_core.security.license import (
    LicenseValidationError,
    LicenseValidationResult,
    validate_license_from_config,
)
from bot_core.security.tokens import ServiceTokenValidator
from bot_core.runtime.tco_reporting import RuntimeTCOReporter

if TYPE_CHECKING:  # pragma: no cover - tylko do typów
    from bot_core.alerts import (
        DefaultAlertRouter,
        EmailChannel,
        MessengerChannel,
        SMSChannel,
        SignalChannel,
        TelegramChannel,
        WhatsAppChannel,
    )
    from bot_core.alerts.base import AlertAuditLog, AlertChannel
    from bot_core.alerts.channels.providers import SmsProviderConfig
    from bot_core.config.models import (
        CoreConfig,
        DecisionJournalConfig,
        EmailChannelSettings,
        EnvironmentConfig,
        MessengerChannelSettings,
        RiskProfileConfig,
        SMSProviderSettings,
        SignalChannelSettings,
        TelegramChannelSettings,
        WhatsAppChannelSettings,
    )
else:  # pragma: no cover - w runtime typy nie są wymagane
    DefaultAlertRouter = Any  # type: ignore[misc,assignment]
    AlertAuditLog = Any  # type: ignore[misc,assignment]
    AlertChannel = Any  # type: ignore[misc,assignment]
    EmailChannel = Any  # type: ignore[misc,assignment]
    SMSChannel = Any  # type: ignore[misc,assignment]
    TelegramChannel = Any  # type: ignore[misc,assignment]
    SignalChannel = Any  # type: ignore[misc,assignment]
    WhatsAppChannel = Any  # type: ignore[misc,assignment]
    MessengerChannel = Any  # type: ignore[misc,assignment]
    SmsProviderConfig = Any  # type: ignore[misc,assignment]
    CoreConfig = Any  # type: ignore[misc,assignment]
    DecisionJournalConfig = Any  # type: ignore[misc,assignment]
    EmailChannelSettings = Any  # type: ignore[misc,assignment]
    EnvironmentConfig = Any  # type: ignore[misc,assignment]
    MessengerChannelSettings = Any  # type: ignore[misc,assignment]
    RiskProfileConfig = Any  # type: ignore[misc,assignment]
    SMSProviderSettings = Any  # type: ignore[misc,assignment]
    SignalChannelSettings = Any  # type: ignore[misc,assignment]
    TelegramChannelSettings = Any  # type: ignore[misc,assignment]
    WhatsAppChannelSettings = Any  # type: ignore[misc,assignment]

from bot_core.runtime.journal import (
    InMemoryTradingDecisionJournal,
    JsonlTradingDecisionJournal,
    TradingDecisionJournal,
)
from bot_core.runtime.file_metadata import (
    collect_security_warnings,
    file_reference_metadata,
    log_security_warnings,
)
from bot_core.observability.metrics import get_global_metrics_registry
from bot_core.portfolio import PortfolioDecisionLog

try:  # pragma: no cover - DecisionOrchestrator może być opcjonalny
    from bot_core.decision import DecisionOrchestrator  # type: ignore
except Exception:  # pragma: no cover
    DecisionOrchestrator = None  # type: ignore

try:  # pragma: no cover - integracja z AIManagerem jest opcjonalna
    from KryptoLowca.ai_manager import AIManager  # type: ignore
except Exception:  # pragma: no cover - środowiska bez modułu KryptoLowca
    AIManager = None  # type: ignore

# --- Metrics service (opcjonalny – w niektórych gałęziach może nie istnieć) ---
try:  # pragma: no cover - środowiska bez grpcio lub wygenerowanych stubów
    from bot_core.runtime.metrics_service import (  # type: ignore
        MetricsServer,
        build_metrics_server_from_config,
    )
except Exception:  # pragma: no cover - brak zależności opcjonalnych
    MetricsServer = None  # type: ignore
    build_metrics_server_from_config = None  # type: ignore

try:  # pragma: no cover - risk service jest opcjonalny
    from bot_core.runtime.risk_service import (  # type: ignore
        RiskServer,
        RiskSnapshotBuilder,
        RiskSnapshotPublisher,
        build_risk_server_from_config,
    )
except Exception:  # pragma: no cover
    RiskServer = None  # type: ignore
    RiskSnapshotBuilder = None  # type: ignore
    RiskSnapshotPublisher = None  # type: ignore
    build_risk_server_from_config = None  # type: ignore

try:  # pragma: no cover - eksporter metryk może być niedostępny
    from bot_core.runtime.risk_metrics import RiskMetricsExporter  # type: ignore
except Exception:  # pragma: no cover
    RiskMetricsExporter = None  # type: ignore

try:  # pragma: no cover - PortfolioGovernor może nie istnieć w tej gałęzi
    from bot_core.portfolio import PortfolioGovernor  # type: ignore
except Exception:
    PortfolioGovernor = None  # type: ignore

try:  # pragma: no cover - sink telemetrii może być pominięty
    from bot_core.runtime.metrics_alerts import (  # type: ignore
        DEFAULT_UI_ALERTS_JSONL_PATH,
        UiTelemetryAlertSink,
    )
except Exception:  # pragma: no cover - brak telemetrii UI
    UiTelemetryAlertSink = None  # type: ignore
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

try:  # pragma: no cover - presety profili ryzyka mogą nie istnieć
    from bot_core.runtime.telemetry_risk_profiles import (  # type: ignore
        MetricsRiskProfileResolver,
        load_risk_profiles_with_metadata,
        reset_risk_profile_store,
        summarize_risk_profile,
    )
except Exception:  # pragma: no cover - brak presetów
    MetricsRiskProfileResolver = None  # type: ignore
    load_risk_profiles_with_metadata = None  # type: ignore
    reset_risk_profile_store = None  # type: ignore
    summarize_risk_profile = None  # type: ignore

_DEFAULT_ADAPTERS: dict[str, ExchangeAdapterFactory] = {
    "binance_spot": BinanceSpotAdapter,
    "binance_futures": BinanceFuturesAdapter,
    "kraken_spot": KrakenSpotAdapter,
    "kraken_futures": KrakenFuturesAdapter,
    "coinbase_spot": CoinbaseSpotAdapter,
    "bitfinex_spot": BitfinexSpotAdapter,
    "okx_spot": OKXSpotAdapter,
    "nowa_gielda_spot": NowaGieldaSpotAdapter,
    "zonda_spot": ZondaSpotAdapter,
    "bybit_spot": BybitSpotAdapter,
    "kucoin_spot": KuCoinSpotAdapter,
}

_MISSING = object()


def get_registered_adapter_factories() -> dict[str, ExchangeAdapterFactory]:
    """Zwraca aktualną mapę fabryk adapterów dostępnych w bootstrapie."""

    return dict(_DEFAULT_ADAPTERS)


def register_adapter_factory(
    name: str, factory: ExchangeAdapterFactory, *, override: bool = False
) -> None:
    """Dodaje lub aktualizuje wpis fabryki adaptera dostępnej w bootstrapie."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Nazwa fabryki adaptera musi być niepustym łańcuchem.")

    if not callable(factory):
        raise TypeError("Fabryka adaptera musi być wywoływalna.")

    normalized = name.strip()
    if normalized in _DEFAULT_ADAPTERS and not override:
        raise ValueError(
            f"Fabryka adaptera '{normalized}' jest już zarejestrowana – użyj override=True."
        )

    _DEFAULT_ADAPTERS[normalized] = factory


def unregister_adapter_factory(name: str) -> bool:
    """Usuwa fabrykę adaptera z domyślnego rejestru bootstrapu."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Nazwa fabryki adaptera musi być niepustym łańcuchem.")

    normalized = name.strip()
    return _DEFAULT_ADAPTERS.pop(normalized, None) is not None


def register_adapter_factory_from_path(
    name: str,
    target: str,
    *,
    override: bool = False,
) -> ExchangeAdapterFactory:
    """Rozwiązuje ścieżkę do obiektu i rejestruje go jako fabrykę adaptera."""

    factory = _load_callable_from_path(target)
    register_adapter_factory(name, factory, override=override)
    return factory


def _normalize_adapter_factory_spec(
    name: str,
    spec: object,
    *,
    source: str,
) -> tuple[ExchangeAdapterFactory | None, bool, bool]:
    """Zwraca fabrykę, flagę usunięcia i override dla zadeklarowanej specyfikacji."""

    override = False
    candidate = spec

    if isinstance(spec, Mapping):
        raw_override = spec.get("override")
        if raw_override is not None and not isinstance(raw_override, bool):
            raise TypeError(
                f"Pole 'override' dla fabryki '{name}' w {source} musi być wartością logiczną."
            )
        override = bool(raw_override)

        raw_remove = spec.get("remove")
        if raw_remove is not None and not isinstance(raw_remove, bool):
            raise TypeError(
                f"Pole 'remove' dla fabryki '{name}' w {source} musi być wartością logiczną."
            )
        if raw_remove:
            unexpected_keys = {"path", "factory", "callable"} & set(spec)
            if unexpected_keys:
                joined = ", ".join(sorted(unexpected_keys))
                raise ValueError(
                    f"Specyfikacja fabryki '{name}' w {source} ustawia remove=True i jednocześnie "
                    f"deklaruje klucze: {joined}."
                )
            return None, True, override

        if "factory" in spec:
            candidate = spec["factory"]
        elif "callable" in spec:
            candidate = spec["callable"]
        elif "path" in spec:
            candidate = spec["path"]
        else:
            raise ValueError(
                f"Specyfikacja fabryki '{name}' w {source} musi zawierać klucz 'path' lub 'factory'."
            )

    if candidate is None:
        return None, True, override

    if isinstance(candidate, str):
        path = candidate.strip()
        if not path:
            raise ValueError(
                f"Fabryka adaptera '{name}' w {source} wymaga niepustej ścieżki do obiektu."
            )
        factory = _load_callable_from_path(path)
        return factory, False, override

    if callable(candidate):
        return candidate, False, override

    raise TypeError(
        f"Fabryka adaptera '{name}' w {source} musi być wywoływalna lub ścieżką do obiektu."
    )


def _apply_adapter_factory_specs(
    factories: dict[str, ExchangeAdapterFactory],
    specs: Mapping[str, object],
    *,
    source: str,
    require_override: bool,
) -> None:
    """Aktualizuje lokalną mapę fabryk zgodnie ze specyfikacją."""

    if not isinstance(specs, Mapping):
        raise TypeError(
            f"Specyfikacja fabryk w {source} musi być mapowaniem nazw na definicje fabryk."
        )

    for raw_name, spec in specs.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                f"Nazwy fabryk w {source} muszą być niepustymi łańcuchami znaków."
            )
        name = raw_name.strip()
        factory, remove, override = _normalize_adapter_factory_spec(
            name,
            spec,
            source=source,
        )

        if remove:
            factories.pop(name, None)
            continue

        effective_override = override or not require_override
        if name in factories and not effective_override:
            raise ValueError(
                f"Fabryka adaptera '{name}' w {source} jest już zdefiniowana – ustaw override=True."
            )

        factories[name] = factory


def parse_adapter_factory_cli_specs(
    entries: Sequence[str] | None,
) -> dict[str, object]:
    """Normalizuje deklaracje CLI na mapę specyfikacji fabryk adapterów."""

    if entries is None:
        return {}

    if not isinstance(entries, Sequence):
        raise TypeError("Lista specyfikacji fabryk musi być sekwencją łańcuchów.")

    result: dict[str, object] = {}
    for index, raw_entry in enumerate(entries, start=1):
        if not isinstance(raw_entry, str):
            raise TypeError(
                "Specyfikacje fabryk przekazane przez CLI muszą być łańcuchami znaków."
            )

        entry = raw_entry.strip()
        if not entry:
            raise ValueError(
                f"Pusta specyfikacja fabryki na pozycji {index} w argumentach CLI."
            )

        if "=" not in entry:
            raise ValueError(
                "Każda specyfikacja fabryki musi mieć format 'nazwa=wartość'."
            )

        name_part, spec_part = entry.split("=", 1)
        name = name_part.strip()
        spec = spec_part.strip()

        if not name:
            raise ValueError(
                f"Specyfikacja fabryki na pozycji {index} nie ma poprawnej nazwy."
            )

        if not spec:
            raise ValueError(
                f"Specyfikacja fabryki '{name}' nie ma zdefiniowanej wartości."
            )

        if name in result:
            raise ValueError(f"Fabryka '{name}' została podana wielokrotnie w argumentach CLI.")

        lowered = spec.lower()
        if lowered in {"!remove", "remove"}:
            result[name] = {"remove": True}
            continue

        if lowered.startswith("json:"):
            payload = spec[5:].lstrip()
            if not payload:
                raise ValueError(
                    f"Specyfikacja fabryki '{name}' ma prefiks json:, ale nie zawiera ładunku."
                )
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensywnie
                raise ValueError(
                    f"Nie udało się zdekodować JSON dla fabryki '{name}' w argumencie CLI."
                ) from exc
            if not isinstance(decoded, Mapping):
                raise TypeError(
                    f"Specyfikacja json: dla fabryki '{name}' musi dekodować się do mapowania."
                )
            result[name] = dict(decoded)
            continue

        if spec.startswith("{"):
            try:
                decoded = json.loads(spec)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensywnie
                raise ValueError(
                    f"Nie udało się zdekodować JSON dla fabryki '{name}' w argumencie CLI."
                ) from exc
            if not isinstance(decoded, Mapping):
                raise TypeError(
                    f"Specyfikacja JSON dla fabryki '{name}' musi być mapowaniem."
                )
            result[name] = dict(decoded)
            continue

        result[name] = spec

    return result


@contextmanager
def temporary_adapter_factories(
    *,
    add: Mapping[str, ExchangeAdapterFactory] | None = None,
    remove: Iterable[str] | None = None,
    override: bool = False,
) -> Iterator[dict[str, ExchangeAdapterFactory]]:
    """Tymczasowo modyfikuje rejestr fabryk adapterów w kontrolowanym kontekście."""

    additions: dict[str, ExchangeAdapterFactory] = {}
    if add is not None and not isinstance(add, Mapping):
        raise TypeError("Parametr 'add' musi być mapowaniem.")

    if add:
        for raw_name, factory in add.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("Nazwy fabryk w 'add' muszą być niepustymi łańcuchami.")
            if not callable(factory):
                raise TypeError("Fabryki dodawane w 'add' muszą być wywoływalne.")
            normalized = raw_name.strip()
            if normalized in additions:
                raise ValueError(
                    f"Fabryka adaptera '{normalized}' została podana wielokrotnie w 'add'."
                )
            additions[normalized] = factory

    removals: list[str] = []
    if remove is not None and not isinstance(remove, Iterable):
        raise TypeError("Parametr 'remove' musi być iterowalny.")

    if remove:
        for raw_name in remove:
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("Nazwy fabryk w 'remove' muszą być niepustymi łańcuchami.")
            normalized = raw_name.strip()
            if normalized not in removals:
                removals.append(normalized)

    overlap = [name for name in removals if name in additions]
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(
            f"Nie można jednocześnie usuwać i dodawać tych samych fabryk: {joined}."
        )

    snapshot: dict[str, object] = {}
    try:
        for name in removals:
            snapshot[name] = _DEFAULT_ADAPTERS.pop(name, _MISSING)

        for name, factory in additions.items():
            existing = _DEFAULT_ADAPTERS.get(name, _MISSING)
            if existing is not _MISSING and not override and name not in snapshot:
                raise ValueError(
                    f"Fabryka adaptera '{name}' jest już zarejestrowana – użyj override=True."
                )
            if name not in snapshot:
                snapshot[name] = existing
            _DEFAULT_ADAPTERS[name] = factory

        yield get_registered_adapter_factories()
    finally:
        for name, previous in reversed(list(snapshot.items())):
            if previous is _MISSING:
                _DEFAULT_ADAPTERS.pop(name, None)
            else:
                _DEFAULT_ADAPTERS[name] = previous  # type: ignore[assignment]


def _load_callable_from_path(target: str) -> Callable[..., Any]:
    """Resolve dotted/colon-separated path to a callable."""

    if not isinstance(target, str) or not target.strip():
        raise ValueError("Ścieżka do funkcji pipeline'u musi być niepustym łańcuchem.")

    module_path: str
    attr_path: str
    if ":" in target:
        module_path, attr_path = target.split(":", 1)
    else:
        module_path, sep, attr_path = target.rpartition(".")
        if not sep:
            raise ValueError(f"Nie można wyznaczyć modułu dla ścieżki '{target}'.")
    module = importlib.import_module(module_path)
    current: Any = module
    for part in attr_path.split("."):
        if not part:
            raise ValueError(f"Nieprawidłowy fragment ścieżki w '{target}'.")
        current = getattr(current, part)
    if not callable(current):
        raise TypeError(f"Obiekt wskazany przez '{target}' nie jest wywoływalny.")
    return current


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RuntimeEntrypoint:
    """Deklaracja punktu wejścia korzystającego z bootstrap_environment."""

    name: str
    environment: str
    description: str | None
    controller: str | None
    strategy: str | None
    risk_profile: str | None
    tags: tuple[str, ...]
    bootstrap_required: bool


def catalog_runtime_entrypoints(core_config: CoreConfig) -> dict[str, RuntimeEntrypoint]:
    """Buduje indeks punktów wejścia zadeklarowanych w konfiguracji."""

    result: dict[str, RuntimeEntrypoint] = {}
    entrypoint_cfg = getattr(core_config, "runtime_entrypoints", {}) or {}
    for name, cfg in entrypoint_cfg.items():
        tags = tuple(getattr(cfg, "tags", ()) or ())
        result[name] = RuntimeEntrypoint(
            name=name,
            environment=cfg.environment,
            description=getattr(cfg, "description", None),
            controller=getattr(cfg, "controller", None),
            strategy=getattr(cfg, "strategy", None),
            risk_profile=getattr(cfg, "risk_profile", None),
            tags=tags,
            bootstrap_required=bool(getattr(cfg, "bootstrap", True)),
        )
    return result


def resolve_runtime_entrypoint(
    entrypoint_name: str,
    *,
    config_path: str | Path,
    secret_manager: SecretManager | None = None,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    risk_profile_name: str | None = None,
    bootstrap: bool | None = None,
) -> tuple[RuntimeEntrypoint, BootstrapContext | None]:
    """Zwraca deklarację punktu wejścia i opcjonalnie inicjalizuje runtime."""

    core_config = load_core_config(config_path)
    entrypoints = catalog_runtime_entrypoints(core_config)
    if entrypoint_name not in entrypoints:
        available = ", ".join(sorted(entrypoints)) or "<none>"
        raise KeyError(
            f"Punkt wejścia '{entrypoint_name}' nie istnieje w konfiguracji. Dostępne: {available}."
        )

    entrypoint = entrypoints[entrypoint_name]
    should_bootstrap = entrypoint.bootstrap_required if bootstrap is None else bool(bootstrap)
    context: BootstrapContext | None = None
    effective_profile = risk_profile_name or entrypoint.risk_profile

    if should_bootstrap:
        if secret_manager is None:
            raise ValueError(
                "resolve_runtime_entrypoint wymaga SecretManager, gdy bootstrap jest aktywny."
            )
        context = bootstrap_environment(
            entrypoint.environment,
            config_path=config_path,
            secret_manager=secret_manager,
            adapter_factories=adapter_factories,
            risk_profile_name=effective_profile,
            core_config=core_config,
        )

    return entrypoint, context


def _build_ui_alert_audit_metadata(
    router: DefaultAlertRouter | None,
    *,
    requested_backend: str | None,
) -> dict[str, object]:
    """Zwraca metadane backendu audytu alertów UI dostępne w runtime."""

    components = _get_alert_components()
    FileAlertAuditLogCls = components["FileAlertAuditLog"]
    InMemoryAlertAuditLogCls = components["InMemoryAlertAuditLog"]

    normalized_request = (requested_backend or "inherit").lower()
    metadata: dict[str, object] = {"requested": normalized_request}

    audit_log = getattr(router, "audit_log", None)

    if isinstance(audit_log, FileAlertAuditLogCls):
        metadata.update(
            {
                "backend": "file",
                "directory": str(getattr(audit_log, "directory", "")) or None,
                "pattern": getattr(audit_log, "filename_pattern", None),
                "retention_days": getattr(audit_log, "retention_days", None),
                "fsync": bool(getattr(audit_log, "fsync", False)),
            }
        )
    elif isinstance(audit_log, InMemoryAlertAuditLogCls):
        metadata["backend"] = "memory"
    elif audit_log is None:
        metadata["backend"] = None
    else:  # pragma: no cover - diagnostyka innych backendów
        metadata["backend"] = audit_log.__class__.__name__.lower()

    note: str | None = None
    backend_value = metadata.get("backend")
    if normalized_request == "inherit":
        note = "inherited_environment_router"
    elif normalized_request == "file" and backend_value != "file":
        note = "file_backend_unavailable"
    elif normalized_request == "memory" and backend_value != "memory":
        note = "memory_backend_not_selected"

    if backend_value is None and note is None:
        note = "router_missing_audit_log"

    if note is not None:
        metadata["note"] = note

    return metadata


def _config_value(source: object, *names: str) -> Any:
    """Zwraca pierwszą niepustą wartość z obiektu konfiguracji."""

    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return None


def _load_initial_tco_costs(
    config: Any,
    orchestrator: Any,
    portfolio_governor: Any | None = None,
) -> tuple[str | None, Sequence[str]]:
    """Ładuje raporty TCO z konfiguracji do DecisionOrchestratora."""

    warnings: list[str] = []
    tco_config = getattr(config, "tco", None)
    if not tco_config:
        return None, ()

    warn_age = _config_value(tco_config, "warn_report_age_hours")
    max_age = _config_value(tco_config, "max_report_age_hours")
    try:
        warn_age = float(warn_age) if warn_age is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensywnie
        _LOGGER.debug("Nie można zinterpretować warn_report_age_hours=%r", warn_age)
        warn_age = None
    try:
        max_age = float(max_age) if max_age is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensywnie
        _LOGGER.debug("Nie można zinterpretować max_report_age_hours=%r", max_age)
        max_age = None

    report_paths_attr = getattr(tco_config, "report_paths", None)
    if not report_paths_attr:
        report_paths_attr = getattr(tco_config, "reports", ())
    report_paths = tuple(report_paths_attr or ())
    require_at_startup = bool(getattr(tco_config, "require_at_startup", False))
    now = datetime.now(timezone.utc)
    for raw_path in report_paths:
        path = Path(str(raw_path)).expanduser()
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            warning = f"missing:{path}"
            warnings.append(warning)
            _LOGGER.warning("Raport TCO %s nie istnieje", path)
            continue

        age_seconds = max(0.0, now.timestamp() - stat_result.st_mtime)
        age_hours = age_seconds / 3600.0
        if max_age is not None and age_hours >= max_age:
            warning = f"stale_critical:{path}:{age_hours:.2f}h"
            warnings.append(warning)
            _LOGGER.error(
                "Raport TCO %s jest starszy niż maksymalne %s godzin (wiek %.2f h)",
                path,
                max_age,
                age_hours,
            )
            continue
        if warn_age is not None and age_hours >= warn_age:
            warning = f"stale_warning:{path}:{age_hours:.2f}h"
            warnings.append(warning)
            _LOGGER.warning(
                "Raport TCO %s jest starszy niż zalecane %s godzin (wiek %.2f h)",
                path,
                warn_age,
                age_hours,
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            warning = f"missing:{path}"
            warnings.append(warning)
            _LOGGER.warning("Raport TCO %s nie istnieje", path)
            continue
        except Exception:
            warning = f"invalid:{path}"
            warnings.append(warning)
            _LOGGER.warning("Nie udało się wczytać raportu TCO %s", path, exc_info=True)
            continue

        loaded = True
        loaded_path = str(path)
        if orchestrator is not None:
            try:
                orchestrator.update_costs_from_report(payload)
            except Exception:
                warning = f"update_failed:{path}"
                warnings.append(warning)
                _LOGGER.warning(
                    "Nie udało się załadować raportu TCO %s do DecisionOrchestratora",
                    path,
                    exc_info=True,
                )
            else:
                loaded = True
        if portfolio_governor is not None:
            try:
                portfolio_governor.update_costs_from_report(payload)
            except Exception:  # pragma: no cover - defensywnie
                _LOGGER.debug(
                    "Nie udało się zaktualizować kosztów TCO w PortfolioGovernor %s",
                    path,
                    exc_info=True,
                )
            else:
                loaded = True
        if loaded:
            _LOGGER.info("Załadowano raport TCO: %s", path)
            return loaded_path, tuple(warnings)

    if require_at_startup:
        warnings.append("missing_required_tco_report")
        _LOGGER.error(
            "Wymagany raport TCO nie został znaleziony (kandydaci=%s)",
            [str(path) for path in report_paths],
        )
    return None, tuple(warnings)


def _initialize_runtime_tco_reporter(
    config: Any,
    *,
    environment: EnvironmentConfig,
    risk_profile: str,
) -> RuntimeTCOReporter | None:
    """Buduje usługę raportowania TCO w runtime, jeśli konfiguracja ją aktywuje."""

    enabled = bool(getattr(config, "runtime_enabled", False))
    directory_value = getattr(config, "runtime_report_directory", None)
    if not enabled and not directory_value:
        return None

    if directory_value:
        directory = Path(str(directory_value)).expanduser()
    else:
        directory = Path(environment.data_cache_path).expanduser() / "tco_runtime"

    basename = getattr(config, "runtime_report_basename", None)
    formats = getattr(config, "runtime_export_formats", None)
    flush_events = getattr(config, "runtime_flush_events", None)
    flush_normalized = int(flush_events) if flush_events not in (None, "", 0) else None
    clear_after_export = bool(getattr(config, "runtime_clear_after_export", False))

    signing_key_env = getattr(config, "runtime_signing_key_env", None)
    signing_key: bytes | None = None
    if signing_key_env:
        env_value = os.environ.get(signing_key_env)
        if env_value:
            signing_key = env_value.encode("utf-8")
        else:
            _LOGGER.warning(
                "Runtime TCO signing key env %s is not set – artifacts will not be signed.",
                signing_key_env,
            )

    signing_key_id = getattr(config, "runtime_signing_key_id", None)
    metadata = dict(getattr(config, "runtime_metadata", {}) or {})
    metadata.setdefault("environment", environment.name)
    metadata.setdefault("exchange", environment.exchange)
    metadata.setdefault("risk_profile", risk_profile)
    metadata.setdefault("controller", getattr(environment, "default_controller", None) or environment.name)
    cost_limit = getattr(config, "runtime_cost_limit_bps", None)

    try:
        reporter = RuntimeTCOReporter(
            output_dir=directory,
            basename=basename,
            export_formats=formats,
            flush_events=flush_normalized,
            clear_after_export=clear_after_export,
            signing_key=signing_key,
            signing_key_id=signing_key_id,
            metadata=metadata,
            cost_limit_bps=cost_limit,
        )
    except Exception:  # pragma: no cover - reporter jest opcjonalny
        _LOGGER.exception("Nie udało się zainicjalizować RuntimeTCOReporter")
        return None

    return reporter


@dataclass(slots=True)
class BootstrapContext:
    """Zawiera wszystkie komponenty zainicjalizowane dla danego środowiska."""

    core_config: CoreConfig
    environment: EnvironmentConfig
    credentials: ExchangeCredentials
    adapter: ExchangeAdapter
    risk_engine: ThresholdRiskEngine
    risk_repository: RiskRepository
    alert_router: DefaultAlertRouter
    alert_channels: Mapping[str, AlertChannel]
    audit_log: AlertAuditLog
    adapter_settings: Mapping[str, Any]
    decision_journal: TradingDecisionJournal | None
    risk_decision_log: RiskDecisionLog | None
    risk_profile_name: str
    license_validation: LicenseValidationResult
    portfolio_decision_log: PortfolioDecisionLog | None = None
    decision_engine_config: Any | None = None
    decision_orchestrator: Any | None = None
    decision_tco_report_path: str | None = None
    decision_tco_warnings: Sequence[str] | None = None
    tco_reporter: RuntimeTCOReporter | None = None
    portfolio_governor_config: Any | None = None
    portfolio_governor: Any | None = None
    metrics_server: Any | None = None
    metrics_ui_alerts_path: Path | None = None
    metrics_jsonl_path: Path | None = None
    metrics_ui_alert_sink_active: bool = False
    metrics_service_enabled: bool | None = None
    metrics_ui_alerts_metadata: Mapping[str, Any] | None = None
    metrics_jsonl_metadata: Mapping[str, Any] | None = None
    metrics_security_warnings: tuple[str, ...] | None = None
    metrics_security_metadata: Mapping[str, Any] | None = None
    metrics_ui_alerts_settings: Mapping[str, Any] | None = None
    metrics_token_validator: ServiceTokenValidator | None = None
    risk_server: Any | None = None
    risk_snapshot_store: Any | None = None
    risk_snapshot_builder: Any | None = None
    risk_snapshot_publisher: Any | None = None
    risk_service_enabled: bool | None = None
    risk_security_metadata: Mapping[str, Any] | None = None
    risk_security_warnings: tuple[str, ...] | None = None
    risk_token_validator: ServiceTokenValidator | None = None
    ai_manager: Any | None = None
    ai_models_loaded: Sequence[str] | None = None
    ai_threshold_bps: float | None = None
    ai_model_bindings: Sequence[EnvironmentAIModelConfig] | None = None
    ai_ensembles_registered: Sequence[str] | None = None
    ai_pipeline_schedules: Sequence[str] | None = None
    ai_pipeline_pending: Sequence[str] | None = None


def bootstrap_environment(
    environment_name: str,
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    risk_profile_name: str | None = None,
    core_config: CoreConfig | None = None,
) -> BootstrapContext:
    """Tworzy kompletny kontekst uruchomieniowy dla wskazanego środowiska."""
    from bot_core.config.loader import load_core_config as _load_core_config
    from bot_core.config.validation import assert_core_config_valid

    core_config = _load_core_config(config_path)
    validation = assert_core_config_valid(core_config)
    for warning in validation.warnings:
        _LOGGER.warning("Walidacja konfiguracji: %s", warning)

    license_config = getattr(core_config, "license", None)
    if not isinstance(license_config, LicenseValidationConfig):
        license_config = LicenseValidationConfig()
    try:
        license_result = validate_license_from_config(license_config)
    except LicenseValidationError as exc:
        context = exc.result.to_context() if exc.result else {
            "status": "invalid",
            "license_path": str(Path(license_config.license_path).expanduser()),
        }
        emit_alert(
            "Weryfikacja licencji OEM zakończona błędem – zatrzymuję kontroler.",
            severity=AlertSeverity.CRITICAL,
            source="security.license",
            context=context,
            exception=exc,
        )
        _LOGGER.critical("Weryfikacja licencji OEM nie powiodła się: %s", exc)
        raise RuntimeError(str(exc)) from exc
    else:
        if license_result.warnings:
            for message in license_result.warnings:
                _LOGGER.warning("Weryfikacja licencji: %s", message)
        _LOGGER.info(
            "Licencja OEM zweryfikowana (id=%s, fingerprint=%s, wygasa=%s, revocation=%s)",
            license_result.license_id or "brak",
            license_result.fingerprint,
            license_result.expires_at or "brak informacji",
            license_result.revocation_status or "n/a",
        )
        emit_alert(
            "Licencja OEM zweryfikowana pomyślnie." if not license_result.warnings else "Licencja OEM zweryfikowana z ostrzeżeniami.",
            severity=AlertSeverity.WARNING if license_result.warnings else AlertSeverity.INFO,
            source="security.license",
            context=license_result.to_context(),
        )

    if environment_name not in core_config.environments:
        raise KeyError(f"Środowisko '{environment_name}' nie istnieje w konfiguracji")

    environment = core_config.environments[environment_name]
    offline_mode = bool(getattr(environment, "offline_mode", False))
    if offline_mode:
        _LOGGER.info(
            "Środowisko %s działa w trybie offline – pomijam komponenty wymagające sieci.",
            environment.name,
        )
    selected_profile = risk_profile_name or environment.risk_profile
    risk_profile_config = _resolve_risk_profile(core_config.risk_profiles, selected_profile)

    risk_repository_path = Path(environment.data_cache_path) / "risk_state"
    risk_repository = FileRiskRepository(risk_repository_path)
    risk_decision_log = _build_risk_decision_log(core_config, environment)
    risk_engine = ThresholdRiskEngine(
        repository=risk_repository,
        decision_log=risk_decision_log,
    )
    decision_engine_config = getattr(core_config, "decision_engine", None)
    portfolio_governor_config = getattr(core_config, "portfolio_governor", None)
    decision_orchestrator: Any | None = None
    decision_tco_report_path: str | None = None
    decision_tco_warnings: list[str] = []
    tco_reporter: RuntimeTCOReporter | None = None
    portfolio_governor: Any | None = None
    ai_manager_instance: Any | None = None
    ai_models_loaded: list[str] = []
    ai_model_bindings: Sequence[EnvironmentAIModelConfig] | None = None
    ai_threshold_bps: float | None = None
    environment_ai: EnvironmentAIConfig | None = getattr(environment, "ai", None)
    ai_ensembles_registered: list[str] = []
    ai_pipeline_schedules: list[str] = []
    ai_pipeline_pending: list[str] = []
    if portfolio_governor_config and PortfolioGovernor is not None:
        try:
            portfolio_governor = PortfolioGovernor(portfolio_governor_config)
        except Exception:  # pragma: no cover - diagnostyka inicjalizacji
            portfolio_governor = None
            _LOGGER.exception("Nie udało się zainicjalizować PortfolioGovernora")
    elif portfolio_governor_config:
        _LOGGER.debug("Konfiguracja portfolio_governor jest dostępna, ale moduł nie został załadowany")
    if decision_engine_config and DecisionOrchestrator is not None:
        try:
            decision_orchestrator = DecisionOrchestrator(decision_engine_config)
        except Exception:  # pragma: no cover - diagnostyka inicjalizacji
            decision_orchestrator = None
            _LOGGER.exception("Nie udało się zainicjalizować DecisionOrchestratora")
        else:
            try:
                risk_engine.attach_decision_orchestrator(decision_orchestrator)
            except Exception:  # pragma: no cover - diagnostyka integracji
                decision_orchestrator = None
                _LOGGER.exception(
                    "Nie udało się podłączyć DecisionOrchestratora do silnika ryzyka"
                )
            else:
                report_path, warnings = _load_initial_tco_costs(
                    decision_engine_config,
                    decision_orchestrator,
                    portfolio_governor,
                )
                if report_path is not None:
                    decision_tco_report_path = report_path
                if warnings:
                    decision_tco_warnings.extend(str(entry) for entry in warnings)
    elif portfolio_governor is not None and decision_engine_config:
        _, warnings = _load_initial_tco_costs(
            decision_engine_config,
            None,
            portfolio_governor,
        )
        if warnings:
            decision_tco_warnings.extend(str(entry) for entry in warnings)

    if isinstance(environment_ai, EnvironmentAIConfig) and environment_ai.enabled:
        ai_model_bindings = environment_ai.models
        ai_threshold_bps = float(environment_ai.threshold_bps)
        if AIManager is None:
            _LOGGER.warning(
                "Sekcja environment.ai została zignorowana: moduł KryptoLowca.ai_manager nie jest dostępny"
            )
        else:
            model_dir_value = environment_ai.model_dir
            model_dir_path = (
                Path(model_dir_value).expanduser()
                if model_dir_value
                else Path(environment.data_cache_path) / "models" / "ai_manager"
            )
            try:
                model_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - brak uprawnień nie powinien zatrzymać bootstrapu
                _LOGGER.debug(
                    "Nie udało się utworzyć katalogu modeli AI %s", model_dir_path, exc_info=True
                )
            try:
                ai_manager_instance = AIManager(
                    ai_threshold_bps=ai_threshold_bps,
                    model_dir=model_dir_path,
                )
            except Exception:  # pragma: no cover - diagnostyka inicjalizacji
                ai_manager_instance = None
                _LOGGER.exception("Nie udało się zainicjalizować AIManagera")
            else:
                ai_model_bindings = environment_ai.models
                for binding in environment_ai.models:
                    model_name = f"{binding.symbol}:{binding.model_type}"
                    try:
                        asyncio.run(
                            ai_manager_instance.import_model(
                                binding.symbol,
                                binding.model_type,
                                binding.path,
                            )
                        )
                    except FileNotFoundError:
                        _LOGGER.warning(
                            "Model AI %s nie został znaleziony pod ścieżką %s",
                            model_name,
                            binding.path,
                        )
                    except Exception:  # pragma: no cover - logujemy i kontynuujemy bootstrap
                        _LOGGER.exception(
                            "Nie udało się załadować modelu AI %s", model_name
                        )
                    else:
                        ai_models_loaded.append(model_name)
                for preload_entry in environment_ai.preload:
                    if preload_entry not in ai_models_loaded:
                        ai_models_loaded.append(preload_entry)

                if environment_ai.ensembles:
                    for ensemble in environment_ai.ensembles:
                        try:
                            weights = (
                                tuple(float(value) for value in ensemble.weights)
                                if ensemble.weights is not None
                                else None
                            )
                            ai_manager_instance.register_ensemble(
                                ensemble.name,
                                ensemble.components,
                                aggregation=ensemble.aggregation,
                                weights=weights,
                                override=False,
                            )
                        except Exception:
                            _LOGGER.exception(
                                "Nie udało się zarejestrować zespołu modeli %s",
                                ensemble.name,
                            )
                        else:
                            ai_ensembles_registered.append(ensemble.name)

                if environment_ai.pipeline_schedules:
                    for schedule in environment_ai.pipeline_schedules:
                        try:
                            df_provider = _load_callable_from_path(schedule.data_source)
                        except Exception:
                            _LOGGER.exception(
                                "Nie udało się załadować źródła danych pipeline'u %s (%s)",
                                schedule.symbol,
                                schedule.data_source,
                            )
                            continue
                        baseline_provider: Callable[..., Any] | None = None
                        if schedule.baseline_source:
                            try:
                                baseline_provider = _load_callable_from_path(
                                    schedule.baseline_source
                                )
                            except Exception:
                                _LOGGER.exception(
                                    "Nie udało się załadować bazowego źródła danych %s (%s)",
                                    schedule.symbol,
                                    schedule.baseline_source,
                                )
                                continue
                        on_result: Callable[..., Any] | None = None
                        if schedule.result_callback:
                            try:
                                on_result = _load_callable_from_path(
                                    schedule.result_callback
                                )
                            except Exception:
                                _LOGGER.exception(
                                    "Nie udało się załadować callbacku wynikowego %s (%s)",
                                    schedule.symbol,
                                    schedule.result_callback,
                                )
                                continue
                        try:
                            schedule_obj = ai_manager_instance.schedule_pipeline(
                                schedule.symbol,
                                df_provider,
                                schedule.model_types,
                                interval_seconds=schedule.interval_seconds,
                                seq_len=schedule.seq_len,
                                folds=schedule.folds,
                                baseline_provider=baseline_provider,
                                on_result=on_result,
                            )
                        except RuntimeError as exc:
                            if "no running event loop" in str(exc).lower():
                                _LOGGER.warning(
                                    "Harmonogram pipeline'u %s nie został uruchomiony: brak aktywnej pętli zdarzeń",
                                    schedule.symbol,
                                )
                                ai_pipeline_pending.append(schedule.symbol)
                            else:
                                _LOGGER.exception(
                                    "Nie udało się zaplanować pipeline'u %s",
                                    schedule.symbol,
                                )
                            continue
                        except Exception:
                            _LOGGER.exception(
                                "Nie udało się zaplanować pipeline'u %s",
                                schedule.symbol,
                            )
                            continue
                        registered_symbol = getattr(schedule_obj, "symbol", schedule.symbol)
                        ai_pipeline_schedules.append(str(registered_symbol))

    profile = build_risk_profile_from_config(risk_profile_config)
    risk_engine.register_profile(profile)
    # Aktualizujemy konfigurację środowiska, aby dalsze komponenty znały aktywny profil.
    try:
        environment.risk_profile = selected_profile
    except Exception:  # pragma: no cover - defensywnie w razie zmian modelu
        _LOGGER.debug("Nie można nadpisać risk_profile w konfiguracji środowiska", exc_info=True)

    credentials = secret_manager.load_exchange_credentials(
        environment.keychain_key,
        expected_environment=environment.environment,
        purpose=environment.credential_purpose,
        required_permissions=environment.required_permissions,
        forbidden_permissions=environment.forbidden_permissions,
    )

    factories = dict(_DEFAULT_ADAPTERS)
    env_factories = getattr(environment, "adapter_factories", None)
    if env_factories:
        _apply_adapter_factory_specs(
            factories,
            env_factories,
            source=f"konfiguracji środowiska '{environment.name}'",
            require_override=True,
        )
    if adapter_factories:
        _apply_adapter_factory_specs(
            factories,
            adapter_factories,
            source="parametrze 'adapter_factories'",
            require_override=False,
        )
    adapter = _instantiate_adapter(
        environment.exchange,
        credentials,
        factories,
        environment.environment,
        settings=environment.adapter_settings,
        offline_mode=offline_mode,
    )
    adapter.configure_network(ip_allowlist=environment.ip_allowlist or None)

    alert_channels, alert_router, audit_log = build_alert_channels(
        core_config=core_config,
        environment=environment,
        secret_manager=secret_manager,
    )

    decision_journal = _build_decision_journal(environment)
    portfolio_decision_log = _build_portfolio_decision_log(core_config, environment)

    # --- MetricsService (opcjonalny, kompatybilny z różnymi sygnaturami funkcji) ---
    metrics_server: Any | None = None
    metrics_sinks: list[Any] = []
    metrics_ui_alert_path: Path | None = None
    metrics_jsonl_path: Path | None = None
    metrics_ui_alert_sink_active = False
    metrics_service_enabled: bool | None = None
    metrics_ui_alerts_metadata: Mapping[str, Any] | None = None
    metrics_jsonl_metadata: Mapping[str, Any] | None = None
    metrics_ui_alerts_settings: Mapping[str, Any] | None = None
    metrics_risk_profiles_file: Mapping[str, Any] | None = None
    metrics_security_warnings: list[str] = []
    metrics_security_payload: dict[str, object] = {}
    metrics_auth_metadata: Mapping[str, Any] | None = None
    metrics_tls_enabled: bool | None = None
    metrics_token_validator: ServiceTokenValidator | None = None
    risk_server: Any | None = None
    risk_snapshot_store: Any | None = None
    risk_snapshot_builder: Any | None = None
    risk_snapshot_publisher: Any | None = None
    risk_metrics_exporter: Any | None = None
    risk_service_enabled: bool | None = None
    risk_security_payload: dict[str, object] = {}
    risk_security_warnings: list[str] = []
    risk_auth_metadata: Mapping[str, Any] | None = None
    risk_tls_enabled: bool | None = None
    risk_token_validator: ServiceTokenValidator | None = None

    if RiskSnapshotBuilder is not None:
        try:
            risk_snapshot_builder = RiskSnapshotBuilder(risk_engine)
        except Exception:  # pragma: no cover - nie blokujemy bootstrapu
            risk_snapshot_builder = None
            _LOGGER.debug("Nie udało się zainicjalizować RiskSnapshotBuilder", exc_info=True)
    if risk_snapshot_builder is not None and RiskMetricsExporter is not None:
        try:
            risk_metrics_exporter = RiskMetricsExporter(
                get_global_metrics_registry(),
                environment=environment.environment.value,
                stage=environment_name,
            )
            try:
                for profile_name in risk_snapshot_builder.profile_names():
                    snapshot = risk_snapshot_builder.build(profile_name)
                    if snapshot is not None:
                        risk_metrics_exporter(snapshot)
            except Exception:  # pragma: no cover - diagnostyka snapshotów startowych
                _LOGGER.debug(
                    "Nie udało się wygenerować początkowych metryk ryzyka", exc_info=True
                )
        except Exception:  # pragma: no cover - eksporter jest opcjonalny
            risk_metrics_exporter = None
            _LOGGER.debug("Nie udało się zainicjalizować eksportera metryk ryzyka", exc_info=True)
    metrics_config = getattr(core_config, "metrics_service", None)
    if metrics_config is not None:
        metrics_service_enabled = bool(getattr(metrics_config, "enabled", True))
        if offline_mode and metrics_service_enabled:
            _LOGGER.info(
                "Tryb offline: pomijam uruchomienie MetricsService w środowisku %s.",
                environment.name,
            )
            metrics_service_enabled = False
        tls_config = getattr(metrics_config, "tls", None)
        if tls_config is not None:
            from bot_core.security import tls_audit as _tls_audit

            tls_report = _tls_audit.audit_tls_entry(
                tls_config,
                role_prefix="metrics_tls",
                env=os.environ,
            )
            metrics_tls_enabled = bool(tls_report.get("enabled"))
            metrics_security_payload["tls"] = tls_report
            if tls_report.get("warnings"):
                metrics_security_warnings.extend(
                    str(item) for item in tls_report.get("warnings", ())
                )
            if tls_report.get("errors"):
                metrics_security_warnings.extend(
                    str(item) for item in tls_report.get("errors", ())
                )
        else:
            metrics_tls_enabled = False
        token_value = _config_value(metrics_config, "auth_token", "token")
        token_str = str(token_value).strip() if token_value else ""
        token_env_name = _config_value(metrics_config, "auth_token_env")
        token_env_present = bool(token_env_name and os.environ.get(str(token_env_name)))
        token_file_entry = _config_value(metrics_config, "auth_token_file")
        token_file_exists = False
        token_file_permissions: str | None = None
        token_file_over_permissive = False
        if token_file_entry:
            try:
                token_file_path = Path(str(token_file_entry)).expanduser()
            except (OSError, TypeError, ValueError):
                token_file_path = None
            else:
                try:
                    token_file_exists = token_file_path.is_file()
                except OSError:
                    token_file_exists = False
                if token_file_exists:
                    try:
                        file_mode = stat.S_IMODE(token_file_path.stat().st_mode)
                        token_file_permissions = format(file_mode, "#04o")
                        if os.name != "nt" and file_mode & 0o077:
                            token_file_over_permissive = True
                    except OSError:
                        token_file_permissions = None
        rbac_entries = tuple(getattr(metrics_config, "rbac_tokens", ()) or ())
        if rbac_entries:
            try:
                metrics_token_validator = build_service_token_validator(
                    rbac_entries,
                    default_scope="metrics.read",
                )
            except Exception:  # pragma: no cover - diagnostyka konfiguracji
                _LOGGER.exception("Nie udało się zbudować walidatora RBAC dla MetricsService")
                metrics_token_validator = None
                metrics_security_warnings.append(
                    "Nie udało się zainicjalizować walidatora RBAC MetricsService"
                )
            else:
                metrics_security_payload["rbac_tokens"] = metrics_token_validator.metadata()
        token_configured = bool(token_str) or token_env_present or token_file_exists or bool(metrics_token_validator)
        metrics_auth_metadata = {
            "token_configured": token_configured,
        }
        if token_str:
            metrics_auth_metadata["token_length"] = len(token_str)
        if token_env_name:
            metrics_auth_metadata["token_env"] = str(token_env_name)
            metrics_auth_metadata["token_env_present"] = token_env_present
            if not token_env_present:
                metrics_security_warnings.append(
                    (
                        "Zmienna środowiskowa tokenu MetricsService nie jest ustawiona "
                        "(runtime.metrics_service.auth_token_env)."
                    )
                )
        if token_file_entry:
            metrics_auth_metadata["token_file"] = str(token_file_entry)
            metrics_auth_metadata["token_file_exists"] = token_file_exists
            if token_file_permissions:
                metrics_auth_metadata["token_file_permissions"] = token_file_permissions
            if not token_file_exists:
                metrics_security_warnings.append(
                    (
                        "Wskazany plik legacy tokenu MetricsService nie istnieje "
                        "(runtime.metrics_service.auth_token_file)."
                    )
                )
            elif token_file_over_permissive:
                metrics_security_warnings.append(
                    (
                        "Plik tokenu MetricsService ma zbyt szerokie uprawnienia "
                        f"({token_file_permissions or '<unknown>'}); ustaw chmod 600."
                    )
                )
        if metrics_token_validator is not None:
            validator_meta = metrics_token_validator.metadata()
            metrics_auth_metadata["rbac_tokens"] = validator_meta.get("tokens", [])
            metrics_auth_metadata["default_scope"] = validator_meta.get("default_scope")
        metrics_security_payload["auth"] = metrics_auth_metadata
        profiles_file_value = getattr(metrics_config, "ui_alerts_risk_profiles_file", None)
        if profiles_file_value:
            normalized_file = Path(profiles_file_value).expanduser()
            try:
                normalized_file = normalized_file.resolve(strict=False)
            except Exception:  # pragma: no cover - diagnostyka pomocnicza
                normalized_file = normalized_file.absolute()
            if load_risk_profiles_with_metadata is None:  # type: ignore[truthy-bool]
                metrics_risk_profiles_file = {
                    "path": str(normalized_file),
                    "warning": "risk_profile_loader_unavailable",
                }
            else:
                try:
                    _, metrics_risk_profiles_file = load_risk_profiles_with_metadata(  # type: ignore[misc]
                        str(normalized_file),
                        origin_label=f"metrics_service_config:{normalized_file}",
                    )
                except Exception:  # pragma: no cover - diagnostyka konfiguracji
                    _LOGGER.exception(
                        "Nie udało się wczytać profili ryzyka telemetrii z %s", normalized_file
                    )
                    raise
        jsonl_candidate = getattr(metrics_config, "jsonl_path", None)
        if jsonl_candidate:
            try:
                metrics_jsonl_path = Path(jsonl_candidate).expanduser()
            except Exception:  # pragma: no cover - diagnostyka pomocnicza
                _LOGGER.debug(
                    "Nie udało się znormalizować ścieżki JSONL telemetrii", exc_info=True
                )
                metrics_jsonl_path = Path(str(jsonl_candidate))
            try:
                metrics_jsonl_metadata = file_reference_metadata(
                    metrics_jsonl_path, role="jsonl"
                )
            except Exception:  # pragma: no cover - diagnostyka pomocnicza
                _LOGGER.debug(
                    "Nie udało się zebrać metadanych JSONL telemetrii", exc_info=True
                )
            else:
                metrics_security_payload["metrics_jsonl"] = metrics_jsonl_metadata
    if UiTelemetryAlertSink is not None:
        try:
            base_dir_value = getattr(core_config, "source_directory", None)
            base_dir: Path | None = None
            if base_dir_value:
                try:
                    base_dir = Path(base_dir_value).expanduser()
                except Exception:  # pragma: no cover - diagnostyka pomocnicza
                    _LOGGER.debug(
                        "Nie udało się znormalizować katalogu konfiguracji: %s", base_dir_value,
                        exc_info=True,
                    )
            default_path = DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
            if not default_path.is_absolute():
                try:
                    default_path = default_path.resolve(strict=False)
                except Exception:  # pragma: no cover - zachowujemy przybliżenie
                    default_path = default_path.absolute()

            configured_path = None
            if metrics_config and metrics_config.ui_alerts_jsonl_path:
                configured_path = Path(metrics_config.ui_alerts_jsonl_path).expanduser()
                if base_dir is not None and not configured_path.is_absolute():
                    try:
                        configured_path = (base_dir / configured_path).resolve(strict=False)
                    except Exception:  # pragma: no cover - zachowujemy przybliżenie
                        configured_path = (base_dir / configured_path).absolute()
            telemetry_log = configured_path or default_path

            # --- Risk profile resolver (opcjonalny) ---
            risk_profile_meta: Mapping[str, Any] | None = None
            resolver: "MetricsRiskProfileResolver" | None = None
            if metrics_config is not None and getattr(metrics_config, "ui_alerts_risk_profile", None):
                normalized_profile = str(metrics_config.ui_alerts_risk_profile).strip().lower()
                if MetricsRiskProfileResolver is None:
                    risk_profile_meta = {"name": normalized_profile, "warning": "resolver_unavailable"}
                else:
                    try:
                        resolver = MetricsRiskProfileResolver(normalized_profile, metrics_config)
                    except KeyError:
                        risk_profile_meta = {"name": normalized_profile, "error": "unknown_profile"}
                        _LOGGER.warning(
                            "Nieznany profil ryzyka telemetrii UI w bootstrapie: %s", normalized_profile
                        )
                    except Exception:  # pragma: no cover - diagnostyka
                        risk_profile_meta = {"name": normalized_profile}
                        _LOGGER.exception(
                            "Nie udało się zastosować profilu ryzyka %s w bootstrapie", normalized_profile
                        )

            def _resolve_metrics_value(field_name: str, default: Any) -> Any:
                if metrics_config is None:
                    return default
                value = getattr(metrics_config, field_name, default)
                if resolver is not None:
                    value = resolver.override(field_name, value)
                return value

            # Tryby zdarzeń
            reduce_mode = "enable"
            overlay_mode = "enable"
            jank_mode = "disable"
            performance_mode = "disable"

            reduce_candidate = _resolve_metrics_value("reduce_motion_mode", None)
            if reduce_candidate is not None:
                candidate = str(reduce_candidate).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    reduce_mode = candidate
            elif metrics_config is not None:
                reduce_mode = (
                    "enable" if bool(_resolve_metrics_value("reduce_motion_alerts", True)) else "disable"
                )

            overlay_candidate = _resolve_metrics_value("overlay_alert_mode", None)
            if overlay_candidate is not None:
                candidate = str(overlay_candidate).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    overlay_mode = candidate
            elif metrics_config is not None:
                overlay_mode = (
                    "enable" if bool(_resolve_metrics_value("overlay_alerts", True)) else "disable"
                )

            jank_candidate = _resolve_metrics_value("jank_alert_mode", None)
            if jank_candidate is not None:
                candidate = str(jank_candidate).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    jank_mode = candidate
            elif metrics_config is not None:
                jank_mode = (
                    "enable" if bool(_resolve_metrics_value("jank_alerts", False)) else "disable"
                )

            performance_candidate = _resolve_metrics_value(
                "performance_alert_mode", None
            )
            if performance_candidate is not None:
                candidate = str(performance_candidate).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    performance_mode = candidate
            elif metrics_config is not None:
                performance_mode = (
                    "enable"
                    if bool(_resolve_metrics_value("performance_alerts", False))
                    else "disable"
                )

            reduce_dispatch = reduce_mode == "enable"
            overlay_dispatch = overlay_mode == "enable"
            jank_dispatch = jank_mode == "enable"
            performance_dispatch = performance_mode == "enable"
            reduce_logging = reduce_mode in {"enable", "jsonl"}
            overlay_logging = overlay_mode in {"enable", "jsonl"}
            jank_logging = jank_mode in {"enable", "jsonl"}
            performance_logging = performance_mode in {"enable", "jsonl"}

            # Kategorie / severity / progi (z możliwością override przez resolver)
            reduce_category = _resolve_metrics_value("reduce_motion_category", "ui.performance")
            reduce_active = _resolve_metrics_value("reduce_motion_severity_active", "warning")
            reduce_recovered = _resolve_metrics_value("reduce_motion_severity_recovered", "info")

            overlay_category = _resolve_metrics_value("overlay_alert_category", "ui.performance")
            overlay_exceeded = _resolve_metrics_value("overlay_alert_severity_exceeded", "warning")
            overlay_recovered = _resolve_metrics_value("overlay_alert_severity_recovered", "info")
            overlay_critical = _resolve_metrics_value("overlay_alert_severity_critical", "critical")
            overlay_threshold_raw = _resolve_metrics_value("overlay_alert_critical_threshold", 2)

            jank_category = _resolve_metrics_value("jank_alert_category", "ui.performance")
            jank_spike = _resolve_metrics_value("jank_alert_severity_spike", "warning")
            jank_critical = _resolve_metrics_value("jank_alert_severity_critical", None)
            jank_threshold_raw = _resolve_metrics_value("jank_alert_critical_over_ms", None)

            performance_category = _resolve_metrics_value(
                "performance_category", "ui.performance"
            )
            performance_warning = _resolve_metrics_value(
                "performance_severity_warning", "warning"
            )
            performance_critical = _resolve_metrics_value(
                "performance_severity_critical", "critical"
            )
            performance_recovered = _resolve_metrics_value(
                "performance_severity_recovered", "info"
            )
            performance_event_warning_raw = _resolve_metrics_value(
                "performance_event_to_frame_warning_ms", 45.0
            )
            performance_event_critical_raw = _resolve_metrics_value(
                "performance_event_to_frame_critical_ms", 60.0
            )
            cpu_warning_raw = _resolve_metrics_value(
                "cpu_utilization_warning_percent", 85.0
            )
            cpu_critical_raw = _resolve_metrics_value(
                "cpu_utilization_critical_percent", 95.0
            )
            gpu_warning_raw = _resolve_metrics_value(
                "gpu_utilization_warning_percent", None
            )
            gpu_critical_raw = _resolve_metrics_value(
                "gpu_utilization_critical_percent", None
            )
            ram_warning_raw = _resolve_metrics_value(
                "ram_usage_warning_megabytes", None
            )
            ram_critical_raw = _resolve_metrics_value(
                "ram_usage_critical_megabytes", None
            )

            sink_kwargs: dict[str, object] = {
                "jsonl_path": telemetry_log,
                "enable_reduce_motion_alerts": reduce_dispatch,
                "enable_overlay_alerts": overlay_dispatch,
                "log_reduce_motion_events": reduce_logging,
                "log_overlay_events": overlay_logging,
                "enable_jank_alerts": jank_dispatch,
                "log_jank_events": jank_logging,
                "enable_performance_alerts": performance_dispatch,
                "log_performance_events": performance_logging,
                "reduce_motion_category": reduce_category,
                "reduce_motion_severity_active": reduce_active,
                "reduce_motion_severity_recovered": reduce_recovered,
                "overlay_category": overlay_category,
                "overlay_severity_exceeded": overlay_exceeded,
                "overlay_severity_recovered": overlay_recovered,
                "jank_category": jank_category,
                "jank_severity_spike": jank_spike,
                "performance_category": performance_category,
                "performance_severity_warning": performance_warning,
                "performance_severity_critical": performance_critical,
                "performance_severity_recovered": performance_recovered,
            }

            # Walidacja/projekcja progów
            overlay_threshold_value: int | None = None
            if overlay_threshold_raw is not None:
                try:
                    overlay_threshold_value = int(overlay_threshold_raw)
                except (TypeError, ValueError):
                    _LOGGER.debug(
                        "Nieprawidłowy próg overlay_alert_critical_threshold=%s", overlay_threshold_raw
                    )
                else:
                    sink_kwargs["overlay_critical_threshold"] = overlay_threshold_value

            jank_threshold_value: float | None = None
            if jank_threshold_raw is not None:
                try:
                    jank_threshold_value = float(jank_threshold_raw)
                except (TypeError, ValueError):
                    _LOGGER.debug(
                        "Nieprawidłowy próg jank_alert_critical_over_ms=%s", jank_threshold_raw
                    )
                else:
                    sink_kwargs["jank_critical_over_ms"] = jank_threshold_value

            def _normalize_optional_float(value: object, *, field_name: str) -> float | None:
                if value in (None, ""):
                    return None
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    _LOGGER.debug("Nieprawidłowy próg %s=%s", field_name, value)
                    return None
                return numeric

            performance_event_warning = _normalize_optional_float(
                performance_event_warning_raw,
                field_name="performance_event_to_frame_warning_ms",
            )
            performance_event_critical = _normalize_optional_float(
                performance_event_critical_raw,
                field_name="performance_event_to_frame_critical_ms",
            )
            cpu_warning_percent = _normalize_optional_float(
                cpu_warning_raw, field_name="cpu_utilization_warning_percent"
            )
            cpu_critical_percent = _normalize_optional_float(
                cpu_critical_raw, field_name="cpu_utilization_critical_percent"
            )
            gpu_warning_percent = _normalize_optional_float(
                gpu_warning_raw, field_name="gpu_utilization_warning_percent"
            )
            gpu_critical_percent = _normalize_optional_float(
                gpu_critical_raw, field_name="gpu_utilization_critical_percent"
            )
            ram_warning_megabytes = _normalize_optional_float(
                ram_warning_raw, field_name="ram_usage_warning_megabytes"
            )
            ram_critical_megabytes = _normalize_optional_float(
                ram_critical_raw, field_name="ram_usage_critical_megabytes"
            )

            sink_kwargs["performance_event_to_frame_warning_ms"] = (
                performance_event_warning
            )
            sink_kwargs["performance_event_to_frame_critical_ms"] = (
                performance_event_critical
            )
            sink_kwargs["cpu_utilization_warning_percent"] = cpu_warning_percent
            sink_kwargs["cpu_utilization_critical_percent"] = cpu_critical_percent
            sink_kwargs["gpu_utilization_warning_percent"] = gpu_warning_percent
            sink_kwargs["gpu_utilization_critical_percent"] = gpu_critical_percent
            sink_kwargs["ram_usage_warning_megabytes"] = ram_warning_megabytes
            sink_kwargs["ram_usage_critical_megabytes"] = ram_critical_megabytes

            if overlay_critical is not None:
                sink_kwargs["overlay_severity_critical"] = overlay_critical
            if jank_critical is not None:
                sink_kwargs["jank_severity_critical"] = jank_critical

            settings_payload: dict[str, object] = {
                "jsonl_path": str(telemetry_log),
                "reduce_mode": reduce_mode,
                "overlay_mode": overlay_mode,
                "jank_mode": jank_mode,
                "performance_mode": performance_mode,
                "reduce_motion_alerts": reduce_dispatch,
                "overlay_alerts": overlay_dispatch,
                "jank_alerts": jank_dispatch,
                "performance_alerts": performance_dispatch,
                "reduce_motion_logging": reduce_logging,
                "overlay_logging": overlay_logging,
                "jank_logging": jank_logging,
                "performance_logging": performance_logging,
                "reduce_motion_category": reduce_category,
                "reduce_motion_severity_active": reduce_active,
                "reduce_motion_severity_recovered": reduce_recovered,
                "overlay_category": overlay_category,
                "overlay_severity_exceeded": overlay_exceeded,
                "overlay_severity_recovered": overlay_recovered,
                "overlay_severity_critical": overlay_critical,
                "overlay_critical_threshold": overlay_threshold_value,
                "jank_category": jank_category,
                "jank_severity_spike": jank_spike,
                "jank_severity_critical": jank_critical,
                "jank_critical_over_ms": jank_threshold_value,
                "performance_category": performance_category,
                "performance_severity_warning": performance_warning,
                "performance_severity_critical": performance_critical,
                "performance_severity_recovered": performance_recovered,
                "performance_event_to_frame_warning_ms": performance_event_warning,
                "performance_event_to_frame_critical_ms": performance_event_critical,
                "cpu_utilization_warning_percent": cpu_warning_percent,
                "cpu_utilization_critical_percent": cpu_critical_percent,
                "gpu_utilization_warning_percent": gpu_warning_percent,
                "gpu_utilization_critical_percent": gpu_critical_percent,
                "ram_usage_warning_megabytes": ram_warning_megabytes,
                "ram_usage_critical_megabytes": ram_critical_megabytes,
            }
            if resolver is not None:
                risk_profile_meta = resolver.metadata()

            summary_payload: Mapping[str, Any] | None = None
            if risk_profile_meta is not None:
                sink_kwargs["risk_profile"] = dict(risk_profile_meta)
                settings_payload["risk_profile"] = dict(risk_profile_meta)
                summary_payload = risk_profile_meta.get("summary")
                if summary_payload is None and summarize_risk_profile is not None:
                    try:
                        summary_payload = summarize_risk_profile(risk_profile_meta)
                    except Exception:  # pragma: no cover - defensywne
                        summary_payload = None
            if summary_payload:
                sink_kwargs["risk_profile_summary"] = dict(summary_payload)
                settings_payload["risk_profile_summary"] = dict(summary_payload)
            if metrics_risk_profiles_file is not None:
                settings_payload["risk_profiles_file"] = dict(metrics_risk_profiles_file)

            audit_backend = _build_ui_alert_audit_metadata(
                alert_router,
                requested_backend=getattr(metrics_config, "ui_alerts_audit_backend", None),
            )
            settings_payload["audit"] = audit_backend

            metrics_sinks.append(UiTelemetryAlertSink(alert_router, **sink_kwargs))
            metrics_ui_alert_path = telemetry_log
            metrics_ui_alert_sink_active = True
            metrics_ui_alerts_settings = settings_payload
            try:
                metrics_ui_alerts_metadata = file_reference_metadata(
                    telemetry_log, role="ui_alerts_jsonl"
                )
            except Exception:  # pragma: no cover - diagnostyka pomocnicza
                _LOGGER.debug(
                    "Nie udało się zebrać metadanych logu alertów UI", exc_info=True
                )
            else:
                metrics_security_payload["metrics_ui_alerts"] = metrics_ui_alerts_metadata
        except Exception:  # pragma: no cover - nie blokujemy startu runtime
            _LOGGER.exception("Nie udało się zainicjalizować UiTelemetryAlertSink")

        # Jeśli tylko profil/plik profili – również pokaż w settings
        if metrics_ui_alerts_settings is None and risk_profile_meta is not None:
            metrics_ui_alerts_settings = {"risk_profile": dict(risk_profile_meta)}
            if risk_profile_meta.get("summary"):
                metrics_ui_alerts_settings["risk_profile_summary"] = dict(risk_profile_meta["summary"])  # type: ignore[index]

        if metrics_risk_profiles_file is not None:
            if metrics_ui_alerts_settings is None:
                metrics_ui_alerts_settings = {
                    "risk_profiles_file": dict(metrics_risk_profiles_file)
                }
            else:
                metrics_ui_alerts_settings = dict(metrics_ui_alerts_settings)
                metrics_ui_alerts_settings["risk_profiles_file"] = dict(
                    metrics_risk_profiles_file
                )

    if metrics_security_payload:
        warnings_detected = log_security_warnings(
            metrics_security_payload,
            fail_on_warnings=False,
            logger=_LOGGER,
            context="runtime.bootstrap",
        )
        if warnings_detected:
            for entry in collect_security_warnings(metrics_security_payload):
                metrics_security_warnings.extend(
                    str(item) for item in entry.get("warnings", [])
                )

    if metrics_service_enabled:
        if metrics_auth_metadata and not metrics_auth_metadata.get("token_configured"):
            metrics_security_warnings.append(
                (
                    "MetricsService ma włączone API bez tokenu autoryzacyjnego – "
                    "ustaw runtime.metrics_service.auth_token_env lub skonfiguruj RBAC."
                )
            )
        if metrics_tls_enabled is False:
            metrics_security_warnings.append(
                "MetricsService działa bez TLS – włącz runtime.metrics_service.tls.enabled lub dostarcz certyfikaty."
            )

    if metrics_security_warnings:
        metrics_security_warnings = list(dict.fromkeys(metrics_security_warnings))

    if (
        build_metrics_server_from_config is not None
        and metrics_service_enabled
        and not offline_mode
    ):
        try:
            # Najpierw spróbuj najnowszej sygnatury (cfg, sinks, alerts_router, token_validator)
            try:
                metrics_server = build_metrics_server_from_config(  # type: ignore[call-arg]
                    core_config.metrics_service,
                    sinks=metrics_sinks or None,
                    alerts_router=alert_router,
                    token_validator=metrics_token_validator,
                )
            except TypeError:
                # Następnie (cfg, alerts_router, token_validator)
                try:
                    metrics_server = build_metrics_server_from_config(  # type: ignore[call-arg]
                        core_config.metrics_service,
                        alerts_router=alert_router,
                        token_validator=metrics_token_validator,
                    )
                except TypeError:
                    # Potem (cfg, sinks, token_validator)
                    try:
                        metrics_server = build_metrics_server_from_config(  # type: ignore[call-arg]
                            core_config.metrics_service,
                            sinks=metrics_sinks or None,
                            token_validator=metrics_token_validator,
                        )
                    except TypeError:
                        # Kolejny fallback (cfg, token_validator)
                        try:
                            metrics_server = build_metrics_server_from_config(
                                core_config.metrics_service,
                                token_validator=metrics_token_validator,
                            )
                        except TypeError:
                            # Najstarsza postać: tylko (cfg)
                            metrics_server = build_metrics_server_from_config(
                                core_config.metrics_service  # type: ignore[arg-type]
                            )

            if metrics_server is not None:
                metrics_server.start()
                _LOGGER.info(
                    "Serwer MetricsService uruchomiony na %s",
                    getattr(metrics_server, "address", "unknown"),
                )
        except Exception:  # pragma: no cover - telemetria jest opcjonalna
            _LOGGER.exception("Nie udało się uruchomić MetricsService – kontynuuję bez telemetrii")
            metrics_server = None

    risk_config = getattr(core_config, "risk_service", None)
    if risk_config is not None:
        risk_service_enabled = bool(getattr(risk_config, "enabled", True))
        if offline_mode and risk_service_enabled:
            _LOGGER.info(
                "Tryb offline: pomijam uruchomienie RiskService w środowisku %s.",
                environment.name,
            )
            risk_service_enabled = False
        tls_config = getattr(risk_config, "tls", None)
        if tls_config is not None:
            from bot_core.security import tls_audit as _tls_audit

            tls_report = _tls_audit.audit_tls_entry(
                tls_config,
                role_prefix="risk_tls",
                env=os.environ,
            )
            risk_tls_enabled = bool(tls_report.get("enabled"))
            risk_security_payload["tls"] = tls_report
            if tls_report.get("warnings"):
                risk_security_warnings.extend(
                    str(item) for item in tls_report.get("warnings", ())
                )
            if tls_report.get("errors"):
                risk_security_warnings.extend(
                    str(item) for item in tls_report.get("errors", ())
                )
        else:
            risk_tls_enabled = False
        token_value = _config_value(risk_config, "auth_token", "token")
        token_str = str(token_value).strip() if token_value else ""
        rbac_entries = tuple(getattr(risk_config, "rbac_tokens", ()) or ())
        if rbac_entries:
            try:
                risk_token_validator = build_service_token_validator(
                    rbac_entries,
                    default_scope="risk.read",
                )
            except Exception:  # pragma: no cover - diagnostyka konfiguracji
                _LOGGER.exception("Nie udało się zbudować walidatora RBAC dla RiskService")
                risk_token_validator = None
                risk_security_warnings.append(
                    "Nie udało się zainicjalizować walidatora RBAC RiskService"
                )
            else:
                risk_security_payload["rbac_tokens"] = risk_token_validator.metadata()
        risk_auth_metadata = {
            "token_configured": bool(token_str) or bool(risk_token_validator),
        }
        if token_str:
            risk_auth_metadata["token_length"] = len(token_str)
        if risk_token_validator is not None:
            validator_meta = risk_token_validator.metadata()
            risk_auth_metadata["rbac_tokens"] = validator_meta.get("tokens", [])
            risk_auth_metadata["default_scope"] = validator_meta.get("default_scope")
        risk_security_payload["auth"] = risk_auth_metadata

    if (
        build_risk_server_from_config is not None
        and risk_config is not None
        and risk_service_enabled
        and not offline_mode
    ):
        try:
            try:
                candidate = build_risk_server_from_config(
                    risk_config,
                    token_validator=risk_token_validator,
                )
            except TypeError:
                candidate = build_risk_server_from_config(risk_config)
        except Exception:  # pragma: no cover - risk service jest opcjonalny
            _LOGGER.exception("Nie udało się zbudować konfiguracji RiskService")
        else:
            if candidate is not None:
                try:
                    candidate.start()
                except Exception:  # pragma: no cover - brak krytyczny, kontynuujemy bez serwera
                    _LOGGER.exception("Nie udało się uruchomić RiskService – kontynuuję bez serwera ryzyka")
                else:
                    risk_server = candidate
                    risk_snapshot_store = getattr(candidate, "store", None)
                    _LOGGER.info(
                        "Serwer RiskService uruchomiony na %s",
                        getattr(candidate, "address", "unknown"),
                    )
                    if risk_snapshot_builder is not None:
                        try:
                            initial_snapshot = risk_snapshot_builder.build(profile.name)
                            if initial_snapshot is not None:
                                candidate.publish(initial_snapshot)
                                if risk_metrics_exporter is not None:
                                    try:
                                        risk_metrics_exporter(initial_snapshot)
                                    except Exception:  # pragma: no cover - diagnostyka eksportera
                                        _LOGGER.debug(
                                            "Nie udało się zaktualizować metryk ryzyka początkowym snapshotem",
                                            exc_info=True,
                                        )
                        except Exception:  # pragma: no cover - diagnostyka
                            _LOGGER.debug(
                                "Nie udało się opublikować początkowego stanu ryzyka",
                                exc_info=True,
                            )
                        if RiskSnapshotPublisher is not None:
                            publish_interval = 5.0
                            normalized_profiles: tuple[str, ...] = ()
                            try:
                                interval_raw = getattr(risk_config, "publish_interval_seconds", 5.0)
                                publish_interval = float(interval_raw)
                            except (TypeError, ValueError):
                                _LOGGER.debug(
                                    "Nieprawidłowy interwał publish_interval_seconds – używam domyślnego 5s",
                                    exc_info=True,
                                )
                                publish_interval = 5.0
                            raw_profiles = getattr(risk_config, "profiles", ()) or ()
                            normalized_profiles = tuple(
                                dict.fromkeys(
                                    str(name).strip()
                                    for name in raw_profiles
                                    if str(name).strip()
                                )
                            )
                            try:
                                sinks = [candidate.publish]
                                if risk_metrics_exporter is not None:
                                    sinks.append(risk_metrics_exporter)
                                publisher = RiskSnapshotPublisher(
                                    risk_snapshot_builder,
                                    profiles=normalized_profiles or None,
                                    interval_seconds=publish_interval,
                                    sinks=sinks,
                                )
                                publisher.start()
                            except Exception:  # pragma: no cover - diagnostyka publikatora
                                _LOGGER.exception("Nie udało się uruchomić RiskSnapshotPublisher")
                            else:
                                risk_snapshot_publisher = publisher
                    _LOGGER.info(
                        "RiskSnapshotPublisher aktywny (profili=%s, interwał=%.2fs)",
                        ",".join(normalized_profiles) if normalized_profiles else "auto",
                        publish_interval,
                    )

    if reset_risk_profile_store is not None:
        try:
            reset_risk_profile_store()
        except Exception:  # pragma: no cover - defensywne czyszczenie
            _LOGGER.debug("Nie udało się zresetować presetów profili ryzyka", exc_info=True)

    if risk_security_payload:
        warnings_detected = log_security_warnings(
            risk_security_payload,
            fail_on_warnings=False,
            logger=_LOGGER,
            context="risk_service",
        )
        if warnings_detected:
            for entry in collect_security_warnings(risk_security_payload):
                risk_security_warnings.extend(
                    str(item) for item in entry.get("warnings", [])
                )

    if risk_service_enabled:
        if risk_auth_metadata and not risk_auth_metadata.get("token_configured"):
            risk_security_warnings.append(
                "RiskService ma włączone API bez tokenu autoryzacyjnego – ustaw runtime.risk_service.auth_token."
            )
        if risk_tls_enabled is False:
            risk_security_warnings.append(
                "RiskService działa bez TLS – włącz runtime.risk_service.tls.enabled lub dostarcz certyfikaty."
            )

    if risk_security_warnings:
        risk_security_warnings = list(dict.fromkeys(risk_security_warnings))

    return BootstrapContext(
        core_config=core_config,
        environment=environment,
        credentials=credentials,
        adapter=adapter,
        risk_engine=risk_engine,
        risk_repository=risk_repository,
        alert_router=alert_router,
        alert_channels=alert_channels,
        audit_log=audit_log,
        adapter_settings=environment.adapter_settings,
        decision_journal=decision_journal,
        risk_decision_log=risk_decision_log,
        risk_profile_name=selected_profile,
        license_validation=license_result,
        decision_engine_config=decision_engine_config,
        decision_orchestrator=decision_orchestrator,
        decision_tco_report_path=decision_tco_report_path,
        decision_tco_warnings=tuple(decision_tco_warnings)
        if decision_tco_warnings
        else None,
        tco_reporter=tco_reporter,
        portfolio_governor_config=portfolio_governor_config,
        portfolio_governor=portfolio_governor,
        metrics_server=metrics_server,
        metrics_ui_alerts_path=metrics_ui_alert_path,
        metrics_jsonl_path=metrics_jsonl_path,
        metrics_ui_alert_sink_active=metrics_ui_alert_sink_active,
        metrics_service_enabled=metrics_service_enabled,
        metrics_ui_alerts_metadata=metrics_ui_alerts_metadata,
        metrics_jsonl_metadata=metrics_jsonl_metadata,
        metrics_security_warnings=tuple(metrics_security_warnings)
        if metrics_security_warnings
        else None,
        metrics_security_metadata=metrics_security_payload if metrics_security_payload else None,
        metrics_ui_alerts_settings=metrics_ui_alerts_settings,
        metrics_token_validator=metrics_token_validator,
        risk_server=risk_server,
        risk_snapshot_store=risk_snapshot_store,
        risk_snapshot_builder=risk_snapshot_builder,
        risk_snapshot_publisher=risk_snapshot_publisher,
        risk_service_enabled=risk_service_enabled,
        risk_security_metadata=risk_security_payload if risk_security_payload else None,
        risk_security_warnings=tuple(risk_security_warnings) if risk_security_warnings else None,
        risk_token_validator=risk_token_validator,
        portfolio_decision_log=portfolio_decision_log,
        ai_manager=ai_manager_instance,
        ai_models_loaded=tuple(ai_models_loaded) if ai_models_loaded else None,
        ai_threshold_bps=ai_threshold_bps,
        ai_model_bindings=ai_model_bindings,
        ai_ensembles_registered=tuple(ai_ensembles_registered)
        if ai_ensembles_registered
        else None,
        ai_pipeline_schedules=tuple(ai_pipeline_schedules)
        if ai_pipeline_schedules
        else None,
        ai_pipeline_pending=tuple(ai_pipeline_pending) if ai_pipeline_pending else None,
    )


def _instantiate_adapter(
    exchange_name: str,
    credentials: ExchangeCredentials,
    factories: Mapping[str, ExchangeAdapterFactory],
    environment: Environment,
    *,
    settings: Mapping[str, Any] | None = None,
    offline_mode: bool = False,
) -> ExchangeAdapter:
    try:
        factory = factories[exchange_name]
    except KeyError as exc:
        raise KeyError(f"Brak fabryki adaptera dla giełdy '{exchange_name}'") from exc
    try:
        if settings:
            return factory(credentials, environment=environment, settings=settings)
        return factory(credentials, environment=environment)
    except RuntimeError as exc:
        if offline_mode and "ccxt" in str(exc).lower():
            from bot_core.exchanges.ccxt_adapter import (  # noqa: WPS433 - import lokalny
                CCXTExchange,
            )

            dummy_client = CCXTExchange()  # type: ignore[call-arg]
            if not hasattr(dummy_client, "symbols"):
                dummy_client.symbols = []  # type: ignore[attr-defined]
            try:
                if settings:
                    return factory(
                        credentials,
                        environment=environment,
                        settings=settings,
                        client=dummy_client,
                    )
                return factory(
                    credentials,
                    environment=environment,
                    client=dummy_client,
                )
            except TypeError:
                pass
        raise


def _build_risk_decision_log(
    core_config: CoreConfig, environment: EnvironmentConfig
) -> RiskDecisionLog | None:
    config = getattr(core_config, "risk_decision_log", None)
    if config is None or not getattr(config, "enabled", True):
        return None

    configured_path = getattr(config, "path", None)
    if configured_path:
        candidate = Path(str(configured_path)).expanduser()
        if not candidate.is_absolute():
            log_path = Path(environment.data_cache_path) / candidate
        else:
            log_path = candidate
    else:
        log_path = Path(environment.data_cache_path) / "risk_decisions.jsonl"

    max_entries = int(getattr(config, "max_entries", 1_000) or 1_000)
    signing_key = _load_risk_decision_log_key(config)
    signing_key_id = getattr(config, "signing_key_id", None)
    jsonl_fsync = bool(getattr(config, "jsonl_fsync", False))

    try:
        return RiskDecisionLog(
            max_entries=max_entries,
            jsonl_path=log_path,
            signing_key=signing_key,
            signing_key_id=str(signing_key_id) if signing_key_id else None,
            jsonl_fsync=jsonl_fsync,
        )
    except Exception:  # pragma: no cover - log jest opcjonalny
        _LOGGER.exception("Nie udało się zainicjalizować RiskDecisionLog")
        return None


def _load_risk_decision_log_key(config: object) -> bytes | None:
    env_name = getattr(config, "signing_key_env", None)
    if env_name:
        env_value = os.environ.get(str(env_name))
        if env_value:
            return env_value.encode("utf-8")
        _LOGGER.warning(
            "RiskDecisionLog: zmienna środowiskowa %s nie jest ustawiona", env_name
        )

    key_path = getattr(config, "signing_key_path", None)
    if key_path:
        try:
            content = Path(str(key_path)).expanduser().read_bytes()
            stripped = content.strip()
            if stripped:
                return stripped
            _LOGGER.warning("RiskDecisionLog: plik %s nie zawiera klucza", key_path)
        except Exception as exc:  # pragma: no cover - diagnostyka konfiguracji
            _LOGGER.warning("RiskDecisionLog: błąd odczytu klucza z %s: %s", key_path, exc)

    key_value = getattr(config, "signing_key_value", None)
    if key_value not in (None, ""):
        return str(key_value).encode("utf-8")

    return None


def _resolve_risk_profile(
    profiles: Mapping[str, RiskProfileConfig],
    profile_name: str,
) -> RiskProfileConfig:
    try:
        return profiles[profile_name]
    except KeyError as exc:
        raise KeyError(f"Profil ryzyka '{profile_name}' nie istnieje w konfiguracji") from exc


def build_alert_channels(
    *,
    core_config: CoreConfig,
    environment: EnvironmentConfig,
    secret_manager: SecretManager,
) -> tuple[Mapping[str, AlertChannel], DefaultAlertRouter, AlertAuditLog]:
    """Tworzy i rejestruje kanały alertów + router + backend audytu."""
    components = _get_alert_components()
    FileAlertAuditLogCls = components["FileAlertAuditLog"]
    InMemoryAlertAuditLogCls = components["InMemoryAlertAuditLog"]
    AlertThrottleCls = components["AlertThrottle"]
    DefaultAlertRouterCls = components["DefaultAlertRouter"]

    audit_config = getattr(environment, "alert_audit", None)
    if audit_config and getattr(audit_config, "backend", "memory") == "file":
        directory = Path(audit_config.directory) if audit_config.directory else Path("alerts")
        if not directory.is_absolute():
            base = Path(environment.data_cache_path)
            directory = base / directory
        audit_log: AlertAuditLog = FileAlertAuditLogCls(
            directory=directory,
            filename_pattern=audit_config.filename_pattern,
            retention_days=audit_config.retention_days,
            fsync=audit_config.fsync,
        )
    else:
        audit_log = InMemoryAlertAuditLogCls()

    throttle_cfg = getattr(environment, "alert_throttle", None)
    throttle: AlertThrottle | None = None
    if throttle_cfg is not None:
        throttle = AlertThrottleCls(
            window=timedelta(seconds=float(throttle_cfg.window_seconds)),
            exclude_severities=frozenset(throttle_cfg.exclude_severities),
            exclude_categories=frozenset(throttle_cfg.exclude_categories),
            max_entries=int(throttle_cfg.max_entries),
        )

    router = DefaultAlertRouterCls(audit_log=audit_log, throttle=throttle)
    channels: MutableMapping[str, AlertChannel] = {}
    offline_mode = bool(getattr(environment, "offline_mode", False))
    skipped_offline: list[str] = []

    for entry in environment.alert_channels:
        channel_type, _, channel_key = entry.partition(":")
        channel_type = channel_type.strip().lower()
        channel_key = channel_key.strip() or "default"

        requires_network = channel_type in {
            "telegram",
            "email",
            "sms",
            "signal",
            "whatsapp",
            "messenger",
        }
        if offline_mode and requires_network:
            skipped_offline.append(entry)
            continue

        if channel_type == "telegram":
            channel = _build_telegram_channel(core_config.telegram_channels, channel_key, secret_manager)
        elif channel_type == "email":
            channel = _build_email_channel(core_config.email_channels, channel_key, secret_manager)
        elif channel_type == "sms":
            channel = _build_sms_channel(core_config.sms_providers, channel_key, secret_manager)
        elif channel_type == "signal":
            channel = _build_signal_channel(core_config.signal_channels, channel_key, secret_manager)
        elif channel_type == "whatsapp":
            channel = _build_whatsapp_channel(core_config.whatsapp_channels, channel_key, secret_manager)
        elif channel_type == "messenger":
            channel = _build_messenger_channel(core_config.messenger_channels, channel_key, secret_manager)
        else:
            raise KeyError(f"Nieobsługiwany typ kanału alertów: {channel_type}")

        router.register(channel)
        channels[channel.name] = channel

    if skipped_offline:
        _LOGGER.info(
            "Tryb offline środowiska %s: pominięto kanały alertów wymagające sieci: %s",
            environment.name,
            ", ".join(skipped_offline),
        )

    return channels, router, audit_log


def _build_decision_journal(environment: EnvironmentConfig) -> TradingDecisionJournal | None:
    config: DecisionJournalConfig | None = getattr(environment, "decision_journal", None)
    if config is None:
        return None

    backend = getattr(config, "backend", "memory").lower()
    if backend == "memory":
        return InMemoryTradingDecisionJournal()
    if backend == "file":
        directory = Path(config.directory) if config.directory else Path("decisions")
        if not directory.is_absolute():
            base = Path(environment.data_cache_path)
            directory = base / directory
        return JsonlTradingDecisionJournal(
            directory=directory,
            filename_pattern=config.filename_pattern,
            retention_days=config.retention_days,
            fsync=config.fsync,
        )
    return None


def _build_portfolio_decision_log(
    core_config: CoreConfig, environment: EnvironmentConfig
) -> PortfolioDecisionLog | None:
    config = getattr(core_config, "portfolio_decision_log", None)
    if config is None or not getattr(config, "enabled", True):
        return None

    configured_path = getattr(config, "path", None)
    if configured_path:
        candidate = Path(str(configured_path)).expanduser()
        if not candidate.is_absolute():
            log_path = Path(environment.data_cache_path) / candidate
        else:
            log_path = candidate
    else:
        log_path = Path(environment.data_cache_path) / "portfolio_decisions.jsonl"

    max_entries = int(getattr(config, "max_entries", 512) or 512)
    signing_key = _load_portfolio_decision_log_key(config)
    signing_key_id = getattr(config, "signing_key_id", None)
    jsonl_fsync = bool(getattr(config, "jsonl_fsync", False))

    try:
        return PortfolioDecisionLog(
            max_entries=max_entries,
            jsonl_path=log_path,
            signing_key=signing_key,
            signing_key_id=str(signing_key_id) if signing_key_id else None,
            jsonl_fsync=jsonl_fsync,
        )
    except Exception:  # pragma: no cover - log portfelowy jest opcjonalny
        _LOGGER.exception("Nie udało się zainicjalizować PortfolioDecisionLog")
        return None


def _load_portfolio_decision_log_key(config: object) -> bytes | None:
    env_name = getattr(config, "signing_key_env", None)
    if env_name:
        env_value = os.environ.get(str(env_name))
        if env_value:
            return env_value.encode("utf-8")
        _LOGGER.warning(
            "PortfolioDecisionLog: zmienna środowiskowa %s nie jest ustawiona", env_name
        )

    key_path = getattr(config, "signing_key_path", None)
    if key_path:
        try:
            content = Path(str(key_path)).expanduser().read_bytes()
            stripped = content.strip()
            if stripped:
                return stripped
            _LOGGER.warning(
                "PortfolioDecisionLog: plik %s nie zawiera klucza", key_path
            )
        except Exception as exc:  # pragma: no cover - diagnostyka konfiguracji
            _LOGGER.warning(
                "PortfolioDecisionLog: błąd odczytu klucza z %s: %s", key_path, exc
            )

    key_value = getattr(config, "signing_key_value", None)
    if key_value not in (None, ""):
        return str(key_value).encode("utf-8")

    return None


def _build_telegram_channel(
    definitions: Mapping[str, TelegramChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> TelegramChannel:
    components = _get_alert_components()
    TelegramChannelCls = components["TelegramChannel"]
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Telegram '{channel_key}'") from exc

    token = secret_manager.load_secret_value(settings.token_secret, purpose="alerts:telegram")

    return TelegramChannelCls(
        bot_token=token,
        chat_id=settings.chat_id,
        parse_mode=settings.parse_mode,
        name=f"telegram:{channel_key}",
    )


def _build_email_channel(
    definitions: Mapping[str, EmailChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> EmailChannel:
    components = _get_alert_components()
    EmailChannelCls = components["EmailChannel"]
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału e-mail '{channel_key}'") from exc

    username = None
    password = None
    if settings.credential_secret:
        raw_secret = secret_manager.load_secret_value(settings.credential_secret, purpose="alerts:email")
        try:
            parsed = json.loads(raw_secret) if raw_secret else {}
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise SecretStorageError(
                "Sekret dla kanału e-mail musi zawierać poprawny JSON z polami 'username' i 'password'."
            ) from exc
        username = parsed.get("username")
        password = parsed.get("password")

    return EmailChannelCls(
        host=settings.host,
        port=settings.port,
        from_address=settings.from_address,
        recipients=settings.recipients,
        username=username,
        password=password,
        use_tls=settings.use_tls,
        name=f"email:{channel_key}",
    )


def _build_sms_channel(
    definitions: Mapping[str, SMSProviderSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> SMSChannel:
    components = _get_alert_components()
    SMSChannelCls = components["SMSChannel"]
    get_sms_provider_fn = components["get_sms_provider"]
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji dostawcy SMS '{channel_key}'") from exc

    if not settings.credential_key:
        raise SecretStorageError(
            f"Konfiguracja dostawcy SMS '{channel_key}' musi wskazywać 'credential_key'."
        )

    raw_secret = secret_manager.load_secret_value(settings.credential_key, purpose="alerts:sms")
    try:
        payload = json.loads(raw_secret)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SecretStorageError(
            "Sekret dostawcy SMS powinien zawierać JSON z polami 'account_sid' i 'auth_token'."
        ) from exc

    account_sid = payload.get("account_sid")
    auth_token = payload.get("auth_token")
    if not account_sid or not auth_token:
        raise SecretStorageError(
            "Sekret dostawcy SMS musi zawierać pola 'account_sid' oraz 'auth_token'."
        )

    provider_config = _resolve_sms_provider(settings, get_sms_provider_fn)
    sender = (
        settings.sender_id
        if settings.allow_alphanumeric_sender and settings.sender_id
        else settings.from_number
    )
    if not sender:
        raise SecretStorageError(
            f"Konfiguracja dostawcy SMS '{channel_key}' wymaga pola 'from_number' lub 'sender_id'."
        )

    recipients: Sequence[str] = tuple(settings.recipients)
    if not recipients:
        raise SecretStorageError(
            f"Konfiguracja dostawcy SMS '{channel_key}' wymaga co najmniej jednego odbiorcy."
        )

    return SMSChannelCls(
        account_sid=str(account_sid),
        auth_token=str(auth_token),
        from_number=sender,
        recipients=recipients,
        provider=provider_config,
        name=f"sms:{channel_key}",
    )


def _build_signal_channel(
    definitions: Mapping[str, SignalChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> SignalChannel:
    components = _get_alert_components()
    SignalChannelCls = components.get("SignalChannel")
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Signal '{channel_key}'") from exc

    token: str | None = None
    if settings.credential_secret:
        token = secret_manager.load_secret_value(settings.credential_secret, purpose="alerts:signal")

    if SignalChannelCls is None:
        raise KeyError("Kanał Signal jest niedostępny w tej dystrybucji")

    return SignalChannelCls(
        service_url=settings.service_url,
        sender_number=settings.sender_number,
        recipients=settings.recipients,
        auth_token=token,
        verify_tls=settings.verify_tls,
        name=f"signal:{channel_key}",
    )


def _build_whatsapp_channel(
    definitions: Mapping[str, WhatsAppChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> WhatsAppChannel:
    components = _get_alert_components()
    WhatsAppChannelCls = components.get("WhatsAppChannel")
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału WhatsApp '{channel_key}'") from exc

    token = secret_manager.load_secret_value(settings.token_secret, purpose="alerts:whatsapp")

    if WhatsAppChannelCls is None:
        raise KeyError("Kanał WhatsApp nie jest dostępny w tej dystrybucji")

    return WhatsAppChannelCls(
        phone_number_id=settings.phone_number_id,
        access_token=token,
        recipients=settings.recipients,
        api_base_url=settings.api_base_url,
        api_version=settings.api_version,
        name=f"whatsapp:{channel_key}",
    )


def _build_messenger_channel(
    definitions: Mapping[str, MessengerChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> MessengerChannel:
    components = _get_alert_components()
    MessengerChannelCls = components.get("MessengerChannel")
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Messenger '{channel_key}'") from exc

    token = secret_manager.load_secret_value(settings.token_secret, purpose="alerts:messenger")

    if MessengerChannelCls is None:
        raise KeyError("Kanał Messenger nie jest dostępny w tej instalacji")

    return MessengerChannelCls(
        page_id=settings.page_id,
        access_token=token,
        recipients=settings.recipients,
        api_base_url=settings.api_base_url,
        api_version=settings.api_version,
        name=f"messenger:{channel_key}",
    )


def _resolve_sms_provider(
    settings: SMSProviderSettings, get_sms_provider_fn: Any
) -> SmsProviderConfig:
    components = _get_alert_components()
    SmsProviderConfigCls = components.get("SmsProviderConfig")
    if SmsProviderConfigCls is None:
        raise KeyError("Typ SmsProviderConfig nie jest dostępny w module alertów")

    base = get_sms_provider_fn(settings.provider_key)
    return SmsProviderConfigCls(
        provider_id=base.provider_id,
        display_name=base.display_name,
        api_base_url=settings.api_base_url or base.api_base_url,
        iso_country_code=base.iso_country_code,
        supports_alphanumeric_sender=settings.allow_alphanumeric_sender,
        notes=base.notes,
        max_sender_length=base.max_sender_length,
    )


__all__ = [
    "BootstrapContext",
    "RuntimeEntrypoint",
    "bootstrap_environment",
    "catalog_runtime_entrypoints",
    "resolve_runtime_entrypoint",
    "get_registered_adapter_factories",
    "register_adapter_factory",
    "unregister_adapter_factory",
    "register_adapter_factory_from_path",
    "parse_adapter_factory_cli_specs",
    "temporary_adapter_factories",
    "build_alert_channels",
]
