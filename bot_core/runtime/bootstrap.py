"""Procedury rozruchowe spinające konfigurację z modułami runtime."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import os
import stat
from typing import Any, Mapping, MutableMapping, Sequence

from bot_core.alerts import (
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
    get_sms_provider,
)
from bot_core.alerts.base import AlertAuditLog, AlertChannel
from bot_core.alerts.channels.providers import SmsProviderConfig
from bot_core.config.loader import load_core_config
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
from bot_core.config.validation import assert_core_config_valid
from bot_core.exchanges.base import (
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeCredentials,
)
from bot_core.exchanges.binance import BinanceFuturesAdapter, BinanceSpotAdapter
from bot_core.exchanges.kraken import KrakenFuturesAdapter, KrakenSpotAdapter
from bot_core.exchanges.nowa_gielda import NowaGieldaSpotAdapter
from bot_core.exchanges.zonda import ZondaSpotAdapter
from bot_core.risk.base import RiskRepository
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.factory import build_risk_profile_from_config
from bot_core.risk.repository import FileRiskRepository
from bot_core.security import SecretManager, SecretStorageError, build_service_token_validator
from bot_core.security.tokens import ServiceTokenValidator
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

_DEFAULT_ADAPTERS: Mapping[str, ExchangeAdapterFactory] = {
    "binance_spot": BinanceSpotAdapter,
    "binance_futures": BinanceFuturesAdapter,
    "kraken_spot": KrakenSpotAdapter,
    "kraken_futures": KrakenFuturesAdapter,
    "nowa_gielda_spot": NowaGieldaSpotAdapter,
    "zonda_spot": ZondaSpotAdapter,
}

_LOGGER = logging.getLogger(__name__)


def _build_ui_alert_audit_metadata(
    router: DefaultAlertRouter | None,
    *,
    requested_backend: str | None,
) -> dict[str, object]:
    """Zwraca metadane backendu audytu alertów UI dostępne w runtime."""

    normalized_request = (requested_backend or "inherit").lower()
    metadata: dict[str, object] = {"requested": normalized_request}

    audit_log = getattr(router, "audit_log", None)

    if isinstance(audit_log, FileAlertAuditLog):
        metadata.update(
            {
                "backend": "file",
                "directory": str(getattr(audit_log, "directory", "")) or None,
                "pattern": getattr(audit_log, "filename_pattern", None),
                "retention_days": getattr(audit_log, "retention_days", None),
                "fsync": bool(getattr(audit_log, "fsync", False)),
            }
        )
    elif isinstance(audit_log, InMemoryAlertAuditLog):
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

    report_paths_attr = getattr(tco_config, "report_paths", None)
    if not report_paths_attr:
        report_paths_attr = getattr(tco_config, "reports", ())
    report_paths = tuple(report_paths_attr or ())
    require_at_startup = bool(getattr(tco_config, "require_at_startup", False))
    for raw_path in report_paths:
        path = Path(str(raw_path)).expanduser()
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
    portfolio_decision_log: PortfolioDecisionLog | None = None
    decision_engine_config: Any | None = None
    decision_orchestrator: Any | None = None
    decision_tco_report_path: str | None = None
    decision_tco_warnings: Sequence[str] | None = None
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


def bootstrap_environment(
    environment_name: str,
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    risk_profile_name: str | None = None,
) -> BootstrapContext:
    """Tworzy kompletny kontekst uruchomieniowy dla wskazanego środowiska."""
    core_config = load_core_config(config_path)
    validation = assert_core_config_valid(core_config)
    for warning in validation.warnings:
        _LOGGER.warning("Walidacja konfiguracji: %s", warning)
    if environment_name not in core_config.environments:
        raise KeyError(f"Środowisko '{environment_name}' nie istnieje w konfiguracji")

    environment = core_config.environments[environment_name]
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
    portfolio_governor: Any | None = None
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
    if adapter_factories:
        factories.update(adapter_factories)
    adapter = _instantiate_adapter(
        environment.exchange,
        credentials,
        factories,
        environment.environment,
        settings=environment.adapter_settings,
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
        metrics_service_enabled = bool(getattr(metrics_config, "enabled", False))
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

    if build_metrics_server_from_config is not None:
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

    if build_risk_server_from_config is not None and risk_config is not None:
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
        decision_engine_config=decision_engine_config,
        decision_orchestrator=decision_orchestrator,
        decision_tco_report_path=decision_tco_report_path,
        decision_tco_warnings=tuple(decision_tco_warnings)
        if decision_tco_warnings
        else None,
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
    )


def _instantiate_adapter(
    exchange_name: str,
    credentials: ExchangeCredentials,
    factories: Mapping[str, ExchangeAdapterFactory],
    environment: Environment,
    *,
    settings: Mapping[str, Any] | None = None,
) -> ExchangeAdapter:
    try:
        factory = factories[exchange_name]
    except KeyError as exc:
        raise KeyError(f"Brak fabryki adaptera dla giełdy '{exchange_name}'") from exc
    if settings:
        return factory(credentials, environment=environment, settings=settings)
    return factory(credentials, environment=environment)


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
    # audit_log: InMemory (domyślnie) albo FileAlertAuditLog, jeśli skonfigurowano alert_audit
    audit_config = getattr(environment, "alert_audit", None)
    if audit_config and getattr(audit_config, "backend", "memory") == "file":
        directory = Path(audit_config.directory) if audit_config.directory else Path("alerts")
        if not directory.is_absolute():
            base = Path(environment.data_cache_path)
            directory = base / directory
        audit_log: AlertAuditLog = FileAlertAuditLog(
            directory=directory,
            filename_pattern=audit_config.filename_pattern,
            retention_days=audit_config.retention_days,
            fsync=audit_config.fsync,
        )
    else:
        audit_log = InMemoryAlertAuditLog()

    # throttle (opcjonalny)
    throttle_cfg = getattr(environment, "alert_throttle", None)
    throttle: AlertThrottle | None = None
    if throttle_cfg is not None:
        throttle = AlertThrottle(
            window=timedelta(seconds=float(throttle_cfg.window_seconds)),
            exclude_severities=frozenset(throttle_cfg.exclude_severities),
            exclude_categories=frozenset(throttle_cfg.exclude_categories),
            max_entries=int(throttle_cfg.max_entries),
        )

    router = DefaultAlertRouter(audit_log=audit_log, throttle=throttle)
    channels: MutableMapping[str, AlertChannel] = {}

    for entry in environment.alert_channels:
        channel_type, _, channel_key = entry.partition(":")
        channel_type = channel_type.strip().lower()
        channel_key = channel_key.strip() or "default"

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
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Telegram '{channel_key}'") from exc

    token = secret_manager.load_secret_value(settings.token_secret, purpose="alerts:telegram")

    return TelegramChannel(
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

    return EmailChannel(
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

    provider_config = _resolve_sms_provider(settings)
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

    return SMSChannel(
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
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Signal '{channel_key}'") from exc

    token: str | None = None
    if settings.credential_secret:
        token = secret_manager.load_secret_value(settings.credential_secret, purpose="alerts:signal")

    return SignalChannel(
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
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału WhatsApp '{channel_key}'") from exc

    token = secret_manager.load_secret_value(settings.token_secret, purpose="alerts:whatsapp")

    return WhatsAppChannel(
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
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Messenger '{channel_key}'") from exc

    token = secret_manager.load_secret_value(settings.token_secret, purpose="alerts:messenger")

    return MessengerChannel(
        page_id=settings.page_id,
        access_token=token,
        recipients=settings.recipients,
        api_base_url=settings.api_base_url,
        api_version=settings.api_version,
        name=f"messenger:{channel_key}",
    )


def _resolve_sms_provider(settings: SMSProviderSettings) -> SmsProviderConfig:
    base = get_sms_provider(settings.provider_key)
    return SmsProviderConfig(
        provider_id=base.provider_id,
        display_name=base.display_name,
        api_base_url=settings.api_base_url or base.api_base_url,
        iso_country_code=base.iso_country_code,
        supports_alphanumeric_sender=settings.allow_alphanumeric_sender,
        notes=base.notes,
        max_sender_length=base.max_sender_length,
    )


__all__ = ["BootstrapContext", "bootstrap_environment", "build_alert_channels"]
