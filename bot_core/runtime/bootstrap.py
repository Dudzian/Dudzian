"""Procedury rozruchowe spinające konfigurację z modułami runtime."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
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
from bot_core.exchanges.zonda import ZondaSpotAdapter
from bot_core.risk.base import RiskRepository
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.factory import build_risk_profile_from_config
from bot_core.risk.repository import FileRiskRepository
from bot_core.security import SecretManager, SecretStorageError
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

# --- Metrics service (opcjonalny – w niektórych gałęziach może nie istnieć) ---
try:  # pragma: no cover - środowiska bez grpcio lub wygenerowanych stubów
    from bot_core.runtime.metrics_service import (  # type: ignore
        MetricsServer,
        build_metrics_server_from_config,
    )
except Exception:  # pragma: no cover - brak zależności opcjonalnych
    MetricsServer = None  # type: ignore
    build_metrics_server_from_config = None  # type: ignore

try:  # pragma: no cover - sink telemetrii może być pominięty
    from bot_core.runtime.metrics_alerts import (  # type: ignore
        DEFAULT_UI_ALERTS_JSONL_PATH,
        UiTelemetryAlertSink,
    )
except Exception:  # pragma: no cover - brak telemetrii UI
    UiTelemetryAlertSink = None  # type: ignore
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

_DEFAULT_ADAPTERS: Mapping[str, ExchangeAdapterFactory] = {
    "binance_spot": BinanceSpotAdapter,
    "binance_futures": BinanceFuturesAdapter,
    "kraken_spot": KrakenSpotAdapter,
    "kraken_futures": KrakenFuturesAdapter,
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
    risk_profile_name: str
    metrics_server: Any | None = None
    metrics_ui_alerts_path: Path | None = None
    metrics_jsonl_path: Path | None = None
    metrics_ui_alert_sink_active: bool = False
    metrics_service_enabled: bool | None = None
    metrics_ui_alerts_metadata: Mapping[str, Any] | None = None
    metrics_jsonl_metadata: Mapping[str, Any] | None = None
    metrics_security_warnings: tuple[str, ...] | None = None
    metrics_ui_alerts_settings: Mapping[str, Any] | None = None


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
    risk_engine = ThresholdRiskEngine(repository=risk_repository)
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
    metrics_security_warnings: list[str] = []
    metrics_security_payload: dict[str, object] = {}
    metrics_config = getattr(core_config, "metrics_service", None)
    if metrics_config is not None:
        metrics_service_enabled = bool(getattr(metrics_config, "enabled", False))
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
            reduce_mode = "enable"
            overlay_mode = "enable"
            jank_mode = "disable"
            if metrics_config is not None and getattr(metrics_config, "reduce_motion_mode", None) is not None:
                candidate = str(getattr(metrics_config, "reduce_motion_mode", "enable")).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    reduce_mode = candidate
            if metrics_config is not None and getattr(metrics_config, "overlay_alert_mode", None) is not None:
                candidate = str(getattr(metrics_config, "overlay_alert_mode", "enable")).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    overlay_mode = candidate
            if metrics_config is not None and getattr(metrics_config, "jank_alert_mode", None) is not None:
                candidate = str(getattr(metrics_config, "jank_alert_mode", "disable")).lower()
                if candidate in {"enable", "jsonl", "disable"}:
                    jank_mode = candidate
            if metrics_config is not None and getattr(metrics_config, "reduce_motion_mode", None) is None:
                reduce_mode = (
                    "enable"
                    if bool(getattr(metrics_config, "reduce_motion_alerts", True))
                    else "disable"
                )
            if metrics_config is not None and getattr(metrics_config, "overlay_alert_mode", None) is None:
                overlay_mode = (
                    "enable"
                    if bool(getattr(metrics_config, "overlay_alerts", True))
                    else "disable"
                )
            if metrics_config is not None and getattr(metrics_config, "jank_alert_mode", None) is None:
                jank_mode = (
                    "enable"
                    if bool(getattr(metrics_config, "jank_alerts", False))
                    else "disable"
                )

            reduce_dispatch = reduce_mode == "enable"
            overlay_dispatch = overlay_mode == "enable"
            jank_dispatch = jank_mode == "enable"
            reduce_logging = reduce_mode in {"enable", "jsonl"}
            overlay_logging = overlay_mode in {"enable", "jsonl"}
            jank_logging = jank_mode in {"enable", "jsonl"}

            sink_kwargs: dict[str, object] = {
                "jsonl_path": telemetry_log,
                "enable_reduce_motion_alerts": reduce_dispatch,
                "enable_overlay_alerts": overlay_dispatch,
                "log_reduce_motion_events": reduce_logging,
                "log_overlay_events": overlay_logging,
                "enable_jank_alerts": jank_dispatch,
                "log_jank_events": jank_logging,
            }
            settings_payload: dict[str, object] = {
                "jsonl_path": str(telemetry_log),
                "reduce_mode": reduce_mode,
                "overlay_mode": overlay_mode,
                "jank_mode": jank_mode,
                "reduce_motion_alerts": reduce_dispatch,
                "overlay_alerts": overlay_dispatch,
                "jank_alerts": jank_dispatch,
                "reduce_motion_logging": reduce_logging,
                "overlay_logging": overlay_logging,
                "jank_logging": jank_logging,
                "reduce_motion_category": "ui.performance",
                "reduce_motion_severity_active": "warning",
                "reduce_motion_severity_recovered": "info",
                "overlay_category": "ui.performance",
                "overlay_severity_exceeded": "warning",
                "overlay_severity_recovered": "info",
                "overlay_severity_critical": "critical",
                "overlay_critical_threshold": 2,
                "jank_category": "ui.performance",
                "jank_severity_spike": "warning",
                "jank_severity_critical": None,
                "jank_critical_over_ms": None,
            }
            if metrics_config is not None:
                sink_kwargs.update(
                    reduce_motion_category=getattr(
                        metrics_config, "reduce_motion_category", "ui.performance"
                    ),
                    reduce_motion_severity_active=getattr(
                        metrics_config, "reduce_motion_severity_active", "warning"
                    ),
                    reduce_motion_severity_recovered=getattr(
                        metrics_config, "reduce_motion_severity_recovered", "info"
                    ),
                    overlay_category=getattr(
                        metrics_config, "overlay_alert_category", "ui.performance"
                    ),
                    overlay_severity_exceeded=getattr(
                        metrics_config, "overlay_alert_severity_exceeded", "warning"
                    ),
                    overlay_severity_recovered=getattr(
                        metrics_config, "overlay_alert_severity_recovered", "info"
                    ),
                    jank_category=getattr(metrics_config, "jank_alert_category", "ui.performance"),
                    jank_severity_spike=getattr(
                        metrics_config, "jank_alert_severity_spike", "warning"
                    ),
                )
                settings_payload.update(
                    reduce_motion_category=getattr(
                        metrics_config, "reduce_motion_category", settings_payload["reduce_motion_category"]
                    ),
                    reduce_motion_severity_active=getattr(
                        metrics_config,
                        "reduce_motion_severity_active",
                        settings_payload["reduce_motion_severity_active"],
                    ),
                    reduce_motion_severity_recovered=getattr(
                        metrics_config,
                        "reduce_motion_severity_recovered",
                        settings_payload["reduce_motion_severity_recovered"],
                    ),
                    overlay_category=getattr(
                        metrics_config, "overlay_alert_category", settings_payload["overlay_category"]
                    ),
                    overlay_severity_exceeded=getattr(
                        metrics_config,
                        "overlay_alert_severity_exceeded",
                        settings_payload["overlay_severity_exceeded"],
                    ),
                    overlay_severity_recovered=getattr(
                        metrics_config,
                        "overlay_alert_severity_recovered",
                        settings_payload["overlay_severity_recovered"],
                    ),
                    overlay_severity_critical=getattr(
                        metrics_config,
                        "overlay_alert_severity_critical",
                        settings_payload["overlay_severity_critical"],
                    ),
                    overlay_critical_threshold=getattr(
                        metrics_config,
                        "overlay_alert_critical_threshold",
                        settings_payload["overlay_critical_threshold"],
                    ),
                    jank_category=getattr(
                        metrics_config, "jank_alert_category", settings_payload["jank_category"]
                    ),
                    jank_severity_spike=getattr(
                        metrics_config,
                        "jank_alert_severity_spike",
                        settings_payload["jank_severity_spike"],
                    ),
                    jank_severity_critical=getattr(
                        metrics_config,
                        "jank_alert_severity_critical",
                        settings_payload["jank_severity_critical"],
                    ),
                    jank_critical_over_ms=getattr(
                        metrics_config,
                        "jank_alert_critical_over_ms",
                        settings_payload["jank_critical_over_ms"],
                    ),
                )
                overlay_critical = getattr(metrics_config, "overlay_alert_severity_critical", None)
                if overlay_critical is not None:
                    sink_kwargs["overlay_severity_critical"] = overlay_critical
                overlay_threshold = getattr(metrics_config, "overlay_alert_critical_threshold", None)
                if overlay_threshold is not None:
                    try:
                        threshold_value = int(overlay_threshold)
                        sink_kwargs["overlay_critical_threshold"] = threshold_value
                        settings_payload["overlay_critical_threshold"] = threshold_value
                    except (TypeError, ValueError):  # pragma: no cover - diagnostyka
                        _LOGGER.debug(
                            "Nieprawidłowy próg overlay_alert_critical_threshold=%s", overlay_threshold
                        )
                else:
                    settings_payload["overlay_critical_threshold"] = None
                jank_severity_critical = getattr(metrics_config, "jank_alert_severity_critical", None)
                if jank_severity_critical is not None:
                    sink_kwargs["jank_severity_critical"] = jank_severity_critical
                jank_threshold = getattr(metrics_config, "jank_alert_critical_over_ms", None)
                if jank_threshold is not None:
                    try:
                        threshold_ms = float(jank_threshold)
                    except (TypeError, ValueError):  # pragma: no cover - diagnostyka
                        _LOGGER.debug(
                            "Nieprawidłowy próg jank_alert_critical_over_ms=%s", jank_threshold
                        )
                    else:
                        sink_kwargs["jank_critical_over_ms"] = threshold_ms
                        settings_payload["jank_critical_over_ms"] = threshold_ms
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

    if build_metrics_server_from_config is not None:
        try:
            # Najpierw spróbuj pełnej, najnowszej sygnatury (cfg, sinks, alerts_router)
            try:
                metrics_server = build_metrics_server_from_config(  # type: ignore[call-arg]
                    core_config.metrics_service,
                    sinks=metrics_sinks or None,
                    alerts_router=alert_router,
                )
            except TypeError:
                # Następnie (cfg, alerts_router)
                try:
                    metrics_server = build_metrics_server_from_config(  # type: ignore[call-arg]
                        core_config.metrics_service,
                        alerts_router=alert_router,
                    )
                except TypeError:
                    # Potem (cfg, sinks)
                    try:
                        metrics_server = build_metrics_server_from_config(  # type: ignore[call-arg]
                            core_config.metrics_service,
                            sinks=metrics_sinks or None,
                        )
                    except TypeError:
                        # Na końcu najstarsza postać: tylko (cfg)
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
        risk_profile_name=selected_profile,
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
        metrics_ui_alerts_settings=metrics_ui_alerts_settings,
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
