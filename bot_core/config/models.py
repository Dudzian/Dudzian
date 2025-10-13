"""Modele konfiguracji dla nowej architektury."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bot_core.exchanges.base import Environment


# --- Alerty / audyt ----------------------------------------------------------

@dataclass(slots=True)
class AlertThrottleConfig:
    """Parametry okna tłumienia powtarzających się alertów."""
    window_seconds: float
    exclude_severities: Sequence[str] = field(default_factory=tuple)
    exclude_categories: Sequence[str] = field(default_factory=tuple)
    max_entries: int = 2048


@dataclass(slots=True)
class AlertAuditConfig:
    """Konfiguracja repozytorium audytowego alertów."""
    backend: str
    directory: str | None = None
    filename_pattern: str = "alerts-%Y%m%d.jsonl"
    retention_days: int | None = 730
    fsync: bool = False


@dataclass(slots=True)
class DecisionJournalConfig:
    """Konfiguracja dziennika decyzji tradingowych."""
    backend: str
    directory: str | None = None
    filename_pattern: str = "decisions-%Y%m%d.jsonl"
    retention_days: int | None = 730
    fsync: bool = False


@dataclass(slots=True)
class ServiceTokenConfig:
    """Definicja tokenu usługowego wykorzystywanego do RBAC."""

    token_id: str
    token_env: str | None = None
    token_value: str | None = None
    token_hash: str | None = None
    scopes: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class MetricsServiceTlsConfig:
    """Opcjonalna konfiguracja TLS/mTLS dla serwera telemetrii."""
    enabled: bool = False
    certificate_path: str | None = None
    private_key_path: str | None = None
    client_ca_path: str | None = None
    require_client_auth: bool = False
    private_key_password_env: str | None = None
    pinned_fingerprints: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class MetricsServiceConfig:
    """Ustawienia serwera telemetrii `MetricsService`."""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 0
    history_size: int = 1024

    # Rozszerzone pola (autoryzacja / logowanie / pliki)
    auth_token: str | None = None
    log_sink: bool = True
    jsonl_path: str | None = None
    jsonl_fsync: bool = False

    # Dodatkowe ścieżki/bezpieczeństwo
    ui_alerts_jsonl_path: str | None = None
    ui_alerts_audit_backend: str | None = None
    ui_alerts_risk_profile: str | None = None
    ui_alerts_risk_profiles_file: str | None = None
    tls: MetricsServiceTlsConfig | None = None
    rbac_tokens: Sequence[ServiceTokenConfig] = field(default_factory=tuple)

    # Opcjonalne alerty związane z UI/performance
    reduce_motion_alerts: bool = False
    reduce_motion_mode: str | None = None
    reduce_motion_category: str = "ui.performance"
    reduce_motion_severity_active: str = "warning"
    reduce_motion_severity_recovered: str = "info"

    overlay_alerts: bool = False
    overlay_alert_mode: str | None = None
    overlay_alert_category: str = "ui.performance"
    overlay_alert_severity_exceeded: str = "warning"
    overlay_alert_severity_recovered: str = "info"
    overlay_alert_severity_critical: str | None = "critical"
    overlay_alert_critical_threshold: int | None = 2

    jank_alerts: bool = False
    jank_alert_mode: str | None = None
    jank_alert_category: str = "ui.performance"
    jank_alert_severity_spike: str = "warning"
    jank_alert_severity_critical: str | None = None
    jank_alert_critical_over_ms: float | None = None


@dataclass(slots=True)
class RiskServiceConfig:
    """Ustawienia serwera `RiskService`."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 0
    history_size: int = 256
    auth_token: str | None = None
    tls: MetricsServiceTlsConfig | None = None
    publish_interval_seconds: float = 5.0
    profiles: Sequence[str] = field(default_factory=tuple)
    rbac_tokens: Sequence[ServiceTokenConfig] = field(default_factory=tuple)


@dataclass(slots=True)
class RiskDecisionLogConfig:
    """Konfiguracja dziennika decyzji silnika ryzyka."""

    enabled: bool = True
    path: str | None = None
    max_entries: int = 1_000
    signing_key_env: str | None = None
    signing_key_path: str | None = None
    signing_key_value: str | None = None
    signing_key_id: str | None = None
    jsonl_fsync: bool = False


@dataclass(slots=True)
class SecurityBaselineSigningConfig:
    """Ustawienia podpisywania raportów audytu bezpieczeństwa."""

    signing_key_env: str | None = None
    signing_key_path: str | None = None
    signing_key_value: str | None = None
    signing_key_id: str | None = None
    require_signature: bool = False


@dataclass(slots=True)
class SecurityBaselineConfig:
    """Konfiguracja integracji audytu bezpieczeństwa."""

    signing: SecurityBaselineSigningConfig | None = None


# --- Środowiska / rdzeń ------------------------------------------------------

@dataclass(slots=True)
class EnvironmentDataQualityConfig:
    """Progi jakości danych używane przez środowisko."""
    max_gap_minutes: float | None = None
    min_ok_ratio: float | None = None


@dataclass(slots=True)
class CoverageMonitorTargetConfig:
    """Definicja środowiska objętego monitoringiem pokrycia danych."""
    environment: str
    dispatch: bool | None = None
    category: str | None = None
    severity_override: str | None = None


@dataclass(slots=True)
class CoverageMonitoringConfig:
    """Ustawienia globalnego monitoringu pokrycia danych OHLCV."""
    enabled: bool = True
    default_dispatch: bool = True
    default_category: str = "data.ohlcv"
    targets: Sequence[CoverageMonitorTargetConfig] = field(default_factory=tuple)


@dataclass(slots=True)
class EnvironmentConfig:
    """Konfiguracja środowiska (np. live, paper, testnet)."""
    name: str
    exchange: str
    environment: Environment
    keychain_key: str
    data_cache_path: str
    risk_profile: str
    alert_channels: Sequence[str]
    ip_allowlist: Sequence[str] = field(default_factory=tuple)
    credential_purpose: str = "trading"
    instrument_universe: str | None = None
    adapter_settings: Mapping[str, Any] = field(default_factory=dict)
    required_permissions: Sequence[str] = field(default_factory=tuple)
    forbidden_permissions: Sequence[str] = field(default_factory=tuple)
    alert_throttle: AlertThrottleConfig | None = None
    alert_audit: AlertAuditConfig | None = None
    data_quality: EnvironmentDataQualityConfig | None = None
    decision_journal: DecisionJournalConfig | None = None
    default_strategy: str | None = None
    default_controller: str | None = None


@dataclass(slots=True)
class RiskProfileConfig:
    """Parametry wykorzystywane do inicjalizacji profili ryzyka."""

    name: str
    max_daily_loss_pct: float
    max_position_pct: float
    target_volatility: float
    max_leverage: float
    stop_loss_atr_multiple: float
    max_open_positions: int
    hard_drawdown_pct: float
    data_quality: EnvironmentDataQualityConfig | None = None
    strategy_allocations: Mapping[str, float] = field(default_factory=dict)
    instrument_buckets: Sequence[str] = field(default_factory=tuple)


# --- Instrumenty / uniwersa --------------------------------------------------

@dataclass(slots=True)
class InstrumentBackfillWindow:
    """Definicja zakresu danych historycznych dla danego interwału."""
    interval: str
    lookback_days: int


@dataclass(slots=True)
class InstrumentConfig:
    """Opis pojedynczego instrumentu w uniwersum."""
    name: str
    base_asset: str
    quote_asset: str
    categories: Sequence[str]
    exchange_symbols: Mapping[str, str]
    backfill_windows: Sequence[InstrumentBackfillWindow] = field(default_factory=tuple)


@dataclass(slots=True)
class InstrumentUniverseConfig:
    """Zbiór instrumentów przypisany do środowisk."""
    name: str
    description: str
    instruments: Sequence[InstrumentConfig]


@dataclass(slots=True)
class InstrumentBucketConfig:
    """Opis koszyka instrumentów przypisanych do profili ryzyka."""

    name: str
    universe: str
    symbols: Sequence[str]
    max_position_pct: float | None = None
    max_notional_usd: float | None = None
    tags: Sequence[str] = field(default_factory=tuple)


# --- Strategie ----------------------------------------------------------------

@dataclass(slots=True)
class DailyTrendMomentumStrategyConfig:
    """Konfiguracja strategii trend/momentum."""
    name: str
    fast_ma: int
    slow_ma: int
    breakout_lookback: int
    momentum_window: int
    atr_window: int
    atr_multiplier: float
    min_trend_strength: float
    min_momentum: float


@dataclass(slots=True)
class MeanReversionStrategyConfig:
    """Parametry strategii powrotu do średniej."""

    name: str
    lookback: int
    entry_zscore: float
    exit_zscore: float
    max_holding_period: int
    volatility_cap: float
    min_volume_usd: float


@dataclass(slots=True)
class VolatilityTargetingStrategyConfig:
    """Konfiguracja strategii kontroli zmienności portfela."""

    name: str
    target_volatility: float
    lookback: int
    rebalance_threshold: float
    min_allocation: float
    max_allocation: float
    floor_volatility: float


@dataclass(slots=True)
class CrossExchangeArbitrageStrategyConfig:
    """Parametry strategii arbitrażowej cross-exchange."""

    name: str
    primary_exchange: str
    secondary_exchange: str
    spread_entry: float
    spread_exit: float
    max_notional: float
    max_open_seconds: int


@dataclass(slots=True)
class StrategyScheduleConfig:
    """Opis pojedynczego zadania harmonogramu strategii."""

    name: str
    strategy: str
    cadence_seconds: int
    max_drift_seconds: int
    warmup_bars: int
    risk_profile: str
    max_signals: int = 10
    interval: str | None = None


@dataclass(slots=True)
class MultiStrategySchedulerConfig:
    """Konfiguracja scheduler-a wielostrate-gicznego."""

    name: str
    schedules: Sequence[StrategyScheduleConfig]
    telemetry_namespace: str
    decision_log_category: str = "runtime.scheduler"
    health_check_interval: int = 300
    rbac_tokens: Sequence[ServiceTokenConfig] = field(default_factory=tuple)


# --- Kanały alertów -----------------------------------------------------------

@dataclass(slots=True)
class SMSProviderSettings:
    """Definicja lokalnego dostawcy SMS z konfiguracji."""
    name: str
    provider_key: str
    api_base_url: str
    from_number: str
    recipients: Sequence[str]
    allow_alphanumeric_sender: bool = False
    sender_id: str | None = None
    credential_key: str | None = None


@dataclass(slots=True)
class TelegramChannelSettings:
    """Konfiguracja kanału Telegram."""
    name: str
    chat_id: str
    token_secret: str
    parse_mode: str = "MarkdownV2"


@dataclass(slots=True)
class EmailChannelSettings:
    """Konfiguracja kanału e-mail."""
    name: str
    host: str
    port: int
    from_address: str
    recipients: Sequence[str]
    credential_secret: str | None = None
    use_tls: bool = True


@dataclass(slots=True)
class SignalChannelSettings:
    """Konfiguracja kanału Signal opartego o usługę signal-cli."""
    name: str
    service_url: str
    sender_number: str
    recipients: Sequence[str]
    credential_secret: str | None = None
    verify_tls: bool = True


@dataclass(slots=True)
class WhatsAppChannelSettings:
    """Konfiguracja kanału WhatsApp wykorzystującego Graph API."""
    name: str
    phone_number_id: str
    recipients: Sequence[str]
    token_secret: str
    api_base_url: str = "https://graph.facebook.com"
    api_version: str = "v16.0"


@dataclass(slots=True)
class MessengerChannelSettings:
    """Konfiguracja kanału Facebook Messenger."""
    name: str
    page_id: str
    recipients: Sequence[str]
    token_secret: str
    api_base_url: str = "https://graph.facebook.com"
    api_version: str = "v16.0"


# --- Runtime ------------------------------------------------------------------

@dataclass(slots=True)
class ControllerRuntimeConfig:
    """Parametry sterujące cyklem pracy kontrolerów runtime."""
    tick_seconds: float
    interval: str


@dataclass(slots=True)
class RuntimeResourceLimitsConfig:
    """Deklaracja budżetów zasobów dla runtime."""

    cpu_percent: float
    memory_mb: float
    io_read_mb_s: float
    io_write_mb_s: float
    headroom_warning_threshold: float = 0.85


@dataclass(slots=True)
class SmokeArchiveLocalConfig:
    """Konfiguracja lokalnego magazynu archiwów smoke testów."""
    directory: str
    filename_pattern: str = "{environment}_{date}_{hash}.zip"
    fsync: bool = False


@dataclass(slots=True)
class SmokeArchiveS3Config:
    """Konfiguracja wysyłki archiwów smoke testu do S3/MinIO."""
    bucket: str
    object_prefix: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    use_ssl: bool = True
    extra_args: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SmokeArchiveUploadConfig:
    """Parametry wysyłki archiwum smoke testu po udanym przebiegu."""
    backend: str
    credential_secret: str | None = None
    local: SmokeArchiveLocalConfig | None = None
    s3: SmokeArchiveS3Config | None = None


@dataclass(slots=True)
class PaperSmokeJsonSyncLocalConfig:
    """Konfiguracja lokalnego archiwum dziennika JSONL smoke testów."""
    directory: str
    filename_pattern: str = "{environment}_{date}.jsonl"
    fsync: bool = False


@dataclass(slots=True)
class PaperSmokeJsonSyncS3Config:
    """Konfiguracja wysyłki dziennika JSONL smoke testów do S3/MinIO."""
    bucket: str
    object_prefix: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    use_ssl: bool = True
    extra_args: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PaperSmokeJsonSyncConfig:
    """Parametry synchronizacji dziennika JSONL smoke testów."""
    backend: str
    credential_secret: str | None = None
    local: PaperSmokeJsonSyncLocalConfig | None = None
    s3: PaperSmokeJsonSyncS3Config | None = None


@dataclass(slots=True)
class CoreReportingConfig:
    """Konfiguracja sekcji reportingowej CoreConfig."""
    daily_report_time_utc: str | None = None
    weekly_report_day: str | None = None
    retention_months: str | None = None
    smoke_archive_upload: SmokeArchiveUploadConfig | None = None
    paper_smoke_json_sync: PaperSmokeJsonSyncConfig | None = None


@dataclass(slots=True)
class CoreConfig:
    """Najwyższego poziomu konfiguracja aplikacji."""
    environments: Mapping[str, EnvironmentConfig]
    risk_profiles: Mapping[str, RiskProfileConfig]
    instrument_universes: Mapping[str, InstrumentUniverseConfig] = field(default_factory=dict)
    instrument_buckets: Mapping[str, InstrumentBucketConfig] = field(default_factory=dict)
    strategies: Mapping[str, DailyTrendMomentumStrategyConfig] = field(default_factory=dict)
    mean_reversion_strategies: Mapping[str, MeanReversionStrategyConfig] = field(default_factory=dict)
    volatility_target_strategies: Mapping[str, VolatilityTargetingStrategyConfig] = field(default_factory=dict)
    cross_exchange_arbitrage_strategies: Mapping[
        str, CrossExchangeArbitrageStrategyConfig
    ] = field(default_factory=dict)
    multi_strategy_schedulers: Mapping[str, MultiStrategySchedulerConfig] = field(default_factory=dict)
    reporting: CoreReportingConfig | None = None
    sms_providers: Mapping[str, SMSProviderSettings] = field(default_factory=dict)
    telegram_channels: Mapping[str, TelegramChannelSettings] = field(default_factory=dict)
    email_channels: Mapping[str, EmailChannelSettings] = field(default_factory=dict)
    signal_channels: Mapping[str, SignalChannelSettings] = field(default_factory=dict)
    whatsapp_channels: Mapping[str, WhatsAppChannelSettings] = field(default_factory=dict)
    messenger_channels: Mapping[str, MessengerChannelSettings] = field(default_factory=dict)
    runtime_controllers: Mapping[str, ControllerRuntimeConfig] = field(default_factory=dict)
    coverage_monitoring: CoverageMonitoringConfig | None = None
    metrics_service: MetricsServiceConfig | None = None
    risk_service: RiskServiceConfig | None = None
    risk_decision_log: RiskDecisionLogConfig | None = None
    security_baseline: SecurityBaselineConfig | None = None
    runtime_resource_limits: RuntimeResourceLimitsConfig | None = None
    source_path: str | None = None
    source_directory: str | None = None


__all__ = [
    "EnvironmentConfig",
    "EnvironmentDataQualityConfig",
    "CoverageMonitorTargetConfig",
    "CoverageMonitoringConfig",
    "RiskProfileConfig",
    "InstrumentBackfillWindow",
    "InstrumentConfig",
    "InstrumentUniverseConfig",
    "InstrumentBucketConfig",
    "DailyTrendMomentumStrategyConfig",
    "MeanReversionStrategyConfig",
    "VolatilityTargetingStrategyConfig",
    "CrossExchangeArbitrageStrategyConfig",
    "StrategyScheduleConfig",
    "MultiStrategySchedulerConfig",
    "SMSProviderSettings",
    "TelegramChannelSettings",
    "EmailChannelSettings",
    "SignalChannelSettings",
    "WhatsAppChannelSettings",
    "MessengerChannelSettings",
    "ControllerRuntimeConfig",
    "RuntimeResourceLimitsConfig",
    "SmokeArchiveLocalConfig",
    "SmokeArchiveS3Config",
    "SmokeArchiveUploadConfig",
    "PaperSmokeJsonSyncLocalConfig",
    "PaperSmokeJsonSyncS3Config",
    "PaperSmokeJsonSyncConfig",
    "CoreReportingConfig",
    "CoreConfig",
    "AlertThrottleConfig",
    "ServiceTokenConfig",
    "AlertAuditConfig",
    "DecisionJournalConfig",
    "MetricsServiceTlsConfig",
    "MetricsServiceConfig",
    "RiskServiceConfig",
    "RiskDecisionLogConfig",
    "SecurityBaselineConfig",
    "SecurityBaselineSigningConfig",
]
