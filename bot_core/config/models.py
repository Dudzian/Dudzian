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
class EnvironmentDataSourceConfig:
    """Parametry źródła danych środowiska (cache + snapshot REST)."""

    enable_snapshots: bool = True
    cache_namespace: str | None = None


@dataclass(slots=True)
class EnvironmentReportStorageConfig:
    """Opisuje sposób przechowywania raportów operacyjnych środowiska."""

    backend: str
    directory: str | None = None
    filename_pattern: str = "reports-%Y%m%d.json"
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
    auth_token_env: str | None = None
    auth_token_file: str | None = None
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
    grpc_metadata: Sequence[tuple[str, str | bytes]] = field(default_factory=tuple)
    grpc_metadata_files: Sequence[str] = field(default_factory=tuple)
    grpc_metadata_directories: Sequence[str] = field(default_factory=tuple)
    grpc_metadata_sources: Mapping[str, str] = field(default_factory=dict)

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

    performance_alerts: bool = False
    performance_alert_mode: str | None = None
    performance_category: str = "ui.performance"
    performance_severity_warning: str = "warning"
    performance_severity_critical: str = "critical"
    performance_severity_recovered: str = "info"
    performance_event_to_frame_warning_ms: float | None = 45.0
    performance_event_to_frame_critical_ms: float | None = 60.0
    cpu_utilization_warning_percent: float | None = 85.0
    cpu_utilization_critical_percent: float | None = 95.0
    gpu_utilization_warning_percent: float | None = None
    gpu_utilization_critical_percent: float | None = None
    ram_usage_warning_megabytes: float | None = None
    ram_usage_critical_megabytes: float | None = None


@dataclass(slots=True)
class PrometheusAlertRuleConfig:
    """Definicja reguły alertu Prometheusa powiązanego z routerem live."""
    name: str
    expr: str
    for_duration: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    annotations: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LiveRoutingConfig:
    """Konfiguracja routera egzekucji live."""
    enabled: bool = False
    default_route: Sequence[str] = field(default_factory=tuple)
    route_overrides: Mapping[str, Sequence[str]] = field(default_factory=dict)
    latency_histogram_buckets: Sequence[float] = field(default_factory=tuple)
    prometheus_alerts: Sequence[PrometheusAlertRuleConfig] = field(default_factory=tuple)


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
class PortfolioDecisionLogConfig:
    """Konfiguracja dziennika decyzji PortfolioGovernora."""
    enabled: bool = True
    path: str | None = None
    max_entries: int = 512
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


@dataclass(slots=True)
class SLOThresholdConfig:
    """Definicja pojedynczego SLO monitorowanego w Stage5."""
    name: str
    metric: str
    objective: float
    comparator: str = "<="
    window_minutes: float = 1440.0
    aggregation: str = "average"
    label_filters: Mapping[str, str] = field(default_factory=dict)
    min_samples: int = 1


@dataclass(slots=True)
class KeyRotationEntryConfig:
    """Pojedynczy wpis w konfiguracji rotacji kluczy."""
    key: str
    purpose: str
    interval_days: float | None = None
    warn_within_days: float | None = None


@dataclass(slots=True)
class KeyRotationConfig:
    """Parametry modułu rotacji kluczy i przypomnień."""
    registry_path: str
    default_interval_days: float = 90.0
    default_warn_within_days: float = 14.0
    entries: Sequence[KeyRotationEntryConfig] = field(default_factory=tuple)
    signing_key_env: str | None = None
    signing_key_path: str | None = None
    signing_key_value: str | None = None
    signing_key_id: str | None = None
    audit_directory: str = "var/audit/keys"


@dataclass(slots=True)
class PortfolioGovernorScoringWeights:
    """Wagi komponentów scoringu PortfolioGovernora."""
    alpha: float = 1.0
    cost: float = 1.0
    slo: float = 1.0
    risk: float = 0.5


@dataclass(slots=True)
class PortfolioGovernorStrategyConfig:
    """Konfiguracja strategii zarządzanej przez PortfolioGovernora."""
    baseline_weight: float
    min_weight: float = 0.0
    max_weight: float = 1.0
    baseline_max_signals: int | None = None
    max_signal_factor: float = 1.0
    risk_profile: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class PortfolioGovernorConfig:
    """Ustawienia PortfolioGovernora sterującego alokacją między strategiami (Stage5)."""
    enabled: bool = False
    rebalance_interval_minutes: float = 15.0
    smoothing: float = 0.5
    scoring: PortfolioGovernorScoringWeights = field(default_factory=PortfolioGovernorScoringWeights)
    strategies: Mapping[str, PortfolioGovernorStrategyConfig] = field(default_factory=dict)
    default_baseline_weight: float = 0.25
    default_min_weight: float = 0.05
    default_max_weight: float = 0.5
    require_complete_metrics: bool = True
    min_score_threshold: float = 0.0
    default_cost_bps: float = 0.0
    max_signal_floor: int = 1


@dataclass(slots=True)
class MarketIntelSqliteConfig:
    """Źródło danych SQLite wykorzystywane przez agregator Market Intelligence."""
    path: str
    table: str = "market_metrics"
    symbol_column: str = "symbol"
    mid_price_column: str = "mid_price"
    depth_column: str = "avg_depth_usd"
    spread_column: str = "avg_spread_bps"
    funding_column: str = "funding_rate_bps"
    sentiment_column: str = "sentiment_score"
    volatility_column: str = "realized_volatility"
    weight_column: str | None = "weight"


@dataclass(slots=True)
class MarketIntelConfig:
    """Konfiguracja sekcji Market Intelligence Stage6."""
    enabled: bool = False
    output_directory: str = "data/stage6/metrics"
    manifest_path: str | None = None
    sqlite: MarketIntelSqliteConfig | None = None
    required_symbols: Sequence[str] = field(default_factory=tuple)
    default_weight: float = 1.0


@dataclass(slots=True)
class StressLabDatasetConfig:
    """Źródło danych używane przez Stress Lab dla konkretnego instrumentu."""
    symbol: str
    metrics_path: str
    weight: float = 1.0
    allow_synthetic: bool = False


@dataclass(slots=True)
class StressLabShockConfig:
    """Opis pojedynczego szoku w scenariuszu Stress Lab."""
    type: str
    intensity: float = 1.0
    duration_minutes: float | None = None
    notes: str | None = None


@dataclass(slots=True)
class StressLabThresholdsConfig:
    """Progi oceny scenariuszy Stress Lab."""
    max_liquidity_loss_pct: float = 0.65
    max_spread_increase_bps: float = 45.0
    max_volatility_increase_pct: float = 0.85
    max_sentiment_drawdown: float = 0.55
    max_funding_change_bps: float = 30.0
    max_latency_spike_ms: float = 160.0
    max_blackout_minutes: float = 45.0
    max_dispersion_bps: float = 55.0


@dataclass(slots=True)
class StressLabScenarioConfig:
    """Konfiguracja scenariusza Stress Lab dla wielu rynków."""
    name: str
    severity: str = "medium"
    markets: Sequence[str] = field(default_factory=tuple)
    shocks: Sequence[StressLabShockConfig] = field(default_factory=tuple)
    description: str | None = None
    threshold_overrides: StressLabThresholdsConfig | None = None


@dataclass(slots=True)
class StressLabConfig:
    """Konfiguracja modułu Stress Lab w Stage6."""
    enabled: bool = False
    require_success: bool = True
    report_directory: str = "var/audit/stage6/stress_lab"
    signing_key_env: str | None = None
    signing_key_path: str | None = None
    signing_key_id: str | None = None
    datasets: Mapping[str, StressLabDatasetConfig] = field(default_factory=dict)
    scenarios: Sequence[StressLabScenarioConfig] = field(default_factory=tuple)
    thresholds: StressLabThresholdsConfig = field(default_factory=StressLabThresholdsConfig)


@dataclass(slots=True)
class ResilienceDrillThresholdsConfig:
    """Progi akceptacyjne dla drillów failover w Stage6."""
    max_latency_ms: float = 250.0
    max_error_rate: float = 0.05
    max_failover_duration_seconds: float = 120.0
    max_orders_failed: int = 0


@dataclass(slots=True)
class ResilienceDrillConfig:
    """Definicja pojedynczego drill'u failover."""
    name: str
    primary: str
    fallbacks: Sequence[str] = field(default_factory=tuple)
    dataset_path: str = ""
    thresholds: ResilienceDrillThresholdsConfig = field(default_factory=ResilienceDrillThresholdsConfig)
    description: str | None = None


@dataclass(slots=True)
class ResilienceConfig:
    """Konfiguracja modułu resilience & failover (Stage6)."""
    enabled: bool = False
    require_success: bool = True
    report_directory: str = "var/audit/stage6/resilience"
    signing_key_env: str | None = None
    signing_key_path: str | None = None
    signing_key_id: str | None = None
    drills: Sequence[ResilienceDrillConfig] = field(default_factory=tuple)


@dataclass(slots=True)
class ObservabilityConfig:
    """Konfiguracja rozszerzonej obserwowalności Stage5."""
    slo: Mapping[str, SLOThresholdConfig] = field(default_factory=dict)
    key_rotation: KeyRotationConfig | None = None


@dataclass(slots=True)
class DecisionOrchestratorThresholds:
    """Progi podejmowania decyzji przez DecisionOrchestrator."""
    max_cost_bps: float
    min_net_edge_bps: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_position_ratio: float
    max_open_positions: int
    max_latency_ms: float | None = None
    max_trade_notional: float | None = None


@dataclass(slots=True)
class DecisionStressTestConfig:
    """Parametry testów stresowych wykorzystywanych przed przejściem live."""
    cost_shock_bps: float = 0.0
    latency_spike_ms: float = 0.0
    slippage_multiplier: float = 1.0


@dataclass(slots=True)
class DecisionEngineTCOConfig:
    """Ścieżki raportów TCO wykorzystywanych przez DecisionOrchestrator."""

    report_paths: Sequence[str] = field(default_factory=tuple)
    reports: Sequence[str] | None = field(default=None, repr=False)
    require_at_startup: bool = False
    runtime_enabled: bool = False
    runtime_report_directory: str | None = None
    runtime_report_basename: str | None = None
    runtime_export_formats: Sequence[str] = field(default_factory=lambda: ("json",))
    runtime_flush_events: int | None = None
    runtime_clear_after_export: bool = False
    runtime_signing_key_env: str | None = None
    runtime_signing_key_id: str | None = None
    runtime_metadata: Mapping[str, object] = field(default_factory=dict)
    runtime_cost_limit_bps: float | None = None
    warn_report_age_hours: float | None = 24.0
    max_report_age_hours: float | None = 72.0

    def __post_init__(self) -> None:
        if self.reports and self.report_paths:
            raise ValueError(
                "DecisionEngineTCOConfig nie może otrzymać jednocześnie 'reports' i 'report_paths'"
            )

        paths_source: Sequence[str]
        if self.reports:
            paths_source = self.reports
        else:
            paths_source = self.report_paths

        normalized_paths = tuple(str(path) for path in paths_source if str(path).strip())
        self.report_paths = normalized_paths
        self.reports = normalized_paths

        if self.warn_report_age_hours is not None:
            self.warn_report_age_hours = float(self.warn_report_age_hours)
        if self.max_report_age_hours is not None:
            self.max_report_age_hours = float(self.max_report_age_hours)
        self.runtime_clear_after_export = bool(self.runtime_clear_after_export)


@dataclass(slots=True)
class DecisionEngineConfig:
    """Konfiguracja rozszerzonego decision engine'u (Etap 5)."""
    orchestrator: DecisionOrchestratorThresholds
    profile_overrides: Mapping[str, DecisionOrchestratorThresholds] = field(default_factory=dict)
    stress_tests: DecisionStressTestConfig | None = None
    min_probability: float = 0.0
    require_cost_data: bool = False
    penalty_cost_bps: float = 0.0
    tco: DecisionEngineTCOConfig | None = None


# --- AI ---------------------------------------------------------------------


@dataclass(slots=True)
class EnvironmentAIModelConfig:
    """Definicja modelu AI przypiętego do konkretnego symbolu."""

    symbol: str
    model_type: str
    path: str
    strategy: str | None = None
    risk_profile: str | None = None
    notional: float | None = None
    action: str | None = None


@dataclass(slots=True)
class EnvironmentAIConfig:
    """Konfiguracja integracji AIManagera w środowisku runtime."""

    enabled: bool = True
    model_dir: str | None = None
    threshold_bps: float = 5.0
    default_strategy: str | None = None
    default_risk_profile: str | None = None
    default_notional: float | None = None
    default_action: str = "enter"
    preload: Sequence[str] = field(default_factory=tuple)
    models: Sequence[EnvironmentAIModelConfig] = field(default_factory=tuple)


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
    offline_mode: bool = False
    data_source: EnvironmentDataSourceConfig | None = None
    report_storage: EnvironmentReportStorageConfig | None = None
    permission_profile: str | None = None
    required_permissions: Sequence[str] = field(default_factory=tuple)
    forbidden_permissions: Sequence[str] = field(default_factory=tuple)
    alert_throttle: AlertThrottleConfig | None = None
    alert_audit: AlertAuditConfig | None = None
    data_quality: EnvironmentDataQualityConfig | None = None
    decision_journal: DecisionJournalConfig | None = None
    default_strategy: str | None = None
    default_controller: str | None = None
    ai: EnvironmentAIConfig | None = None


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


@dataclass(slots=True)
class PermissionProfileConfig:
    """Zawiera listy minimalnych i zabronionych uprawnień API."""

    name: str
    required_permissions: Sequence[str] = field(default_factory=tuple)
    forbidden_permissions: Sequence[str] = field(default_factory=tuple)


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
    portfolio_governor: str | None = None
    portfolio_inputs: "PortfolioRuntimeInputsConfig" | None = None


@dataclass(slots=True)
class PortfolioRuntimeInputsConfig:
    """Ścieżki artefaktów wykorzystywanych przez PortfolioGovernora w runtime."""
    slo_report_path: str | None = None
    slo_max_age_minutes: int | None = None
    stress_lab_report_path: str | None = None
    stress_max_age_minutes: int | None = None


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


# --- Runtime (Stage6 – portfel) -----------------------------------------------

@dataclass(slots=True)
class PortfolioDriftToleranceConfig:
    """Parametry dryfu akceptowanego przez PortfolioGovernor."""
    absolute: float = 0.01
    relative: float = 0.25


@dataclass(slots=True)
class PortfolioRiskBudgetConfig:
    """Budżet ryzyka przypisywany do koszyka aktywów."""
    name: str
    max_var_pct: float | None = None
    max_drawdown_pct: float | None = None
    max_leverage: float | None = None
    severity: str = "warning"
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class PortfolioAssetConfig:
    """Konfiguracja aktywa zarządzanego przez PortfolioGovernor."""
    symbol: str
    target_weight: float
    min_weight: float | None = None
    max_weight: float | None = None
    max_volatility_pct: float | None = None
    min_liquidity_usd: float | None = None
    risk_budget: str | None = None
    notes: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class PortfolioSloOverrideConfig:
    """Reguła reagująca na statusy SLO w Observability Stage6."""
    slo_name: str
    apply_on: Sequence[str] = field(default_factory=lambda: ("warning", "breach"))
    weight_multiplier: float | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    severity: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    force_rebalance: bool = False


@dataclass(slots=True)
class PortfolioGovernorV6Config:
    """Deklaracja PortfolioGovernora Stage6 (zarządzanie portfelem)."""
    name: str
    portfolio_id: str
    drift_tolerance: PortfolioDriftToleranceConfig = field(default_factory=PortfolioDriftToleranceConfig)
    rebalance_cooldown_seconds: int = 900
    min_rebalance_value: float = 0.0
    min_rebalance_weight: float = 0.0
    assets: Sequence[PortfolioAssetConfig] = field(default_factory=tuple)
    risk_budgets: Mapping[str, PortfolioRiskBudgetConfig] = field(default_factory=dict)
    risk_overrides: Sequence[str] = field(default_factory=tuple)
    slo_overrides: Sequence[PortfolioSloOverrideConfig] = field(default_factory=tuple)
    market_intel_interval: str | None = None
    market_intel_lookback_bars: int = 168


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
    permission_profiles: Mapping[str, PermissionProfileConfig] = field(default_factory=dict)
    instrument_universes: Mapping[str, InstrumentUniverseConfig] = field(default_factory=dict)
    instrument_buckets: Mapping[str, InstrumentBucketConfig] = field(default_factory=dict)
    strategies: Mapping[str, DailyTrendMomentumStrategyConfig] = field(default_factory=dict)
    mean_reversion_strategies: Mapping[str, MeanReversionStrategyConfig] = field(default_factory=dict)
    volatility_target_strategies: Mapping[str, VolatilityTargetingStrategyConfig] = field(default_factory=dict)
    cross_exchange_arbitrage_strategies: Mapping[str, CrossExchangeArbitrageStrategyConfig] = field(default_factory=dict)
    multi_strategy_schedulers: Mapping[str, MultiStrategySchedulerConfig] = field(default_factory=dict)

    # Stage5
    portfolio_governors: Mapping[str, PortfolioGovernorConfig] = field(default_factory=dict)
    decision_engine: DecisionEngineConfig | None = None

    # Stage6
    portfolio_governor: PortfolioGovernorV6Config | None = None
    stress_lab: StressLabConfig | None = None
    resilience: ResilienceConfig | None = None

    reporting: CoreReportingConfig | None = None
    sms_providers: Mapping[str, SMSProviderSettings] = field(default_factory=dict)
    telegram_channels: Mapping[str, TelegramChannelSettings] = field(default_factory=dict)
    email_channels: Mapping[str, EmailChannelSettings] = field(default_factory=dict)
    signal_channels: Mapping[str, SignalChannelSettings] = field(default_factory=dict)
    whatsapp_channels: Mapping[str, WhatsAppChannelSettings] = field(default_factory=dict)
    messenger_channels: Mapping[str, MessengerChannelSettings] = field(default_factory=dict)
    runtime_controllers: Mapping[str, ControllerRuntimeConfig] = field(default_factory=dict)
    coverage_monitoring: CoverageMonitoringConfig | None = None
    live_routing: LiveRoutingConfig | None = None
    metrics_service: MetricsServiceConfig | None = None
    risk_service: RiskServiceConfig | None = None
    risk_decision_log: RiskDecisionLogConfig | None = None
    portfolio_decision_log: PortfolioDecisionLogConfig | None = None
    security_baseline: SecurityBaselineConfig | None = None
    observability: ObservabilityConfig | None = None
    runtime_resource_limits: RuntimeResourceLimitsConfig | None = None
    market_intel: MarketIntelConfig | None = None
    source_path: str | None = None
    source_directory: str | None = None


__all__ = [
    "EnvironmentConfig",
    "EnvironmentDataSourceConfig",
    "EnvironmentReportStorageConfig",
    "EnvironmentDataQualityConfig",
    "CoverageMonitorTargetConfig",
    "CoverageMonitoringConfig",
    "RiskProfileConfig",
    "InstrumentBackfillWindow",
    "InstrumentConfig",
    "InstrumentUniverseConfig",
    "InstrumentBucketConfig",
    "PermissionProfileConfig",
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
    "PortfolioGovernorConfig",
    "PortfolioAssetConfig",
    "PortfolioRiskBudgetConfig",
    "PortfolioDriftToleranceConfig",
    "PortfolioSloOverrideConfig",
    "PortfolioRuntimeInputsConfig",
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
    "LiveRoutingConfig",
    "PrometheusAlertRuleConfig",
    "RiskDecisionLogConfig",
    "PortfolioDecisionLogConfig",
    "SecurityBaselineConfig",
    "SecurityBaselineSigningConfig",
    "SLOThresholdConfig",
    "KeyRotationEntryConfig",
    "KeyRotationConfig",
    "ObservabilityConfig",
    "DecisionEngineConfig",
    "EnvironmentAIConfig",
    "EnvironmentAIModelConfig",
    "DecisionEngineTCOConfig",
    "DecisionOrchestratorThresholds",
    "DecisionStressTestConfig",
    "PortfolioGovernorStrategyConfig",
    "PortfolioGovernorScoringWeights",
    "MarketIntelConfig",
    "MarketIntelSqliteConfig",
    "StressLabConfig",
    "StressLabDatasetConfig",
    "StressLabScenarioConfig",
    "StressLabShockConfig",
    "StressLabThresholdsConfig",
    "ResilienceConfig",
    "ResilienceDrillConfig",
    "ResilienceDrillThresholdsConfig",
    "PortfolioGovernorV6Config",
]
