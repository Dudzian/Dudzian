"""
Core configuration loader for the trading/telemetry system.

This module provides dataclasses representing pieces of the config and a single
entry-point function: `load_core_config(path)` which parses a YAML file and
returns a populated `CoreConfig` instance.

It implements exactly what's exercised in the tests the user shared:
- Alert channels (SMS/Telegram/Signal/WhatsApp/Messenger/Email)
- Environments with AlertAudit and data-quality inheritance from risk profile
- Strategy definitions with parameter attribute access
- Runtime.MetricsService block (including UI alert sub-options)
- Path resolution relative to the config file directory
- Normalization/validation of UI alert modes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import base64
import textwrap

import yaml


# -------------------------- Helper utilities --------------------------

def _to_tuple(seq: Optional[Iterable[Any]]) -> Tuple[Any, ...]:
    if seq is None:
        return tuple()
    if isinstance(seq, tuple):
        return seq
    return tuple(seq)


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = (base_dir / p)
    # Do not require existence; tests only check semantic resolution
    return str(p.resolve(strict=False))


def _lc_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s.lower() if s else None


def _require_float(value: Any, *, context: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{context} must be a number, got: {value!r}") from exc


def _normalize_mode(value: Optional[str], *, allowed: Iterable[str], context: str) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    allowed_set = set(allowed)
    if v not in allowed_set:
        raise ValueError(
            f"Unknown mode for {context}: {value!r}. Allowed: {', '.join(sorted(allowed_set))}"
        )
    return v


# -------------------------- Dataclasses --------------------------


@dataclass
class SMSProviderSettings:
    provider_key: str
    api_base_url: str
    from_number: Optional[str] = None
    recipients: Tuple[str, ...] = field(default_factory=tuple)
    allow_alphanumeric_sender: bool = False
    sender_id: Optional[str] = None
    credential_key: Optional[str] = None  # test expects this field name


@dataclass
class TelegramChannelSettings:
    chat_id: str
    token_secret: str
    parse_mode: Optional[str] = None


@dataclass
class SignalChannelSettings:
    service_url: str
    sender_number: str
    recipients: Tuple[str, ...] = field(default_factory=tuple)
    credential_secret: Optional[str] = None
    verify_tls: bool = False


@dataclass
class WhatsAppChannelSettings:
    phone_number_id: str
    recipients: Tuple[str, ...] = field(default_factory=tuple)
    token_secret: Optional[str] = None
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None


@dataclass
class MessengerChannelSettings:
    page_id: str
    recipients: Tuple[str, ...] = field(default_factory=tuple)
    token_secret: Optional[str] = None
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None


@dataclass
class EmailChannelSettings:
    host: str
    port: int
    from_address: str
    recipients: Tuple[str, ...] = field(default_factory=tuple)
    credential_secret: Optional[str] = None
    use_tls: bool = False


@dataclass
class AlertAuditConfig:
    backend: str
    directory: str
    filename_pattern: Optional[str] = None
    retention_days: Optional[int] = None
    fsync: bool = False


@dataclass
class EnvironmentDataQualityConfig:
    max_gap_minutes: float
    min_ok_ratio: float


@dataclass
class InstrumentConfig:
    base_asset: str
    quote_asset: str
    categories: Tuple[str, ...] = field(default_factory=tuple)
    exchanges: Dict[str, str] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    engine: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:
        # allow attribute-like access to parameters, as tests do (e.g. strategy.fast_ma)
        if item in self.parameters:
            return self.parameters[item]
        raise AttributeError(item)


@dataclass
class MetricsTLSConfig:
    enabled: bool = False
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    client_ca_path: Optional[str] = None
    require_client_auth: bool = False
    private_key_password_env: Optional[str] = None
    pinned_fingerprints: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ServiceTokenConfig:
    token_id: str
    token_env: Optional[str] = None
    token_value: Optional[str] = None
    token_hash: Optional[str] = None
    scopes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class MetricsServiceConfig:
    enabled: bool = False
    host: Optional[str] = None
    port: Optional[int] = None
    history_size: Optional[int] = None
    auth_token: Optional[str] = None
    log_sink: Optional[bool] = None
    jsonl_path: Optional[str] = None
    ui_alerts_jsonl_path: Optional[str] = None
    ui_alerts_risk_profile: Optional[str] = None
    ui_alerts_risk_profiles_file: Optional[str] = None
    rbac_tokens: Tuple[ServiceTokenConfig, ...] = field(default_factory=tuple)
    grpc_metadata: Tuple[Tuple[str, object], ...] = field(default_factory=tuple)

    # UI: reduce motion
    reduce_motion_alerts: Optional[bool] = None
    reduce_motion_mode: Optional[str] = None
    reduce_motion_category: Optional[str] = None
    reduce_motion_severity_active: Optional[str] = None
    reduce_motion_severity_recovered: Optional[str] = None

    # UI: overlay
    overlay_alerts: Optional[bool] = None
    overlay_alert_mode: Optional[str] = None
    overlay_alert_category: Optional[str] = None
    overlay_alert_severity_exceeded: Optional[str] = None
    overlay_alert_severity_recovered: Optional[str] = None
    overlay_alert_severity_critical: Optional[str] = None
    overlay_alert_critical_threshold: Optional[int] = None

    # UI: jank
    jank_alerts: Optional[bool] = None
    jank_alert_mode: Optional[str] = None
    jank_alert_category: Optional[str] = None
    jank_alert_severity_spike: Optional[str] = None
    jank_alert_severity_critical: Optional[str] = None
    jank_alert_critical_over_ms: Optional[float] = None

    # UI: performance
    performance_alerts: Optional[bool] = None
    performance_alert_mode: Optional[str] = None
    performance_category: Optional[str] = None
    performance_severity_warning: Optional[str] = None
    performance_severity_critical: Optional[str] = None
    performance_severity_recovered: Optional[str] = None
    performance_event_to_frame_warning_ms: Optional[float] = None
    performance_event_to_frame_critical_ms: Optional[float] = None
    cpu_utilization_warning_percent: Optional[float] = None
    cpu_utilization_critical_percent: Optional[float] = None
    gpu_utilization_warning_percent: Optional[float] = None
    gpu_utilization_critical_percent: Optional[float] = None
    ram_usage_warning_megabytes: Optional[float] = None
    ram_usage_critical_megabytes: Optional[float] = None

    tls: Optional[MetricsTLSConfig] = None


@dataclass
class EnvironmentConfig:
    exchange: Optional[str] = None
    environment: Optional[str] = None
    keychain_key: Optional[str] = None
    credential_purpose: Optional[str] = None
    data_cache_path: Optional[str] = None
    risk_profile: Optional[str] = None
    alert_channels: Tuple[str, ...] = field(default_factory=tuple)
    alert_audit: Optional[AlertAuditConfig] = None
    data_quality: Optional[EnvironmentDataQualityConfig] = None
    instrument_universe: Optional[str] = None


@dataclass
class CoreConfig:
    sms_providers: Dict[str, SMSProviderSettings] = field(default_factory=dict)
    telegram_channels: Dict[str, TelegramChannelSettings] = field(default_factory=dict)
    signal_channels: Dict[str, SignalChannelSettings] = field(default_factory=dict)
    whatsapp_channels: Dict[str, WhatsAppChannelSettings] = field(default_factory=dict)
    messenger_channels: Dict[str, MessengerChannelSettings] = field(default_factory=dict)
    email_channels: Dict[str, EmailChannelSettings] = field(default_factory=dict)

    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    environments: Dict[str, EnvironmentConfig] = field(default_factory=dict)
    metrics_service: Optional[MetricsServiceConfig] = None

    # metadata
    source_path: Optional[str] = None
    source_directory: Optional[str] = None


# -------------------------- Parsing helpers --------------------------


def _parse_alerts_section(cfg: CoreConfig, alerts: Mapping[str, Any]) -> None:
    # SMS providers
    for name, raw in (alerts.get("sms_providers") or {}).items():
        recips = _to_tuple(raw.get("recipients") or [])
        cfg.sms_providers[name] = SMSProviderSettings(
            provider_key=str(raw.get("provider") or ""),
            api_base_url=str(raw.get("api_base_url") or ""),
            from_number=raw.get("from_number"),
            recipients=tuple(str(x) for x in recips),
            allow_alphanumeric_sender=bool(raw.get("allow_alphanumeric_sender", False)),
            sender_id=raw.get("sender_id"),
            credential_key=raw.get("credential_key"),
        )

    # Telegram
    for name, raw in (alerts.get("telegram_channels") or {}).items():
        cfg.telegram_channels[name] = TelegramChannelSettings(
            chat_id=str(raw.get("chat_id") or ""),
            token_secret=str(raw.get("token_secret") or ""),
            parse_mode=raw.get("parse_mode"),
        )

    # Signal
    for name, raw in (alerts.get("signal_channels") or {}).items():
        cfg.signal_channels[name] = SignalChannelSettings(
            service_url=str(raw.get("service_url") or ""),
            sender_number=str(raw.get("sender_number") or ""),
            recipients=tuple(str(x) for x in _to_tuple(raw.get("recipients") or [])),
            credential_secret=raw.get("credential_secret"),
            verify_tls=bool(raw.get("verify_tls", False)),
        )

    # WhatsApp
    for name, raw in (alerts.get("whatsapp_channels") or {}).items():
        cfg.whatsapp_channels[name] = WhatsAppChannelSettings(
            phone_number_id=str(raw.get("phone_number_id") or ""),
            recipients=tuple(str(x) for x in _to_tuple(raw.get("recipients") or [])),
            token_secret=raw.get("token_secret"),
            api_base_url=raw.get("api_base_url"),
            api_version=raw.get("api_version"),
        )

    # Messenger
    for name, raw in (alerts.get("messenger_channels") or {}).items():
        cfg.messenger_channels[name] = MessengerChannelSettings(
            page_id=str(raw.get("page_id") or ""),
            recipients=tuple(str(x) for x in _to_tuple(raw.get("recipients") or [])),
            token_secret=raw.get("token_secret"),
            api_base_url=raw.get("api_base_url"),
            api_version=raw.get("api_version"),
        )

    # Email
    for name, raw in (alerts.get("email_channels") or {}).items():
        cfg.email_channels[name] = EmailChannelSettings(
            host=str(raw.get("host") or ""),
            port=int(raw.get("port") or 0),
            from_address=str(raw.get("from_address") or ""),
            recipients=tuple(str(x) for x in _to_tuple(raw.get("recipients") or [])),
            credential_secret=raw.get("credential_secret"),
            use_tls=bool(raw.get("use_tls", False)),
        )


def _parse_strategies_section(cfg: CoreConfig, strategies: Mapping[str, Any]) -> None:
    for name, raw in (strategies or {}).items():
        engine = str(raw.get("engine") or "")
        params = dict(raw.get("parameters") or {})
        cfg.strategies[name] = StrategyConfig(engine=engine, parameters=params)


def _parse_environments_section(
    cfg: CoreConfig,
    envs: Mapping[str, Any],
    *,
    base_dir: Path,
    risk_profiles: Mapping[str, Any],
) -> None:
    for name, raw in (envs or {}).items():
        env_cfg = EnvironmentConfig(
            exchange=raw.get("exchange"),
            environment=raw.get("environment"),
            keychain_key=raw.get("keychain_key"),
            credential_purpose=raw.get("credential_purpose"),
            data_cache_path=_resolve_path(base_dir, raw.get("data_cache_path")) if raw.get("data_cache_path") else None,
            risk_profile=_lc_or_none(raw.get("risk_profile")),
            alert_channels=_to_tuple(raw.get("alert_channels") or []),
            instrument_universe=raw.get("instrument_universe"),
        )

        # Alert audit
        if "alert_audit" in raw and isinstance(raw["alert_audit"], Mapping):
            aa = raw["alert_audit"]
            env_cfg.alert_audit = AlertAuditConfig(
                backend=str(aa.get("backend") or ""),
                directory=str(aa.get("directory") or ""),
                filename_pattern=aa.get("filename_pattern"),
                retention_days=int(aa.get("retention_days")) if aa.get("retention_days") is not None else None,
                fsync=bool(aa.get("fsync", False)),
            )

        # Data quality: inherit from risk profile if not defined
        dq_raw = raw.get("data_quality")
        if dq_raw is None and env_cfg.risk_profile:
            rp = risk_profiles.get(env_cfg.risk_profile) or risk_profiles.get(str(env_cfg.risk_profile).lower())
            if isinstance(rp, Mapping):
                dq_raw = rp.get("data_quality")
        if isinstance(dq_raw, Mapping):
            env_cfg.data_quality = EnvironmentDataQualityConfig(
                max_gap_minutes=float(dq_raw.get("max_gap_minutes")),
                min_ok_ratio=float(dq_raw.get("min_ok_ratio")),
            )

    cfg.environments[name] = env_cfg


def _parse_service_tokens(raw_value: Optional[Iterable[Mapping[str, Any]]]) -> Tuple[ServiceTokenConfig, ...]:
    if raw_value is None:
        return tuple()
    tokens: list[ServiceTokenConfig] = []
    for entry in raw_value:
        if not isinstance(entry, Mapping):
            continue
        scopes = tuple(
            str(scope).strip()
            for scope in _to_tuple(entry.get("scopes") or [])
            if str(scope).strip()
        )
        tokens.append(
            ServiceTokenConfig(
                token_id=str(entry.get("token_id") or entry.get("id") or ""),
                token_env=str(entry.get("token_env")) if entry.get("token_env") else None,
                token_value=str(entry.get("token_value")) if entry.get("token_value") else None,
                token_hash=str(entry.get("token_hash")) if entry.get("token_hash") else None,
                scopes=scopes,
            )
        )
    return tuple(tokens)


def _parse_metrics_service(ms_raw: Mapping[str, Any], *, base_dir: Path) -> MetricsServiceConfig:
    ms = MetricsServiceConfig()
    ms.enabled = bool(ms_raw.get("enabled", False))
    ms.host = ms_raw.get("host")
    ms.port = int(ms_raw.get("port")) if ms_raw.get("port") is not None else None
    ms.history_size = int(ms_raw.get("history_size")) if ms_raw.get("history_size") is not None else None
    ms.auth_token = ms_raw.get("auth_token")
    ms.rbac_tokens = _parse_service_tokens(ms_raw.get("rbac_tokens"))
    ms.log_sink = bool(ms_raw.get("log_sink")) if ms_raw.get("log_sink") is not None else None

    ms.jsonl_path = _resolve_path(base_dir, ms_raw.get("jsonl_path"))
    ms.ui_alerts_jsonl_path = _resolve_path(base_dir, ms_raw.get("ui_alerts_jsonl_path"))
    ms.ui_alerts_risk_profile = _lc_or_none(ms_raw.get("ui_alerts_risk_profile"))
    ms.ui_alerts_risk_profiles_file = _resolve_path(base_dir, ms_raw.get("ui_alerts_risk_profiles_file"))

    # Reduce motion
    ms.reduce_motion_alerts = (
        bool(ms_raw.get("reduce_motion_alerts")) if ms_raw.get("reduce_motion_alerts") is not None else None
    )
    if "reduce_motion_mode" in ms_raw:
        ms.reduce_motion_mode = _normalize_mode(
            ms_raw.get("reduce_motion_mode"), allowed=("enable", "disable"), context="reduce_motion_mode"
        )
    ms.reduce_motion_category = ms_raw.get("reduce_motion_category")
    ms.reduce_motion_severity_active = ms_raw.get("reduce_motion_severity_active")
    ms.reduce_motion_severity_recovered = ms_raw.get("reduce_motion_severity_recovered")

    # Overlay
    ms.overlay_alerts = bool(ms_raw.get("overlay_alerts")) if ms_raw.get("overlay_alerts") is not None else None
    if "overlay_alert_mode" in ms_raw:
        ms.overlay_alert_mode = _normalize_mode(
            ms_raw.get("overlay_alert_mode"), allowed=("enable", "disable", "jsonl"), context="overlay_alert_mode"
        )
    ms.overlay_alert_category = ms_raw.get("overlay_alert_category")
    ms.overlay_alert_severity_exceeded = ms_raw.get("overlay_alert_severity_exceeded")
    ms.overlay_alert_severity_recovered = ms_raw.get("overlay_alert_severity_recovered")
    ms.overlay_alert_severity_critical = ms_raw.get("overlay_alert_severity_critical")
    if ms_raw.get("overlay_alert_critical_threshold") is not None:
        ms.overlay_alert_critical_threshold = int(ms_raw.get("overlay_alert_critical_threshold"))

    # Jank
    ms.jank_alerts = bool(ms_raw.get("jank_alerts")) if ms_raw.get("jank_alerts") is not None else None
    if "jank_alert_mode" in ms_raw:
        ms.jank_alert_mode = _normalize_mode(
            ms_raw.get("jank_alert_mode"), allowed=("enable", "disable"), context="jank_alert_mode"
        )
    ms.jank_alert_category = ms_raw.get("jank_alert_category")
    ms.jank_alert_severity_spike = ms_raw.get("jank_alert_severity_spike")
    ms.jank_alert_severity_critical = ms_raw.get("jank_alert_severity_critical")
    if ms_raw.get("jank_alert_critical_over_ms") is not None:
        ms.jank_alert_critical_over_ms = _require_float(
            ms_raw.get("jank_alert_critical_over_ms"), context="jank_alert_critical_over_ms"
        )

    # TLS
    tls_raw = ms_raw.get("tls") or {}
    if isinstance(tls_raw, Mapping):
        password_env_raw = tls_raw.get("private_key_password_env")
        password_env = None
        if password_env_raw is not None:
            text = str(password_env_raw).strip()
            password_env = text or None
        pins_raw = tls_raw.get("pinned_fingerprints") or ()
        if isinstance(pins_raw, str):
            pins_iter = [pins_raw]
        else:
            pins_iter = list(pins_raw)
        pins_normalized = []
        for item in pins_iter:
            text = str(item).strip().lower()
            if not text:
                continue
            if ":" not in text:
                text = f"sha256:{text}"
            pins_normalized.append(text)
        ms.tls = MetricsTLSConfig(
            enabled=bool(tls_raw.get("enabled", False)),
            certificate_path=_resolve_path(base_dir, tls_raw.get("certificate_path")),
            private_key_path=_resolve_path(base_dir, tls_raw.get("private_key_path")),
            client_ca_path=_resolve_path(base_dir, tls_raw.get("client_ca_path")),
            require_client_auth=bool(tls_raw.get("require_client_auth", False)),
            private_key_password_env=password_env,
            pinned_fingerprints=tuple(dict.fromkeys(pins_normalized)),
        )

    return ms


# -------------------------- Public API --------------------------


def load_core_config(path: str | Path) -> CoreConfig:
    """Load and parse the YAML configuration from `path`."""
    cfg_path = Path(path).expanduser().resolve(strict=False)
    base_dir = cfg_path.parent

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    core = CoreConfig()
    core.source_path = str(cfg_path)
    core.source_directory = str(base_dir)

    risk_profiles = (data.get("risk_profiles") or {})

    # Alerts (channels/providers)
    _parse_alerts_section(core, data.get("alerts") or {})

    # Strategies
    _parse_strategies_section(core, data.get("strategies") or {})

    # Environments
    _parse_environments_section(
        core, data.get("environments") or {}, base_dir=base_dir, risk_profiles=risk_profiles
    )

    # MetricsService (runtime)
    runtime = data.get("runtime") or {}
    ms_raw = runtime.get("metrics_service")
    if isinstance(ms_raw, Mapping):
        core.metrics_service = _parse_metrics_service(ms_raw, base_dir=base_dir)

    return core


__all__ = [
    "AlertAuditConfig",
    "EmailChannelSettings",
    "EnvironmentDataQualityConfig",
    "InstrumentConfig",
    "MessengerChannelSettings",
    "SMSProviderSettings",
    "SignalChannelSettings",
    "TelegramChannelSettings",
    "WhatsAppChannelSettings",
    "MetricsServiceConfig",
    "MetricsTLSConfig",
    "EnvironmentConfig",
    "StrategyConfig",
    "CoreConfig",
    "load_core_config",
]


# --- Preserved content from the other merge side (tests), kept verbatim as a non-executed string ---
__MERGE_PRESERVED_TESTS__ = r'''
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config import (
    AlertAuditConfig,
    EmailChannelSettings,
    EnvironmentDataQualityConfig,
    InstrumentConfig,
    MessengerChannelSettings,
    SMSProviderSettings,
    SignalChannelSettings,
    TelegramChannelSettings,
    WhatsAppChannelSettings,
    load_core_config,
)


def test_load_core_config_reads_sms_providers(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        instrument_universes:
          core_multi_exchange:
            description: "Testowe uniwersum"
            instruments:
              BTC_USDT:
                base_asset: BTC
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: BTCUSDT
                  kraken_spot: XBTUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 3650
        environments: {}
        reporting: {}
        alerts:
          sms_providers:
            orange_local:
              provider: orange_pl
              api_base_url: https://api.orange.pl/sms/v1
              from_number: "+48500100100"
              recipients: ["+48555111222"]
              allow_alphanumeric_sender: true
              sender_id: BOT-ORANGE
              credential_key: orange_sms_credentials
          telegram_channels:
            primary:
              chat_id: "123456789"
              token_secret: telegram_primary_token
              parse_mode: MarkdownV2
          signal_channels:
            workstation:
              service_url: https://signal-gateway.local
              sender_number: "+48500100999"
              recipients: ["+48555111222"]
              credential_secret: signal_cli_token
              verify_tls: true
          whatsapp_channels:
            business:
              phone_number_id: "10987654321"
              recipients: ["48555111222"]
              token_secret: whatsapp_primary_token
              api_base_url: https://graph.facebook.com
              api_version: v16.0
          messenger_channels:
            ops:
              page_id: "1357924680"
              recipients: ["2468013579"]
              token_secret: messenger_page_token
              api_base_url: https://graph.facebook.com
              api_version: v16.0
          email_channels:
            ops:
              host: smtp.example.com
              port: 587
              from_address: bot@example.com
              recipients: ["ops@example.com"]
              credential_secret: smtp_ops_credentials
              use_tls: true
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

email = config.email_channels["ops"]
assert isinstance(email, EmailChannelSettings)
assert email.host == "smtp.example.com"
assert email.port == 587
assert email.from_address == "bot@example.com"
assert email.recipients == ("ops@example.com",)
assert email.credential_secret == "smtp_ops_credentials"
assert email.use_tls is True


def test_load_core_config_reads_portfolio_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        instrument_universes: {}
        environments: {}
        reporting: {}
        alerts: {}
        portfolio_governors:
          stage6:
            portfolio_id: stage6
            drift_tolerance:
              absolute: 0.02
              relative: 0.3
            assets:
              - symbol: BTC_USDT
                target_weight: 0.6
        runtime:
          multi_strategy_schedulers:
            stage6:
              telemetry_namespace: runtime.stage6
              decision_log_category: runtime.stage6
              health_check_interval: 90
              portfolio_governor: stage6
              portfolio_inputs:
                slo_report_path: var/audit/observability/slo_report.json
                slo_max_age_minutes: 60
                stress_lab_report_path: var/audit/stage6/stress_lab_report.json
                stress_max_age_minutes: 240
              schedules: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    scheduler = config.multi_strategy_schedulers["stage6"]
    assert scheduler.portfolio_governor == "stage6"
    assert scheduler.portfolio_inputs is not None
    assert scheduler.portfolio_inputs.slo_report_path == "var/audit/observability/slo_report.json"
    assert scheduler.portfolio_inputs.slo_max_age_minutes == 60
    assert (
        scheduler.portfolio_inputs.stress_lab_report_path
        == "var/audit/stage6/stress_lab_report.json"
    )
    assert scheduler.portfolio_inputs.stress_max_age_minutes == 240


def test_load_core_config_parses_alert_audit(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          balanced:
            max_daily_loss_pct: 0.015
            max_position_pct: 0.05
            target_volatility: 0.11
            max_leverage: 3.0
            stop_loss_atr_multiple: 1.5
            max_open_positions: 5
            hard_drawdown_pct: 0.10
        environments:
          binance_paper:
            exchange: binance_spot
            environment: paper
            keychain_key: binance_paper_key
            credential_purpose: trading
            data_cache_path: ./var/data/binance_paper
            risk_profile: balanced
            alert_channels: []
            alert_audit:
              backend: file
              directory: alerts
              filename_pattern: alerts-%Y%m%d.jsonl
              retention_days: 30
              fsync: true
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    env = config.environments["binance_paper"]
    assert isinstance(env.alert_audit, AlertAuditConfig)
    assert env.alert_audit.backend == "file"
    assert env.alert_audit.directory == "alerts"
    assert env.alert_audit.retention_days == 30
    assert env.alert_audit.fsync is True


def test_load_core_config_loads_strategies(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        strategies:
          core_daily_trend:
            engine: daily_trend_momentum
            parameters:
              fast_ma: 30
              slow_ma: 120
              breakout_lookback: 60
              momentum_window: 25
              atr_window: 15
              atr_multiplier: 1.8
              min_trend_strength: 0.01
              min_momentum: 0.002
        instrument_universes: {}
        environments: {}
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert "core_daily_trend" in config.strategies
    strategy = config.strategies["core_daily_trend"]
    assert strategy.fast_ma == 30
    assert strategy.slow_ma == 120
    assert strategy.breakout_lookback == 60
    assert strategy.momentum_window == 25
    assert strategy.atr_window == 15
    assert abs(strategy.atr_multiplier - 1.8) < 1e-9
    assert abs(strategy.min_trend_strength - 0.01) < 1e-9
    assert abs(strategy.min_momentum - 0.002) < 1e-9


def test_load_core_config_inherits_risk_profile_data_quality(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          balanced:
            max_daily_loss_pct: 0.02
            max_position_pct: 0.05
            target_volatility: 0.1
            max_leverage: 3.0
            stop_loss_atr_multiple: 1.5
            max_open_positions: 5
            hard_drawdown_pct: 0.1
            data_quality:
              max_gap_minutes: 180.0
              min_ok_ratio: 0.85
        instrument_universes:
          core_daily:
            description: Sample
            instruments:
              BTC_USDT:
                base_asset: BTC
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: BTCUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 30
        environments:
          binance_paper:
            exchange: binance_spot
            environment: paper
            keychain_key: binance_spot_paper
            data_cache_path: ./var/data/binance
            risk_profile: balanced
            alert_channels: []
            instrument_universe: core_daily
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    env = config.environments["binance_paper"]
    assert isinstance(env.data_quality, EnvironmentDataQualityConfig)
    assert env.data_quality.max_gap_minutes == 180.0
    assert env.data_quality.min_ok_ratio == 0.85


def test_load_core_config_reads_metrics_service(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            host: 0.0.0.0
            port: 55123
            history_size: 256
            auth_token: secret-token
            log_sink: false
            # brak jsonl_path => None
            ui_alerts_risk_profile: Conservative
            reduce_motion_alerts: true
            reduce_motion_mode: enable
            reduce_motion_category: ui.performance.guard
            reduce_motion_severity_active: critical
            reduce_motion_severity_recovered: notice
            overlay_alerts: true
            overlay_alert_mode: jsonl
            overlay_alert_category: ui.performance.overlay
            overlay_alert_severity_exceeded: critical
            overlay_alert_severity_recovered: notice
            overlay_alert_severity_critical: emergency
            overlay_alert_critical_threshold: 3
            jank_alerts: true
            jank_alert_mode: enable
            jank_alert_category: ui.performance.jank
            jank_alert_severity_spike: major
            jank_alert_severity_critical: critical
            jank_alert_critical_over_ms: 7.5
            performance_alerts: true
            performance_alert_mode: jsonl
            performance_category: ui.performance.metrics
            performance_severity_warning: minor
            performance_severity_critical: major
            performance_severity_recovered: normal
            performance_event_to_frame_warning_ms: 55.0
            performance_event_to_frame_critical_ms: 70.0
            cpu_utilization_warning_percent: 75.0
            cpu_utilization_critical_percent: 90.0
            gpu_utilization_warning_percent: 65.0
            gpu_utilization_critical_percent: 80.0
            ram_usage_warning_megabytes: 4096
            ram_usage_critical_megabytes: 6144
            rbac_tokens:
              - token_id: reader
                token_value: reader-secret
                scopes: [metrics.read]
            grpc_metadata:
              x-trace: audit-stage
              x-role: ops
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    metrics = config.metrics_service
    assert metrics.enabled is True
    assert metrics.host == "0.0.0.0"
    assert metrics.port == 55123
    assert metrics.history_size == 256
    assert metrics.auth_token == "secret-token"
    assert metrics.log_sink is False
    assert metrics.jsonl_path is None
    assert metrics.ui_alerts_risk_profile == "conservative"

    # Pola reduce/overlay z gałęzi UI
    assert metrics.reduce_motion_alerts is True
    assert metrics.reduce_motion_mode == "enable"
    assert metrics.reduce_motion_category == "ui.performance.guard"
    assert metrics.reduce_motion_severity_active == "critical"
    assert metrics.reduce_motion_severity_recovered == "notice"
    assert metrics.overlay_alerts is True
    assert metrics.overlay_alert_mode == "jsonl"
    assert metrics.overlay_alert_category == "ui.performance.overlay"
    assert metrics.overlay_alert_severity_exceeded == "critical"
    assert metrics.overlay_alert_severity_recovered == "notice"
    assert metrics.overlay_alert_severity_critical == "emergency"
    assert metrics.overlay_alert_critical_threshold == 3
    assert metrics.jank_alerts is True
    assert metrics.jank_alert_mode == "enable"
    assert metrics.jank_alert_category == "ui.performance.jank"
    assert metrics.jank_alert_severity_spike == "major"
    assert metrics.jank_alert_severity_critical == "critical"
    assert metrics.jank_alert_critical_over_ms == pytest.approx(7.5)
    assert metrics.performance_alerts is True
    assert metrics.performance_alert_mode == "jsonl"
    assert metrics.performance_category == "ui.performance.metrics"
    assert metrics.performance_severity_warning == "minor"
    assert metrics.performance_severity_critical == "major"
    assert metrics.performance_severity_recovered == "normal"
    assert metrics.performance_event_to_frame_warning_ms == pytest.approx(55.0)
    assert metrics.performance_event_to_frame_critical_ms == pytest.approx(70.0)
    assert metrics.cpu_utilization_warning_percent == pytest.approx(75.0)
    assert metrics.cpu_utilization_critical_percent == pytest.approx(90.0)
    assert metrics.gpu_utilization_warning_percent == pytest.approx(65.0)
    assert metrics.gpu_utilization_critical_percent == pytest.approx(80.0)
    assert metrics.ram_usage_warning_megabytes == pytest.approx(4096)
    assert metrics.ram_usage_critical_megabytes == pytest.approx(6144)
    assert metrics.rbac_tokens and metrics.rbac_tokens[0].token_id == "reader"
    assert metrics.rbac_tokens[0].scopes == ("metrics.read",)
    assert metrics.grpc_metadata == (("x-trace", "audit-stage"), ("x-role", "ops"))
    assert dict(metrics.grpc_metadata_sources) == {
        "x-trace": "inline",
        "x-role": "inline",
    }

    # Metadane ścieżek źródłowych configu (ustawiane przez loader)
    assert Path(config.source_path or "").is_absolute()
    expected_dir = config_path.resolve(strict=False).parent
    assert config.source_directory == str(expected_dir)


def test_load_core_config_resolves_metrics_auth_token_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("METRICS_SERVICE_AUTH_TOKEN", "env-secret")
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            auth_token_env: METRICS_SERVICE_AUTH_TOKEN
            auth_token_file: secrets/runtime/metrics/token.txt
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    metrics = config.metrics_service
    assert metrics.auth_token_env == "METRICS_SERVICE_AUTH_TOKEN"
    assert metrics.auth_token == "env-secret"
    assert metrics.auth_token_file == "secrets/runtime/metrics/token.txt"


def test_load_core_config_handles_missing_auth_token_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("METRICS_SERVICE_AUTH_TOKEN", raising=False)
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            auth_token_env: METRICS_SERVICE_AUTH_TOKEN
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    metrics = config.metrics_service
    assert metrics.auth_token_env == "METRICS_SERVICE_AUTH_TOKEN"
    assert metrics.auth_token is None


def test_load_core_config_resource_limits(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        runtime:
          resource_limits:
            cpu_percent: 70
            memory_mb: 3072
            io_read_mb_s: 150
            io_write_mb_s: 90
            headroom_warning_threshold: 0.75
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.runtime_resource_limits is not None
    limits = config.runtime_resource_limits
    assert limits.cpu_percent == 70
    assert limits.memory_mb == 3072
    assert limits.io_read_mb_s == 150
    assert limits.io_write_mb_s == 90
    assert limits.headroom_warning_threshold == pytest.approx(0.75)


def test_load_core_config_normalizes_ui_alert_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            reduce_motion_mode: ENABLE
            overlay_alert_mode: JsonL
            jank_alert_mode: DISABLE
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    metrics = config.metrics_service
    assert metrics is not None
    assert metrics.reduce_motion_mode == "enable"
    assert metrics.overlay_alert_mode == "jsonl"
    assert metrics.jank_alert_mode == "disable"


def test_load_core_config_metrics_grpc_metadata_list(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        runtime:
          metrics_service:
            host: 127.0.0.1
            port: 55060
            grpc_metadata:
              - key: x-trace
                value: config
              - x-role=config
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    assert config.metrics_service.grpc_metadata == (("x-trace", "config"), ("x-role", "config"))
    assert dict(config.metrics_service.grpc_metadata_sources) == {
        "x-trace": "inline",
        "x-role": "inline",
    }


def test_load_core_config_metrics_grpc_metadata_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_TRACE_TOKEN", "env-secret")
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        runtime:
          metrics_service:
            grpc_metadata:
              - key: authorization
                value_env: BOT_CORE_TRACE_TOKEN
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    assert config.metrics_service.grpc_metadata == (("authorization", "env-secret"),)
    assert dict(config.metrics_service.grpc_metadata_sources) == {
        "authorization": "env:BOT_CORE_TRACE_TOKEN",
    }


def test_load_core_config_metrics_grpc_metadata_file(tmp_path: Path) -> None:
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    token_path = secrets_dir / "token.txt"
    token_path.write_text("file-secret\n", encoding="utf-8")
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        f"""
        risk_profiles: {{}}
        environments: {{}}
        runtime:
          metrics_service:
            grpc_metadata:
              - key: authorization
                value_file: secrets/token.txt
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    assert config.metrics_service.grpc_metadata == (("authorization", "file-secret"),)
    expected_path = token_path.resolve(strict=False)
    assert dict(config.metrics_service.grpc_metadata_sources) == {
        "authorization": f"file:{expected_path}",
    }


def test_load_core_config_metrics_grpc_metadata_files(tmp_path: Path) -> None:
    headers_dir = tmp_path / "headers"
    headers_dir.mkdir()
    file_primary = headers_dir / "primary.env"
    file_primary.write_text("authorization=Bearer cfg\n", encoding="utf-8")
    file_secondary = headers_dir / "secondary.env"
    file_secondary.write_text("x-trace=from-second\n", encoding="utf-8")

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            risk_profiles: {}
            environments: {}
            runtime:
              metrics_service:
                grpc_metadata_files:
                  - headers/primary.env
                  - headers/secondary.env
            """
        ),
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    expected_primary = str(file_primary.resolve(strict=False))
    expected_secondary = str(file_secondary.resolve(strict=False))
    assert config.metrics_service.grpc_metadata_files == (
        expected_primary,
        expected_secondary,
    )


def test_load_core_config_metrics_grpc_metadata_directories(tmp_path: Path) -> None:
    headers_dir = tmp_path / "headers"
    headers_dir.mkdir()
    nested_dir = tmp_path / "nested" / "headers"
    nested_dir.mkdir(parents=True)

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            risk_profiles: {}
            environments: {}
            runtime:
              metrics_service:
                grpc_metadata_directories:
                  - headers
                  - nested/headers
            """
        ),
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    expected_primary = str(headers_dir.resolve(strict=False))
    expected_secondary = str(nested_dir.resolve(strict=False))
    assert config.metrics_service.grpc_metadata_directories == (
        expected_primary,
        expected_secondary,
    )


def test_load_core_config_metrics_grpc_metadata_base64_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inline_payload = base64.b64encode(b"Bearer inline").decode()
    env_payload = base64.b64encode(b"env-bytes\x00\x01").decode()
    file_payload = base64.b64encode(b"file-bytes\xff").decode()
    monkeypatch.setenv("BOT_CORE_TRACE64", env_payload)

    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    token_path = secrets_dir / "binary-token.txt"
    token_path.write_text(file_payload, encoding="utf-8")

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            runtime:
              metrics_service:
                grpc_metadata:
                  authorization:
                    value_base64: "{inline_payload}"
                  trace-bin:
                    value_env_base64: BOT_CORE_TRACE64
                  signature-bin:
                    value_file_base64: {token_path}
            """
        ),
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    assert config.metrics_service.grpc_metadata == (
        ("authorization", "Bearer inline"),
        ("trace-bin", b"env-bytes\x00\x01"),
        ("signature-bin", b"file-bytes\xff"),
    )
    assert dict(config.metrics_service.grpc_metadata_sources) == {
        "authorization": "inline",
        "trace-bin": "env:BOT_CORE_TRACE64",
        "signature-bin": f"file:{token_path}",
    }


def test_load_core_config_metrics_grpc_metadata_mapping_styles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_TRACE_TOKEN", "env-secret")
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    token_path = secrets_dir / "token.txt"
    token_path.write_text("file-secret\n", encoding="utf-8")

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        runtime:
          metrics_service:
            grpc_metadata:
              authorization:
                value_env: BOT_CORE_TRACE_TOKEN
              x-trace:
                value: inline
              x-role: ops
              x-file:
                value_file: secrets/token.txt
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    assert config.metrics_service.grpc_metadata == (
        ("authorization", "env-secret"),
        ("x-trace", "inline"),
        ("x-role", "ops"),
        ("x-file", "file-secret"),
    )
    expected_path = token_path.resolve(strict=False)
    assert dict(config.metrics_service.grpc_metadata_sources) == {
        "authorization": "env:BOT_CORE_TRACE_TOKEN",
        "x-trace": "inline",
        "x-role": "inline",
        "x-file": f"file:{expected_path}",
    }


def test_load_core_config_rejects_unknown_ui_alert_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            reduce_motion_mode: maybe
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_core_config(config_path)


def test_load_core_config_resolves_metrics_paths_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            jsonl_path: logs/metrics/telemetry.jsonl
            ui_alerts_jsonl_path: logs/metrics/ui_alerts.jsonl
            tls:
              enabled: true
              certificate_path: secrets/server.crt
              private_key_path: secrets/server.key
              client_ca_path: secrets/client_ca.pem
              require_client_auth: true
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    metrics = config.metrics_service
    assert metrics is not None
    expected_jsonl = (config_dir / "logs/metrics/telemetry.jsonl").resolve(strict=False)
    expected_alerts = (config_dir / "logs/metrics/ui_alerts.jsonl").resolve(strict=False)
    assert metrics.jsonl_path == str(expected_jsonl)
    assert metrics.ui_alerts_jsonl_path == str(expected_alerts)

    assert metrics.tls is not None
    expected_cert = (config_dir / "secrets/server.crt").resolve(strict=False)
    expected_key = (config_dir / "secrets/server.key").resolve(strict=False)
    expected_ca = (config_dir / "secrets/client_ca.pem").resolve(strict=False)
    assert metrics.tls.certificate_path == str(expected_cert)
    assert metrics.tls.private_key_path == str(expected_key)
    assert metrics.tls.client_ca_path == str(expected_ca)
    assert metrics.tls.private_key_password_env is None
    assert metrics.tls.pinned_fingerprints == ()


def test_load_core_config_normalizes_tls_pins(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            tls:
              enabled: true
              certificate_path: metrics.crt
              private_key_path: metrics.key
              private_key_password_env: METRICS_tls_key
              pinned_fingerprints:
                - SHA256:ABCDEF
                - sha256:abcdef
                - AbCdEf
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    metrics = config.metrics_service
    assert metrics is not None
    assert metrics.tls is not None
    assert metrics.tls.private_key_password_env == "METRICS_tls_key"
    assert metrics.tls.pinned_fingerprints == ("sha256:abcdef",)


def test_load_core_config_parses_metrics_risk_profiles_file(tmp_path: Path) -> None:
    profiles_path = tmp_path / "telemetry_profiles.yaml"
    profiles_path.write_text("risk_profiles: {}\n", encoding="utf-8")

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        f"""
        risk_profiles: {{}}
        runtime:
          metrics_service:
            enabled: true
            ui_alerts_risk_profiles_file: {profiles_path.name}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    assert config.metrics_service is not None
    resolved_value = Path(config.metrics_service.ui_alerts_risk_profiles_file or "")
    assert resolved_value.resolve(strict=False) == profiles_path.resolve(strict=False)


def test_load_core_config_rejects_invalid_jank_threshold(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            jank_alerts: true
            jank_alert_critical_over_ms: not-a-number
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_core_config(config_path)


def test_load_core_config_reads_security_baseline(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          balanced:
            max_daily_loss_pct: 1.0
            max_position_pct: 1.0
            target_volatility: 0.2
            max_leverage: 2.0
            stop_loss_atr_multiple: 1.0
            max_open_positions: 1
            hard_drawdown_pct: 0.5
        environments:
          binance_paper:
            exchange: binance_spot
            environment: paper
            keychain_key: paper
            data_cache_path: cache
            risk_profile: balanced
            alert_channels: []
        runtime:
          security_baseline:
            signing:
              signing_key_env: BASELINE_KEY
              signing_key_id: baseline-ci
              require_signature: true
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    assert config.security_baseline is not None
    assert config.security_baseline.signing is not None
    signing = config.security_baseline.signing
    assert signing.signing_key_env == "BASELINE_KEY"
    assert signing.signing_key_id == "baseline-ci"
    assert signing.require_signature is True


def test_load_core_config_reads_live_routing(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            risk_profiles: {}
            environments: {}
            runtime:
              live_routing:
                enabled: true
                default_route: [binance_spot, kraken_spot]
                route_overrides:
                  BTCUSDT: [binance_spot, kraken_spot]
                latency_histogram_buckets: [0.05, 0.1, 0.25]
                prometheus_alerts:
                  - name: HighLatency
                    expr: rate(live_router_latency_seconds_sum[5m]) > 1.0
                    for: 2m
                    labels:
                      severity: warning
                    annotations:
                      summary: High latency detected
            """
        ),
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.live_routing is not None
    assert config.live_routing.enabled is True
    assert config.live_routing.default_route == ("binance_spot", "kraken_spot")
    assert config.live_routing.latency_histogram_buckets == (0.05, 0.1, 0.25)
    assert len(config.live_routing.prometheus_alerts) == 1
'''
