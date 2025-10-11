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
        raise ValueError(f"Unknown mode for {context}: {value!r}. Allowed: {', '.join(sorted(allowed_set))}")
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


def _parse_environments_section(cfg: CoreConfig, envs: Mapping[str, Any], *, base_dir: Path, risk_profiles: Mapping[str, Any]) -> None:
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


def _parse_metrics_service(ms_raw: Mapping[str, Any], *, base_dir: Path) -> MetricsServiceConfig:
    ms = MetricsServiceConfig()
    ms.enabled = bool(ms_raw.get("enabled", False))
    ms.host = ms_raw.get("host")
    ms.port = int(ms_raw.get("port")) if ms_raw.get("port") is not None else None
    ms.history_size = int(ms_raw.get("history_size")) if ms_raw.get("history_size") is not None else None
    ms.auth_token = ms_raw.get("auth_token")
    ms.log_sink = bool(ms_raw.get("log_sink")) if ms_raw.get("log_sink") is not None else None

    ms.jsonl_path = _resolve_path(base_dir, ms_raw.get("jsonl_path"))
    ms.ui_alerts_jsonl_path = _resolve_path(base_dir, ms_raw.get("ui_alerts_jsonl_path"))
    ms.ui_alerts_risk_profile = _lc_or_none(ms_raw.get("ui_alerts_risk_profile"))
    ms.ui_alerts_risk_profiles_file = _resolve_path(base_dir, ms_raw.get("ui_alerts_risk_profiles_file"))

    # Reduce motion
    ms.reduce_motion_alerts = bool(ms_raw.get("reduce_motion_alerts")) if ms_raw.get("reduce_motion_alerts") is not None else None
    if "reduce_motion_mode" in ms_raw:
        ms.reduce_motion_mode = _normalize_mode(ms_raw.get("reduce_motion_mode"), allowed=("enable", "disable"), context="reduce_motion_mode")
    ms.reduce_motion_category = ms_raw.get("reduce_motion_category")
    ms.reduce_motion_severity_active = ms_raw.get("reduce_motion_severity_active")
    ms.reduce_motion_severity_recovered = ms_raw.get("reduce_motion_severity_recovered")

    # Overlay
    ms.overlay_alerts = bool(ms_raw.get("overlay_alerts")) if ms_raw.get("overlay_alerts") is not None else None
    if "overlay_alert_mode" in ms_raw:
        ms.overlay_alert_mode = _normalize_mode(ms_raw.get("overlay_alert_mode"), allowed=("enable", "disable", "jsonl"), context="overlay_alert_mode")
    ms.overlay_alert_category = ms_raw.get("overlay_alert_category")
    ms.overlay_alert_severity_exceeded = ms_raw.get("overlay_alert_severity_exceeded")
    ms.overlay_alert_severity_recovered = ms_raw.get("overlay_alert_severity_recovered")
    ms.overlay_alert_severity_critical = ms_raw.get("overlay_alert_severity_critical")
    if ms_raw.get("overlay_alert_critical_threshold") is not None:
        ms.overlay_alert_critical_threshold = int(ms_raw.get("overlay_alert_critical_threshold"))

    # Jank
    ms.jank_alerts = bool(ms_raw.get("jank_alerts")) if ms_raw.get("jank_alerts") is not None else None
    if "jank_alert_mode" in ms_raw:
        ms.jank_alert_mode = _normalize_mode(ms_raw.get("jank_alert_mode"), allowed=("enable", "disable"), context="jank_alert_mode")
    ms.jank_alert_category = ms_raw.get("jank_alert_category")
    ms.jank_alert_severity_spike = ms_raw.get("jank_alert_severity_spike")
    ms.jank_alert_severity_critical = ms_raw.get("jank_alert_severity_critical")
    if ms_raw.get("jank_alert_critical_over_ms") is not None:
        ms.jank_alert_critical_over_ms = _require_float(ms_raw.get("jank_alert_critical_over_ms"), context="jank_alert_critical_over_ms")

    # TLS
    tls_raw = ms_raw.get("tls") or {}
    if isinstance(tls_raw, Mapping):
        ms.tls = MetricsTLSConfig(
            enabled=bool(tls_raw.get("enabled", False)),
            certificate_path=_resolve_path(base_dir, tls_raw.get("certificate_path")),
            private_key_path=_resolve_path(base_dir, tls_raw.get("private_key_path")),
            client_ca_path=_resolve_path(base_dir, tls_raw.get("client_ca_path")),
            require_client_auth=bool(tls_raw.get("require_client_auth", False)),
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
    _parse_environments_section(core, data.get("environments") or {}, base_dir=base_dir, risk_profiles=risk_profiles)

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
