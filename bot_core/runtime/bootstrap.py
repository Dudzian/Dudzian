"""Procedury rozruchowe spinające konfigurację z modułami runtime."""
from __future__ import annotations

import json
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
from bot_core.exchanges.base import (
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeCredentials,
)
from bot_core.exchanges.binance import BinanceFuturesAdapter, BinanceSpotAdapter
from bot_core.exchanges.kraken import KrakenFuturesAdapter, KrakenSpotAdapter
from bot_core.exchanges.zonda import ZondaSpotAdapter
from bot_core.risk.base import RiskProfile, RiskRepository
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.risk.repository import FileRiskRepository
from bot_core.security import SecretManager, SecretStorageError
from bot_core.runtime.journal import (
    InMemoryTradingDecisionJournal,
    JsonlTradingDecisionJournal,
    TradingDecisionJournal,
)

_DEFAULT_ADAPTERS: Mapping[str, ExchangeAdapterFactory] = {
    "binance_spot": BinanceSpotAdapter,
    "binance_futures": BinanceFuturesAdapter,
    "kraken_spot": KrakenSpotAdapter,
    "kraken_futures": KrakenFuturesAdapter,
    "zonda_spot": ZondaSpotAdapter,
}

_PROFILE_CLASS_BY_NAME: Mapping[str, type[RiskProfile]] = {
    "conservative": ConservativeProfile,
    "balanced": BalancedProfile,
    "aggressive": AggressiveProfile,
}


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


def bootstrap_environment(
    environment_name: str,
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
) -> BootstrapContext:
    """Tworzy kompletny kontekst uruchomieniowy dla wskazanego środowiska."""
    core_config = load_core_config(config_path)
    if environment_name not in core_config.environments:
        raise KeyError(f"Środowisko '{environment_name}' nie istnieje w konfiguracji")

    environment = core_config.environments[environment_name]
    risk_profile_config = _resolve_risk_profile(core_config.risk_profiles, environment.risk_profile)

    risk_repository_path = Path(environment.data_cache_path) / "risk_state"
    risk_repository = FileRiskRepository(risk_repository_path)
    risk_engine = ThresholdRiskEngine(repository=risk_repository)
    profile = _build_risk_profile(risk_profile_config)
    risk_engine.register_profile(profile)

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


def _build_risk_profile(config: RiskProfileConfig) -> RiskProfile:
    """Tworzy profil ryzyka na podstawie konfiguracji.

    - Jeśli `name` = "manual" → użyj parametrów z configu (ManualProfile).
    - Jeśli `name` w {conservative, balanced, aggressive} → użyj predefiniowanych profili (domyślne wartości).
    - W innym przypadku fallback do ManualProfile z parametrami z configu.
    """
    profile_key = config.name.lower()
    if profile_key == "manual":
        return ManualProfile(
            name=config.name,
            max_positions=config.max_open_positions,
            max_leverage=config.max_leverage,
            drawdown_limit=config.hard_drawdown_pct,
            daily_loss_limit=config.max_daily_loss_pct,
            max_position_pct=config.max_position_pct,
            target_volatility=config.target_volatility,
            stop_loss_atr_multiple=config.stop_loss_atr_multiple,
        )

    profile_class = _PROFILE_CLASS_BY_NAME.get(profile_key)
    if profile_class is not None:
        # Predefiniowane profile mają sensowne domyślne parametry;
        # jeśli potrzebujesz parametryzacji — użyj profilu "manual".
        return profile_class()

    # Fallback: parametry z konfiguracji jako profil "manual"
    return ManualProfile(
        name=config.name,
        max_positions=config.max_open_positions,
        max_leverage=config.max_leverage,
        drawdown_limit=config.hard_drawdown_pct,
        daily_loss_limit=config.max_daily_loss_pct,
        max_position_pct=config.max_position_pct,
        target_volatility=config.target_volatility,
        stop_loss_atr_multiple=config.stop_loss_atr_multiple,
    )


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
        if not directory is None and not directory.is_absolute():
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
