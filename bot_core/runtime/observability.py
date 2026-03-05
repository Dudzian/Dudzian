"""Interfejsy i pomocnicze funkcje obserwowalności runtime."""

from __future__ import annotations

import json
import logging
import re
import sys
import inspect
from datetime import timedelta
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, MutableMapping, Protocol

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
from bot_core.alerts.base import AlertAuditLog, AlertChannel, AlertMessage
from bot_core.config.models import (
    CoreConfig,
    EmailChannelSettings,
    EnvironmentConfig,
    MessengerChannelSettings,
    SMSProviderSettings,
    SignalChannelSettings,
    TelegramChannelSettings,
    WhatsAppChannelSettings,
)
from bot_core.runtime.paths import RuntimePaths
from bot_core.security import SecretManager, SecretStorageError
from bot_core.security.guards import get_capability_guard

_LOGGER = logging.getLogger(__name__)

_ADVANCED_ALERT_TYPES = frozenset({"sms", "signal", "whatsapp", "messenger"})

_SMS_PROVIDERS_MODULE = "bot_core.alerts.channels.providers"
_SMS_PROVIDERS_STUB_FLAG = "__bootstrap_sms_provider_stub__"
_ISO_COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")
_E164_PATTERN = re.compile(r"^\+[1-9]\d{1,14}$")
_ALPHANUMERIC_SENDER_PATTERN = re.compile(r"^[A-Z0-9 _-]+$", re.IGNORECASE)


class AlertSink(Protocol):
    """Minimalny interfejs sinka alertów używany w runtime."""

    def dispatch(self, message: AlertMessage) -> None: ...

    def health_snapshot(self) -> Mapping[str, Mapping[str, object]]: ...


class RouterAlertSink(AlertSink):
    """Adapter sinka oparty o domyślny router alertów."""

    def __init__(self, router: DefaultAlertRouter) -> None:
        self._router = router

    def dispatch(self, message: AlertMessage) -> None:
        self._router.dispatch(message)

    def health_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        return self._router.health_snapshot()

    @property
    def router(self) -> DefaultAlertRouter:
        return self._router


@lru_cache(maxsize=1)
def _get_alert_components() -> Mapping[str, Any]:
    """Zwraca klasowe zależności alertów wymagane podczas bootstrapa."""

    components: dict[str, Any] = {
        "FileAlertAuditLog": FileAlertAuditLog,
        "InMemoryAlertAuditLog": InMemoryAlertAuditLog,
        "AlertThrottle": AlertThrottle,
        "DefaultAlertRouter": DefaultAlertRouter,
        "EmailChannel": EmailChannel,
        "TelegramChannel": TelegramChannel,
        "SMSChannel": SMSChannel,
        "SignalChannel": SignalChannel,
        "WhatsAppChannel": WhatsAppChannel,
        "MessengerChannel": MessengerChannel,
        "get_sms_provider": get_sms_provider,
    }

    try:  # pragma: no cover - zależność opcjonalna w dystrybucjach bez SMS
        from bot_core.alerts.channels.providers import (
            SmsProviderConfig as _SmsProviderConfig,
        )
    except Exception:  # pragma: no cover - brak modułu providers
        _SmsProviderConfig = None  # type: ignore[assignment]

    components["SmsProviderConfig"] = _SmsProviderConfig
    return components


def build_ui_alert_audit_metadata(
    router: DefaultAlertRouter | None, *, requested_backend: str | None
) -> dict[str, object]:
    """Zwraca metadane backendu audytu alertów UI dostępne w runtime."""

    components = _get_alert_components()
    FileAlertAuditLogCls = components["FileAlertAuditLog"]
    InMemoryAlertAuditLogCls = components["InMemoryAlertAuditLog"]

    normalized_request = (requested_backend or "inherit").lower()
    metadata: dict[str, object] = {"requested": normalized_request}

    audit_log = getattr(router, "audit_log", None)

    backend: str
    note: str | None = None

    if isinstance(audit_log, FileAlertAuditLogCls):
        backend = "file"
        metadata.update(
            {
                "directory": str(getattr(audit_log, "directory", "")) or None,
                "pattern": getattr(audit_log, "filename_pattern", None),
                "retention_days": getattr(audit_log, "retention_days", None),
                "fsync": bool(getattr(audit_log, "fsync", False)),
            }
        )
    elif isinstance(audit_log, InMemoryAlertAuditLogCls):
        backend = "memory"
    else:
        # Brak skonfigurowanego backendu traktujemy jako degradację do pamięci.
        backend = "memory"

    if normalized_request == "file" and backend != "file":
        note = "file_backend_unavailable"
    elif normalized_request == "memory" and backend != "memory":
        note = "memory_backend_not_selected"
    elif normalized_request == "inherit":
        note = "inherited_environment_router"

    metadata["backend"] = backend
    if note:
        metadata["note"] = note

    return metadata


def build_alert_channels(
    *,
    core_config: CoreConfig,
    environment: EnvironmentConfig,
    secret_manager: SecretManager,
    runtime_paths: RuntimePaths | None = None,
) -> tuple[Mapping[str, AlertChannel], DefaultAlertRouter, AlertAuditLog]:
    """Tworzy i rejestruje kanały alertów + router + backend audytu."""

    runtime_paths = runtime_paths or RuntimePaths.from_environment(environment)
    components = _get_alert_components()
    FileAlertAuditLogCls = components["FileAlertAuditLog"]
    InMemoryAlertAuditLogCls = components["InMemoryAlertAuditLog"]
    AlertThrottleCls = components["AlertThrottle"]
    DefaultAlertRouterCls = components["DefaultAlertRouter"]

    audit_config = getattr(environment, "alert_audit", None)
    if audit_config and getattr(audit_config, "backend", "memory") == "file":
        directory = runtime_paths.resolve_data_path(
            getattr(audit_config, "directory", None), default="alerts"
        )
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
    guard = get_capability_guard()

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

        if guard and channel_type in _ADVANCED_ALERT_TYPES:
            guard.require_module(
                "alerts_advanced",
                message=("Kanały SMS/Signal/WhatsApp/Messenger wymagają modułu Alerts Advanced."),
            )

        if channel_type == "telegram":
            channel = _build_telegram_channel(
                core_config.telegram_channels, channel_key, secret_manager
            )
        elif channel_type == "email":
            channel = _build_email_channel(core_config.email_channels, channel_key, secret_manager)
        elif channel_type == "sms":
            channel = _build_sms_channel(core_config.sms_providers, channel_key, secret_manager)
        elif channel_type == "signal":
            channel = _build_signal_channel(
                core_config.signal_channels, channel_key, secret_manager
            )
        elif channel_type == "whatsapp":
            channel = _build_whatsapp_channel(
                core_config.whatsapp_channels, channel_key, secret_manager
            )
        elif channel_type == "messenger":
            channel = _build_messenger_channel(
                core_config.messenger_channels, channel_key, secret_manager
            )
        else:
            _LOGGER.warning("Nieznany typ kanału alertów: %s", channel_type)
            continue

        channels[channel.name] = channel
        # Kompatybilność: starsze routery miały register(), nowsze integracje mogły wołać register_channel().
        register_fn = getattr(router, "register_channel", None)
        if callable(register_fn):
            register_fn(channel)
        else:
            router.register(channel)

    if skipped_offline:
        _LOGGER.info("Pominięto kanały alertów w trybie offline: %s", ", ".join(skipped_offline))

    return channels, router, audit_log


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
        raw_secret = secret_manager.load_secret_value(
            settings.credential_secret, purpose="alerts:email"
        )
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

    wants_alphanumeric = bool(settings.allow_alphanumeric_sender)
    has_sender_id = bool(settings.sender_id)

    if wants_alphanumeric and not has_sender_id:
        raise SecretStorageError(
            (
                f"Konfiguracja dostawcy SMS '{channel_key}' ma włączone pole "
                "'allow_alphanumeric_sender', ale nie dostarczono 'sender_id'."
            )
        )

    if not wants_alphanumeric and has_sender_id:
        raise SecretStorageError(
            (
                f"Konfiguracja dostawcy SMS '{channel_key}' zawiera 'sender_id', lecz "
                "'allow_alphanumeric_sender' pozostaje wyłączone."
            )
        )

    provider_config = _resolve_sms_provider(
        settings,
        get_sms_provider_fn,
        channel_key=channel_key,
    )
    supports_alphanumeric = bool(getattr(provider_config, "supports_alphanumeric_sender", False))
    if wants_alphanumeric and not supports_alphanumeric:
        raise SecretStorageError(
            (
                f"Konfiguracja dostawcy SMS '{channel_key}' ma włączone pole "
                "'allow_alphanumeric_sender', ale wybrany operator go nie obsługuje (brak wsparcia "
                "nadawców alfanumerycznych)."
            )
        )

    sender = (
        settings.sender_id
        or getattr(settings, "from_number", None)
        or getattr(provider_config, "default_sender", None)
    )
    if not wants_alphanumeric:
        if not sender:
            raise SecretStorageError(
                f"Konfiguracja dostawcy SMS '{channel_key}' wymaga numeru nadawcy w formacie E.164."
            )
        _validate_e164_number(str(sender).strip(), channel_key, field="from_number")
    if wants_alphanumeric and sender:
        if not _ALPHANUMERIC_SENDER_PATTERN.match(sender):
            raise SecretStorageError(
                (
                    f"Identyfikator nadawcy '{sender}' dostawcy SMS '{channel_key}' może "
                    "zawierać wyłącznie litery A-Z, cyfry 0-9, spacje oraz znaki '- _'."
                )
            )
        if len(sender) < 3:
            raise SecretStorageError(
                (
                    f"Identyfikator nadawcy '{sender}' dostawcy SMS '{channel_key}' "
                    "musi mieć co najmniej trzy znaki."
                )
            )
        if not re.search(r"[A-Za-z]", sender):
            raise SecretStorageError(
                (
                    f"Identyfikator nadawcy '{sender}' dostawcy SMS '{channel_key}' "
                    "musi zawierać co najmniej jedną literę."
                )
            )
        max_length = getattr(provider_config, "max_sender_length", 11) or 11
        if len(sender) > max_length:
            raise SecretStorageError(
                (
                    f"Identyfikator nadawcy '{sender}' dostawcy SMS '{channel_key}' "
                    f"przekracza dopuszczalny limit: maksymalna długość to {max_length} znaków."
                )
            )

    raw_recipients = getattr(settings, "to", None) or getattr(settings, "recipients", None)
    if not raw_recipients:
        raise SecretStorageError(
            f"Konfiguracja dostawcy SMS '{channel_key}' wymaga pola 'to' lub 'recipients'."
        )

    recipients = (
        list(raw_recipients) if isinstance(raw_recipients, (list, tuple, set)) else [raw_recipients]
    )
    normalized_recipients = [str(recipient).strip() for recipient in recipients]
    if len(set(normalized_recipients)) != len(normalized_recipients):
        raise SecretStorageError(
            f"Konfiguracja dostawcy SMS '{channel_key}' zawiera zduplikowane numery odbiorców."
        )
    for recipient in normalized_recipients:
        _validate_e164_number(recipient, channel_key, field="to")

    init_params = inspect.signature(SMSChannelCls).parameters
    sms_kwargs: dict[str, Any] = {
        "provider": provider_config,
        "name": f"sms:{channel_key}",
        "account_sid": account_sid,
        "auth_token": auth_token,
    }

    if "recipients" in init_params:
        sms_kwargs["recipients"] = normalized_recipients
    elif "to" in init_params:
        sms_kwargs["to"] = (
            normalized_recipients[0] if len(normalized_recipients) == 1 else normalized_recipients
        )

    if "from_number" in init_params:
        sms_kwargs["from_number"] = sender
    elif "sender" in init_params:
        sms_kwargs["sender"] = sender

    sms_channel = SMSChannelCls(**sms_kwargs)

    return sms_channel


def _build_signal_channel(
    definitions: Mapping[str, SignalChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> SignalChannel:
    components = _get_alert_components()
    SignalChannelCls = components["SignalChannel"]
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Signal '{channel_key}'") from exc

    if not settings.recipients:
        raise SecretStorageError(
            f"Konfiguracja kanału Signal '{channel_key}' wymaga listy 'recipients'."
        )

    _validate_e164_number(settings.sender, channel_key, field="sender")
    for recipient in settings.recipients:
        _validate_e164_number(recipient, channel_key, field="recipient")

    credential = None
    if settings.credential_secret:
        credential = secret_manager.load_secret_value(
            settings.credential_secret, purpose="alerts:signal"
        )

    return SignalChannelCls(
        sender=settings.sender,
        recipients=tuple(settings.recipients),
        message_template=settings.message_template,
        rate_limit_interval=settings.rate_limit_interval,
        credential=credential,
        name=f"signal:{channel_key}",
    )


def _build_whatsapp_channel(
    definitions: Mapping[str, WhatsAppChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> WhatsAppChannel:
    components = _get_alert_components()
    WhatsAppChannelCls = components["WhatsAppChannel"]
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału WhatsApp '{channel_key}'") from exc

    if not settings.recipients:
        raise SecretStorageError(
            f"Konfiguracja kanału WhatsApp '{channel_key}' wymaga listy 'recipients'."
        )

    _validate_e164_number(settings.sender, channel_key, field="sender")
    for recipient in settings.recipients:
        _validate_e164_number(recipient, channel_key, field="recipient")

    credential = None
    if settings.credential_secret:
        credential = secret_manager.load_secret_value(
            settings.credential_secret, purpose="alerts:whatsapp"
        )

    return WhatsAppChannelCls(
        sender=settings.sender,
        recipients=tuple(settings.recipients),
        message_template=settings.message_template,
        rate_limit_interval=settings.rate_limit_interval,
        credential=credential,
        name=f"whatsapp:{channel_key}",
    )


def _build_messenger_channel(
    definitions: Mapping[str, MessengerChannelSettings],
    channel_key: str,
    secret_manager: SecretManager,
) -> MessengerChannel:
    components = _get_alert_components()
    MessengerChannelCls = components["MessengerChannel"]
    try:
        settings = definitions[channel_key]
    except KeyError as exc:
        raise KeyError(f"Brak definicji kanału Messenger '{channel_key}'") from exc

    if not settings.recipient:
        raise SecretStorageError(
            f"Konfiguracja kanału Messenger '{channel_key}' wymaga pola 'recipient'."
        )

    credential = None
    if settings.credential_secret:
        credential = secret_manager.load_secret_value(
            settings.credential_secret, purpose="alerts:messenger"
        )

    return MessengerChannelCls(
        recipient=settings.recipient,
        credential=credential,
        name=f"messenger:{channel_key}",
    )


def _resolve_sms_provider(
    settings: SMSProviderSettings,
    get_sms_provider_fn: Any,
    *,
    channel_key: str,
) -> Any:
    components = _get_alert_components()
    SmsProviderConfigCls = components.get("SmsProviderConfig")
    if SmsProviderConfigCls is None:
        raise KeyError("Typ SmsProviderConfig nie jest dostępny w module alertów")

    is_stub_registry = getattr(get_sms_provider_fn, _SMS_PROVIDERS_STUB_FLAG, False)
    try:
        override_iso_code = _normalize_iso_country_code(settings.iso_country_code)
    except ValueError:
        raise SecretStorageError(
            (
                f"Konfiguracja dostawcy SMS '{channel_key}' zawiera nieprawidłowy kod kraju "
                f"'{settings.iso_country_code}'. Oczekiwany jest kod ISO 3166-1 alfa-2."
            )
        ) from None
    try:
        base = get_sms_provider_fn(settings.provider_key)
    except KeyError:
        if not is_stub_registry:
            raise

        _LOGGER.warning(
            "Fallback rejestru SMS: używam konfiguracji lokalnej bez metadanych dostawcy.",
            extra={
                "provider_key": settings.provider_key,
                "channel": settings.name,
            },
        )
        base = SimpleNamespace(
            provider_id=settings.provider_key,
            display_name=settings.display_name or f"{settings.provider_key} (bootstrap)",
            api_base_url=settings.api_base_url,
            iso_country_code=override_iso_code or "ZZ",
            supports_alphanumeric_sender=settings.allow_alphanumeric_sender,
            notes=settings.notes
            or "Bootstrap fallback provider – brak zarejestrowanego operatora.",
            max_sender_length=settings.max_sender_length or 11,
        )

    raw_base_iso = getattr(base, "iso_country_code", None)
    try:
        base_iso_code = _normalize_iso_country_code(raw_base_iso)
    except ValueError:
        _LOGGER.warning(
            (
                "Rejestr dostawców SMS zwrócił nieprawidłowy kod kraju %r dla operatora '%s'. "
                "Zastępuję wartością 'ZZ'."
            ),
            raw_base_iso,
            settings.provider_key,
            extra={
                "provider_key": settings.provider_key,
                "channel": settings.name,
                "invalid_iso_country_code": raw_base_iso,
            },
        )
        base_iso_code = "ZZ"

    display_name = settings.display_name or getattr(base, "display_name", settings.provider_key)
    iso_country_code = override_iso_code or base_iso_code or "ZZ"
    notes = settings.notes if settings.notes is not None else getattr(base, "notes", None)
    max_sender_length = (
        settings.max_sender_length
        if settings.max_sender_length is not None
        else getattr(base, "max_sender_length", 11)
    )
    supports_attr = getattr(base, "supports_alphanumeric_sender", None)
    if supports_attr is None:
        supports_alphanumeric = bool(settings.allow_alphanumeric_sender)
    else:
        supports_alphanumeric = bool(supports_attr)

    provider_config = SmsProviderConfigCls(
        provider_id=base.provider_id,
        display_name=display_name,
        api_base_url=settings.api_base_url or getattr(base, "api_base_url", settings.api_base_url),
        iso_country_code=iso_country_code,
        supports_alphanumeric_sender=supports_alphanumeric,
        notes=notes,
        max_sender_length=max_sender_length,
    )

    if is_stub_registry:
        providers_module = sys.modules.get(_SMS_PROVIDERS_MODULE)
        if providers_module and getattr(providers_module, _SMS_PROVIDERS_STUB_FLAG, False):
            providers_module.DEFAULT_SMS_PROVIDERS[settings.provider_key] = provider_config

    return provider_config


def _normalize_iso_country_code(value: Any) -> str | None:
    if value in (None, "", False):
        return None

    candidate = str(value).strip().upper()
    if _ISO_COUNTRY_PATTERN.match(candidate):
        return candidate

    raise ValueError(candidate)


def _validate_e164_number(value: str, channel_key: str, *, field: str) -> None:
    if not _E164_PATTERN.match(str(value)):
        example = "+48123456789"
        raise SecretStorageError(
            (
                f"Wartość '{value}' dla pola '{field}' dostawcy SMS '{channel_key}' musi być "
                f"w formacie E.164 (np. {example})."
            )
        )


def _install_sms_provider_stub() -> None:
    """Instaluje minimalny rejestr dostawców SMS na potrzeby testów."""

    providers_module = sys.modules.get(_SMS_PROVIDERS_MODULE)
    if providers_module is None:
        providers_module = ModuleType(_SMS_PROVIDERS_MODULE)
        providers_module.DEFAULT_SMS_PROVIDERS = {}

        @dataclass(slots=True)
        class SmsProviderConfig:  # type: ignore[invalid-annotation]
            provider_id: str
            display_name: str
            api_base_url: str | None = None
            iso_country_code: str | None = None
            supports_alphanumeric_sender: bool = False
            notes: str | None = None
            max_sender_length: int = 11

        def get_sms_provider(key: str) -> SmsProviderConfig:
            try:
                return providers_module.DEFAULT_SMS_PROVIDERS[key]
            except KeyError as exc:  # pragma: no cover - diagnostyka testowa
                raise KeyError(f"Brak zarejestrowanego dostawcy SMS '{key}'") from exc

        providers_module.SmsProviderConfig = SmsProviderConfig  # type: ignore[attr-defined]
        providers_module.get_sms_provider = get_sms_provider  # type: ignore[attr-defined]
        sys.modules[_SMS_PROVIDERS_MODULE] = providers_module
    else:
        providers_module.DEFAULT_SMS_PROVIDERS = getattr(
            providers_module, "DEFAULT_SMS_PROVIDERS", {}
        )

        if not hasattr(providers_module, "SmsProviderConfig"):

            @dataclass(slots=True)
            class SmsProviderConfig:  # type: ignore[invalid-annotation]
                provider_id: str
                display_name: str
                api_base_url: str | None = None
                iso_country_code: str | None = None
                supports_alphanumeric_sender: bool = False
                notes: str | None = None
                max_sender_length: int = 11

            providers_module.SmsProviderConfig = SmsProviderConfig  # type: ignore[attr-defined]

        if not hasattr(providers_module, "get_sms_provider"):

            def get_sms_provider(key: str):
                return providers_module.DEFAULT_SMS_PROVIDERS[key]

            providers_module.get_sms_provider = get_sms_provider  # type: ignore[attr-defined]

    setattr(providers_module, _SMS_PROVIDERS_STUB_FLAG, True)
    get_provider_fn = getattr(providers_module, "get_sms_provider")
    setattr(get_provider_fn, _SMS_PROVIDERS_STUB_FLAG, True)


__all__ = [
    "AlertSink",
    "RouterAlertSink",
    "build_alert_channels",
    "build_ui_alert_audit_metadata",
    "_install_sms_provider_stub",
]
