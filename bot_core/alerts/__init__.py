"""Pakiet kanałów alertów i routera."""

from bot_core.alerts.audit import AlertAuditEntry, FileAlertAuditLog, InMemoryAlertAuditLog
from bot_core.alerts.base import (
    AlertAuditLog,
    AlertChannel,
    AlertDeliveryError,
    AlertMessage,
    AlertRouter,
)
from bot_core.alerts.channels import (
    DEFAULT_SMS_PROVIDERS,
    EmailChannel,
    SMSChannel,
    SmsProviderConfig,
    TelegramChannel,
    get_sms_provider,
)
from bot_core.alerts.router import DefaultAlertRouter
from bot_core.alerts.throttle import AlertThrottle

# Optional messenger channels (keep package import-safe if not installed/implemented)
try:  # pragma: no cover
    from bot_core.alerts.channels import SignalChannel  # type: ignore
except Exception:  # pragma: no cover
    SignalChannel = None  # type: ignore

try:  # pragma: no cover
    from bot_core.alerts.channels import WhatsAppChannel  # type: ignore
except Exception:  # pragma: no cover
    WhatsAppChannel = None  # type: ignore

try:  # pragma: no cover
    from bot_core.alerts.channels import MessengerChannel  # type: ignore
except Exception:  # pragma: no cover
    MessengerChannel = None  # type: ignore


__all__ = [
    "AlertAuditEntry",
    "AlertAuditLog",
    "AlertChannel",
    "AlertDeliveryError",
    "AlertMessage",
    "AlertRouter",
    "DefaultAlertRouter",
    "AlertThrottle",
    "EmailChannel",
    "SMSChannel",
    "SmsProviderConfig",
    "TelegramChannel",
    "DEFAULT_SMS_PROVIDERS",
    "get_sms_provider",
    "InMemoryAlertAuditLog",
    "FileAlertAuditLog",
]

# Expose optional channels only when available
if SignalChannel is not None:  # pragma: no cover
    __all__.append("SignalChannel")
if WhatsAppChannel is not None:  # pragma: no cover
    __all__.append("WhatsAppChannel")
if MessengerChannel is not None:  # pragma: no cover
    __all__.append("MessengerChannel")
