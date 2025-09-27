"""Adaptery kanałów powiadomień."""

from bot_core.alerts.channels.email import EmailChannel
from bot_core.alerts.channels.providers import (
    DEFAULT_SMS_PROVIDERS,
    SmsProviderConfig,
    get_sms_provider,
)
from bot_core.alerts.channels.sms import SMSChannel
from bot_core.alerts.channels.telegram import TelegramChannel

# --- Kanały opcjonalne (bez twardej zależności) ---
try:  # pragma: no cover
    from bot_core.alerts.channels.signal import SignalChannel  # type: ignore
except Exception:  # pragma: no cover
    SignalChannel = None  # type: ignore

try:  # pragma: no cover
    from bot_core.alerts.channels.whatsapp import WhatsAppChannel  # type: ignore
except Exception:  # pragma: no cover
    WhatsAppChannel = None  # type: ignore

try:  # pragma: no cover
    from bot_core.alerts.channels.messenger import MessengerChannel  # type: ignore
except Exception:  # pragma: no cover
    MessengerChannel = None  # type: ignore

__all__ = [
    "EmailChannel",
    "SMSChannel",
    "TelegramChannel",
    "SmsProviderConfig",
    "DEFAULT_SMS_PROVIDERS",
    "get_sms_provider",
]

# Eksportuj opcjonalne tylko jeśli dostępne
if SignalChannel is not None:  # pragma: no cover
    __all__.append("SignalChannel")
if WhatsAppChannel is not None:  # pragma: no cover
    __all__.append("WhatsAppChannel")
if MessengerChannel is not None:  # pragma: no cover
    __all__.append("MessengerChannel")
