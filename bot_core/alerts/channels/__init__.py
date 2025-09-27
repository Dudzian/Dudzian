"""Adaptery kanałów powiadomień."""

from bot_core.alerts.channels.email import EmailChannel
from bot_core.alerts.channels.providers import (
    DEFAULT_SMS_PROVIDERS,
    SmsProviderConfig,
    get_sms_provider,
)
from bot_core.alerts.channels.sms import SMSChannel
from bot_core.alerts.channels.telegram import TelegramChannel

__all__ = [
    "EmailChannel",
    "SMSChannel",
    "TelegramChannel",
    "SmsProviderConfig",
    "DEFAULT_SMS_PROVIDERS",
    "get_sms_provider",
]
