"""Nowa modularna architektura bota handlowego."""

from bot_core.alerts import (
    AlertChannel,
    AlertMessage,
    DEFAULT_SMS_PROVIDERS,
    SmsProviderConfig,
    get_sms_provider,
)
from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig
from bot_core.exchanges.base import ExchangeAdapter
from bot_core.observability import MetricsRegistry, get_global_metrics_registry
from bot_core.security import (
    KeyringSecretStorage,
    SecretManager,
    SecretStorageError,
)
from bot_core.runtime import BootstrapContext, bootstrap_environment

__all__ = [
    "AlertChannel",
    "AlertMessage",
    "DEFAULT_SMS_PROVIDERS",
    "SmsProviderConfig",
    "get_sms_provider",
    "CoreConfig",
    "ExchangeAdapter",
    "MetricsRegistry",
    "KeyringSecretStorage",
    "SecretManager",
    "SecretStorageError",
    "BootstrapContext",
    "bootstrap_environment",
    "load_core_config",
    "get_global_metrics_registry",
]
