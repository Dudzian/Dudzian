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
from bot_core.security import (
    KeyringSecretStorage,
    SecretManager,
    SecretStorageError,
)
from bot_core.runtime import BootstrapContext, bootstrap_environment

# Observability is optional; expose when available.
try:  # pragma: no cover - optional dependency
    from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore
except Exception:  # pragma: no cover - keep package import-safe without observability
    MetricsRegistry = None  # type: ignore[assignment]
    get_global_metrics_registry = None  # type: ignore[assignment]

__all__ = [
    "AlertChannel",
    "AlertMessage",
    "DEFAULT_SMS_PROVIDERS",
    "SmsProviderConfig",
    "get_sms_provider",
    "CoreConfig",
    "ExchangeAdapter",
    "KeyringSecretStorage",
    "SecretManager",
    "SecretStorageError",
    "BootstrapContext",
    "bootstrap_environment",
    "load_core_config",
]

# Add observability symbols only if successfully imported
if MetricsRegistry is not None and get_global_metrics_registry is not None:  # pragma: no cover
    __all__ += ["MetricsRegistry", "get_global_metrics_registry"]
