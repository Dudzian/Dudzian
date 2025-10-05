"""Nowa modularna architektura bota handlowego."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig
from bot_core.exchanges.base import ExchangeAdapter
from bot_core.security import (
    KeyringSecretStorage,
    SecretManager,
    SecretStorageError,
)

if TYPE_CHECKING:  # pragma: no cover - wskazówki dla type-checkera
    from bot_core.alerts import (
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
    from bot_core.runtime import BootstrapContext, bootstrap_environment
else:
    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore
    BootstrapContext = bootstrap_environment = None  # type: ignore


_ALERT_EXPORTS = {
    "AlertChannel",
    "AlertMessage",
    "DEFAULT_SMS_PROVIDERS",
    "SmsProviderConfig",
    "get_sms_provider",
}

_RUNTIME_EXPORTS = {"BootstrapContext", "bootstrap_environment"}


def __getattr__(name: str) -> Any:  # pragma: no cover - prosty delegat
    """Leniwe eksporty modułu alertów, aby unikać błędów importu przy testach."""

    if name in _ALERT_EXPORTS:
        module = import_module("bot_core.alerts")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _RUNTIME_EXPORTS:
        module = import_module("bot_core.runtime")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


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
    "BootstrapContext",
    "bootstrap_environment",
    "KeyringSecretStorage",
    "SecretManager",
    "SecretStorageError",
    "load_core_config",
]

# Add observability symbols only if successfully imported
if MetricsRegistry is not None and get_global_metrics_registry is not None:  # pragma: no cover
    __all__ += ["MetricsRegistry", "get_global_metrics_registry"]
