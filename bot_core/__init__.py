"""Podstawowy pakiet runtime bota handlowego."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - tylko podpowiedzi typów
    from .auto_trader import AutoTrader, EmitterLike, RiskDecision
    from .trading import TradingEngine, TradingEngineFactory, TradingParameters, TradingStrategies

# Eksporty wymagające zależności opcjonalnych – ładowane leniwie w __getattr__
_OPTIONAL_IMPORTS: dict[str, str] = {
    "AutoTrader": ".auto_trader",
    "EmitterLike": ".auto_trader",
    "RiskDecision": ".auto_trader",
    "TradingEngine": ".trading",
    "TradingEngineFactory": ".trading",
    "TradingParameters": ".trading",
    "TradingStrategies": ".trading",
}

try:  # pragma: no cover - kanały alertów mogą wymagać zależności opcjonalnych
    from .alerts import (
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
except Exception:  # pragma: no cover - eksponujemy stabilny interfejs nawet bez zależności
    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]

    def get_sms_provider(*_: Any, **__: Any) -> Callable[..., Any]:  # type: ignore[misc]
        raise RuntimeError("SMS providers are not available in this environment")

try:  # pragma: no cover - loader konfiguracji może wymagać dodatkowych pakietów
    from .config.loader import load_core_config
except Exception:  # pragma: no cover
    def load_core_config(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("Core configuration loader is not available")

try:  # pragma: no cover - modele konfiguracji są opcjonalne
    from .config.models import CoreConfig
except Exception:  # pragma: no cover
    CoreConfig = None  # type: ignore[assignment]

try:  # pragma: no cover - manager bazy danych może być opcjonalny
    from .database import DatabaseManager
except Exception:  # pragma: no cover
    DatabaseManager = None  # type: ignore[assignment]

try:  # pragma: no cover - niektóre giełdy mogą nie być dostępne w środowisku CI
    from .exchanges import (
        BaseBackend,
        Event,
        EventBus,
        ExchangeAdapter,
        ExchangeManager,
        MarketRules,
        Mode,
        OrderDTO,
        OrderResult,
        OrderSide,
        OrderStatus,
        OrderType,
        PaperBackend,
        PositionDTO,
    )
except Exception:  # pragma: no cover
    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]

try:  # pragma: no cover - środowisko runtime może nie być kompletnie zainstalowane
    from .runtime import BootstrapContext, bootstrap_environment
except Exception:  # pragma: no cover
    BootstrapContext = None  # type: ignore[assignment]

    def bootstrap_environment(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("Runtime bootstrap is not available")

try:  # pragma: no cover - komponenty bezpieczeństwa mogą nie być obecne
    from .security import (
        KeyringSecretStorage,
        SecretManager,
        SecretStorageError,
    )
except Exception:  # pragma: no cover
    KeyringSecretStorage = SecretManager = SecretStorageError = None  # type: ignore[assignment]

__all__ = [
    "AlertChannel",
    "AlertMessage",
    "AutoTrader",
    "BaseBackend",
    "BootstrapContext",
    "CoreConfig",
    "DatabaseManager",
    "DEFAULT_SMS_PROVIDERS",
    "EmitterLike",
    "Event",
    "EventBus",
    "ExchangeAdapter",
    "ExchangeManager",
    "KeyringSecretStorage",
    "MarketRules",
    "Mode",
    "OrderDTO",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperBackend",
    "PositionDTO",
    "RiskDecision",
    "SmsProviderConfig",
    "SecretManager",
    "SecretStorageError",
    "TradingEngine",
    "TradingEngineFactory",
    "TradingParameters",
    "TradingStrategies",
    "bootstrap_environment",
    "get_sms_provider",
    "load_core_config",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - leniwe importy
    module_name = _OPTIONAL_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'bot_core' has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, name)
    except Exception as exc:  # pragma: no cover - zachowaj degradację
        raise RuntimeError(f"{name} is not available: {exc}") from exc
    globals()[name] = value
    return value
