"""Podstawowy pakiet runtime bota handlowego."""
from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - komponent auto_trader może nie być dostępny
    from .auto_trader import AutoTrader, EmitterLike, RiskDecision
except ImportError as exc:  # pragma: no cover - zapewnia bezpieczny import pakietu
    AutoTrader = None  # type: ignore[assignment]
    EmitterLike = object  # type: ignore[assignment]
    RiskDecision = None  # type: ignore[assignment]

try:  # pragma: no cover - kanały alertów mogą wymagać zależności opcjonalnych
    from .alerts import (
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
except ImportError as exc:  # pragma: no cover - eksponujemy stabilny interfejs nawet bez zależności
    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]

    def get_sms_provider(*_: Any, **__: Any) -> Callable[..., Any]:  # type: ignore[misc]
        raise RuntimeError("SMS providers are not available in this environment") from exc

try:  # pragma: no cover - loader konfiguracji może wymagać dodatkowych pakietów
    from .config.loader import load_core_config
except ImportError as exc:  # pragma: no cover
    def load_core_config(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("Core configuration loader is not available") from exc

try:  # pragma: no cover - modele konfiguracji są opcjonalne
    from .config.models import CoreConfig
except ImportError:  # pragma: no cover
    CoreConfig = None  # type: ignore[assignment]

try:  # pragma: no cover - manager bazy danych może być opcjonalny
    from .database import DatabaseManager
except ImportError:  # pragma: no cover
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
except ImportError:  # pragma: no cover
    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]

try:  # pragma: no cover - środowisko runtime może nie być kompletnie zainstalowane
    from .runtime import BootstrapContext, bootstrap_environment
except ImportError as exc:  # pragma: no cover
    BootstrapContext = None  # type: ignore[assignment]

    def bootstrap_environment(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("Runtime bootstrap is not available") from exc

try:  # pragma: no cover - moduły tradingu mogą wymagać dodatkowych zależności
    from .trading import (
        TradingEngine,
        TradingEngineFactory,
        TradingParameters,
        TradingStrategies,
    )
except ImportError as exc:  # pragma: no cover
    TradingEngine = TradingEngineFactory = TradingStrategies = None  # type: ignore[assignment]

    class TradingParameters:  # type: ignore[override]
        """Zaślepka używana, gdy moduł tradingu jest niedostępny."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("Trading module is not available") from exc

try:  # pragma: no cover - komponenty bezpieczeństwa mogą nie być obecne
    from .security import (
        KeyringSecretStorage,
        SecretManager,
        SecretStorageError,
    )
except ImportError:  # pragma: no cover
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
