"""Podstawowy pakiet runtime bota handlowego."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - moduł auto_trader może być opcjonalny w CI
    from .auto_trader import AutoTrader, EmitterLike, RiskDecision
except Exception:  # pragma: no cover - zapewniamy bezpieczny import pakietu
    AutoTrader = None  # type: ignore[assignment]
    EmitterLike = object  # type: ignore[assignment]
    RiskDecision = None  # type: ignore[assignment]

try:  # pragma: no cover - importy warunkowe dla środowiska testowego
    from .alerts import (
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
except Exception:  # pragma: no cover - zapewniamy zgodność, gdy brak zależności
    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]

    def get_sms_provider(*_: Any, **__: Any) -> Callable[..., Any]:  # type: ignore[misc]
        raise RuntimeError("SMS providers are not available in this environment")

try:  # pragma: no cover - ładowanie konfiguracji może wymagać zależności opcjonalnych
    from .config.loader import load_core_config
except Exception:  # pragma: no cover - fallback gdy loader niedostępny
    def load_core_config(*_: Any, **__: Any) -> Any:  # type: ignore[override]
        raise RuntimeError("Core configuration loader is not available")

try:  # pragma: no cover - modele konfiguracyjne mogą nie być obecne
    from .config.models import CoreConfig
except Exception:  # pragma: no cover
    CoreConfig = None  # type: ignore[assignment]

try:  # pragma: no cover - połączenie z bazą danych może wymagać zależności
    from .database import DatabaseManager
except Exception:  # pragma: no cover
    DatabaseManager = None  # type: ignore[assignment]

try:  # pragma: no cover - zależności giełdowe mogą być pominięte
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
except Exception:  # pragma: no cover - zapewniamy import nawet bez pełnych modułów
    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]

try:  # pragma: no cover - środowisko runtime może nie być kompletnie zainstalowane
    from .runtime import BootstrapContext, bootstrap_environment
except Exception:  # pragma: no cover
    BootstrapContext = None  # type: ignore[assignment]

    def bootstrap_environment(*_: Any, **__: Any) -> Any:  # type: ignore[override]
        raise RuntimeError("Runtime bootstrap is not available")

try:  # pragma: no cover - trading może wymagać zależności opcjonalnych
    from .trading import (
        TradingEngine,
        TradingEngineFactory,
        TradingParameters,
        TradingStrategies,
    )
except Exception:  # pragma: no cover
    TradingEngine = TradingEngineFactory = TradingStrategies = None  # type: ignore[assignment]

    class TradingParameters:  # type: ignore[override]
        pass

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