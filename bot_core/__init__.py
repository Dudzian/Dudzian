"""Podstawowy pakiet runtime bota handlowego."""

from __future__ import annotations
try:  # pragma: no cover - defensywne importy podczas testów
    from bot_core.alerts import (
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
    from bot_core.auto_trader import AutoTrader, EmitterLike, RiskDecision
    from bot_core.config.loader import load_core_config
    from bot_core.config.models import CoreConfig
    from bot_core.database import DatabaseManager
    from bot_core.exchanges import (
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
    from bot_core.runtime import BootstrapContext, bootstrap_environment
    from bot_core.trading import (
        TradingEngine,
        TradingEngineFactory,
        TradingParameters,
        TradingStrategies,
    )
    from bot_core.security import (
        KeyringSecretStorage,
        SecretManager,
        SecretStorageError,
    )
except Exception:  # pragma: no cover - brak zależności testowych
    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]
    get_sms_provider = None  # type: ignore[assignment]
    AutoTrader = EmitterLike = RiskDecision = None  # type: ignore[assignment]
    load_core_config = CoreConfig = None  # type: ignore[assignment]
    DatabaseManager = None  # type: ignore[assignment]
    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]
    BootstrapContext = bootstrap_environment = None  # type: ignore[assignment]
    TradingEngine = TradingEngineFactory = None  # type: ignore[assignment]
    TradingParameters = TradingStrategies = None  # type: ignore[assignment]
    KeyringSecretStorage = SecretManager = SecretStorageError = None  # type: ignore[assignment]

import os

if os.environ.get("BOT_CORE_MINIMAL_CORE") == "1":  # pragma: no cover - tryb testowy
    __all__: list[str] = []
else:
    from bot_core.alerts import (  # noqa: WPS433
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
    from bot_core.auto_trader import AutoTrader, EmitterLike, RiskDecision  # noqa: WPS433
    from bot_core.config.loader import load_core_config  # noqa: WPS433
    from bot_core.config.models import CoreConfig  # noqa: WPS433
    from bot_core.database import DatabaseManager  # noqa: WPS433
    from bot_core.exchanges import (  # noqa: WPS433
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
    from bot_core.runtime import BootstrapContext, bootstrap_environment  # noqa: WPS433
    from bot_core.trading import (  # noqa: WPS433
        TradingEngine,
        TradingEngineFactory,
        TradingParameters,
        TradingStrategies,
    )
    from bot_core.security import (  # noqa: WPS433
        KeyringSecretStorage,
        SecretManager,
        SecretStorageError,
    )

    try:  # pragma: no cover - optional dependency
        from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore # noqa: WPS433
    except Exception:  # pragma: no cover - keep package import-safe without observability
        MetricsRegistry = None  # type: ignore[assignment]
        get_global_metrics_registry = None  # type: ignore[assignment]

    __all__ = [
        "AlertChannel",
        "AlertMessage",
        "AutoTrader",
        "DatabaseManager",
        "BootstrapContext",
        "CoreConfig",
        "DEFAULT_SMS_PROVIDERS",
        "EmitterLike",
        "BaseBackend",
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
        "bootstrap_environment",
        "get_sms_provider",
        "load_core_config",
        "SecretManager",
        "SecretStorageError",
        "TradingEngine",
        "TradingEngineFactory",
        "TradingParameters",
        "TradingStrategies",
    ]

    if MetricsRegistry is not None and get_global_metrics_registry is not None:  # pragma: no cover
        __all__ += ["MetricsRegistry", "get_global_metrics_registry"]
if MetricsRegistry is not None and get_global_metrics_registry is not None:  # pragma: no cover
    __all__ += ["MetricsRegistry", "get_global_metrics_registry"]
__all__: list[str] = []
