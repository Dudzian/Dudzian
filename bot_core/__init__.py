"""Podstawowy pakiet runtime bota handlowego."""

from __future__ import annotations

import os

# Nie wszystkie moduły są wymagane w środowisku testowym – importujemy je
# leniwie, aby utrzymać kompatybilność z lekkimi adapterami.
try:  # pragma: no cover - moduł auto_trader może nie być kompletny w CI
    from bot_core.auto_trader import AutoTrader, EmitterLike, RiskDecision
except Exception:  # pragma: no cover - zapewniamy bezpieczny import pakietu
    AutoTrader = None  # type: ignore[assignment]
    EmitterLike = object  # type: ignore[assignment]
    RiskDecision = None  # type: ignore[assignment]

try:
    from bot_core.alerts import (
        AlertChannel,
        AlertMessage,
        DEFAULT_SMS_PROVIDERS,
        SmsProviderConfig,
        get_sms_provider,
    )
except Exception:  # pragma: no cover - brak kanałów alertowych w środowisku testowym
    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]
    get_sms_provider = None  # type: ignore[assignment]

try:
    from bot_core.config.loader import load_core_config
    from bot_core.config.models import CoreConfig
except Exception:  # pragma: no cover - utrzymanie kompatybilności importu pakietu
    load_core_config = CoreConfig = None  # type: ignore[assignment]

try:
    from bot_core.database import DatabaseManager
except Exception:  # pragma: no cover - brak warstwy bazy danych w testach
    DatabaseManager = None  # type: ignore[assignment]

try:
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
except Exception:  # pragma: no cover - brak zależności giełdowych
    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]

try:
    from bot_core.runtime import BootstrapContext, bootstrap_environment
except Exception:  # pragma: no cover - brak runtime w środowisku testowym
    BootstrapContext = bootstrap_environment = None  # type: ignore[assignment]

try:
    from bot_core.trading import (
        TradingEngine,
        TradingEngineFactory,
        TradingParameters,
        TradingStrategies,
    )
except Exception:  # pragma: no cover - brak silnika tradingowego w testach
    TradingEngine = TradingEngineFactory = None  # type: ignore[assignment]
    TradingParameters = TradingStrategies = None  # type: ignore[assignment]

try:
    from bot_core.security import (
        KeyringSecretStorage,
        SecretManager,
        SecretStorageError,
    )
except Exception:  # pragma: no cover - brak warstwy bezpieczeństwa
    KeyringSecretStorage = SecretManager = SecretStorageError = None  # type: ignore[assignment]

try:  # pragma: no cover - opcjonalne obserwowalności
    from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore # noqa: WPS433
except Exception:  # pragma: no cover - zachowaj kompatybilność bez obserwowalności
    MetricsRegistry = None  # type: ignore[assignment]
    get_global_metrics_registry = None  # type: ignore[assignment]

if os.environ.get("BOT_CORE_MINIMAL_CORE") == "1":  # pragma: no cover - tryb testowy
    __all__: list[str] = []
else:
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

if AutoTrader is not None and RiskDecision is not None:  # pragma: no cover - opcjonalna ekspozycja
    __all__ += ["AutoTrader", "RiskDecision"]
if EmitterLike is not object:  # pragma: no cover - nazwa klasy emitera
    __all__.append("EmitterLike")
