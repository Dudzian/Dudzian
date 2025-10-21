"""Nowa modularna architektura bota handlowego."""

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
except Exception:  # pragma: no cover - utrzymanie kompatybilności importu pakietu
    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]
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

try:  # pragma: no cover - optional dependency
    from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore
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
