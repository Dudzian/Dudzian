 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/bot_core/__init__.py b/bot_core/__init__.py
index 1ac66e6100a33dffd9006464787d3a71c0024353..ff460e4948f4f6d59b8fe7d22285a75eb5f8d251 100644
--- a/bot_core/__init__.py
+++ b/bot_core/__init__.py
@@ -1,223 +1,132 @@
 """Podstawowy pakiet runtime bota handlowego."""
 
-from bot_core.alerts import (
-    AlertChannel,
-    AlertMessage,
-    DEFAULT_SMS_PROVIDERS,
-    SmsProviderConfig,
-    get_sms_provider,
-)
-# Nie wszystkie moduły są wymagane w środowisku testowym – importujemy je leniwie,
-# aby uniknąć błędów składniowych w opcjonalnych komponentach.
-try:  # pragma: no cover - moduł auto_trader może nie być kompletny w CI
-    from bot_core.auto_trader import AutoTrader, EmitterLike, RiskDecision
+from __future__ import annotations
+
+from typing import Any, Callable
+
+try:  # pragma: no cover - moduł auto_trader może być opcjonalny w CI
+    from .auto_trader import AutoTrader, EmitterLike, RiskDecision
 except Exception:  # pragma: no cover - zapewniamy bezpieczny import pakietu
     AutoTrader = None  # type: ignore[assignment]
     EmitterLike = object  # type: ignore[assignment]
     RiskDecision = None  # type: ignore[assignment]
-from bot_core.config.loader import load_core_config
-from bot_core.config.models import CoreConfig
-from bot_core.database import DatabaseManager
-try:
-from __future__ import annotations
-try:  # pragma: no cover - defensywne importy podczas testów
-    from bot_core.alerts import (
+
+try:  # pragma: no cover - importy warunkowe dla środowiska testowego
+    from .alerts import (
         AlertChannel,
         AlertMessage,
         DEFAULT_SMS_PROVIDERS,
         SmsProviderConfig,
         get_sms_provider,
     )
-    from bot_core.auto_trader import AutoTrader, EmitterLike, RiskDecision
-    from bot_core.config.loader import load_core_config
-    from bot_core.config.models import CoreConfig
-    from bot_core.database import DatabaseManager
-    from bot_core.exchanges import (
+except Exception:  # pragma: no cover - zapewniamy zgodność, gdy brak zależności
+    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]
+
+    def get_sms_provider(*_: Any, **__: Any) -> Callable[..., Any]:  # type: ignore[misc]
+        raise RuntimeError("SMS providers are not available in this environment")
+
+try:  # pragma: no cover - ładowanie konfiguracji może wymagać zależności opcjonalnych
+    from .config.loader import load_core_config
+except Exception:  # pragma: no cover - fallback gdy loader niedostępny
+    def load_core_config(*_: Any, **__: Any) -> Any:  # type: ignore[override]
+        raise RuntimeError("Core configuration loader is not available")
+
+try:  # pragma: no cover - modele konfiguracyjne mogą nie być obecne
+    from .config.models import CoreConfig
+except Exception:  # pragma: no cover
+    CoreConfig = None  # type: ignore[assignment]
+
+try:  # pragma: no cover - połączenie z bazą danych może wymagać zależności
+    from .database import DatabaseManager
+except Exception:  # pragma: no cover
+    DatabaseManager = None  # type: ignore[assignment]
+
+try:  # pragma: no cover - zależności giełdowe mogą być pominięte
+    from .exchanges import (
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
-except Exception:  # pragma: no cover - utrzymanie kompatybilności importu pakietu
+except Exception:  # pragma: no cover - zapewniamy import nawet bez pełnych modułów
     BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
     MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
     OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]
-from bot_core.runtime import BootstrapContext, bootstrap_environment
-from bot_core.trading import (
-    TradingEngine,
-    TradingEngineFactory,
-    TradingParameters,
-    TradingStrategies,
-)
-from bot_core.security import (
-    KeyringSecretStorage,
-    SecretManager,
-    SecretStorageError,
-)
-    from bot_core.runtime import BootstrapContext, bootstrap_environment
-    from bot_core.trading import (
-        TradingEngine,
-        TradingEngineFactory,
-        TradingParameters,
-        TradingStrategies,
-    )
-    from bot_core.security import (
-        KeyringSecretStorage,
-        SecretManager,
-        SecretStorageError,
-    )
-except Exception:  # pragma: no cover - brak zależności testowych
-    AlertChannel = AlertMessage = DEFAULT_SMS_PROVIDERS = SmsProviderConfig = None  # type: ignore[assignment]
-    get_sms_provider = None  # type: ignore[assignment]
-    AutoTrader = EmitterLike = RiskDecision = None  # type: ignore[assignment]
-    load_core_config = CoreConfig = None  # type: ignore[assignment]
-    DatabaseManager = None  # type: ignore[assignment]
-    BaseBackend = Event = EventBus = ExchangeAdapter = ExchangeManager = None  # type: ignore[assignment]
-    MarketRules = Mode = OrderDTO = OrderResult = None  # type: ignore[assignment]
-    OrderSide = OrderStatus = OrderType = PaperBackend = PositionDTO = None  # type: ignore[assignment]
-    BootstrapContext = bootstrap_environment = None  # type: ignore[assignment]
-    TradingEngine = TradingEngineFactory = None  # type: ignore[assignment]
-    TradingParameters = TradingStrategies = None  # type: ignore[assignment]
-    KeyringSecretStorage = SecretManager = SecretStorageError = None  # type: ignore[assignment]
 
-import os
+try:  # pragma: no cover - środowisko runtime może nie być kompletnie zainstalowane
+    from .runtime import BootstrapContext, bootstrap_environment
+except Exception:  # pragma: no cover
+    BootstrapContext = None  # type: ignore[assignment]
 
-if os.environ.get("BOT_CORE_MINIMAL_CORE") == "1":  # pragma: no cover - tryb testowy
-    __all__: list[str] = []
-else:
-    from bot_core.alerts import (  # noqa: WPS433
-        AlertChannel,
-        AlertMessage,
-        DEFAULT_SMS_PROVIDERS,
-        SmsProviderConfig,
-        get_sms_provider,
-    )
-    from bot_core.auto_trader import AutoTrader, EmitterLike, RiskDecision  # noqa: WPS433
-    from bot_core.config.loader import load_core_config  # noqa: WPS433
-    from bot_core.config.models import CoreConfig  # noqa: WPS433
-    from bot_core.database import DatabaseManager  # noqa: WPS433
-    from bot_core.exchanges import (  # noqa: WPS433
-        BaseBackend,
-        Event,
-        EventBus,
-        ExchangeAdapter,
-        ExchangeManager,
-        MarketRules,
-        Mode,
-        OrderDTO,
-        OrderResult,
-        OrderSide,
-        OrderStatus,
-        OrderType,
-        PaperBackend,
-        PositionDTO,
-    )
-    from bot_core.runtime import BootstrapContext, bootstrap_environment  # noqa: WPS433
-    from bot_core.trading import (  # noqa: WPS433
+    def bootstrap_environment(*_: Any, **__: Any) -> Any:  # type: ignore[override]
+        raise RuntimeError("Runtime bootstrap is not available")
+
+try:  # pragma: no cover - trading może wymagać zależności opcjonalnych
+    from .trading import (
         TradingEngine,
         TradingEngineFactory,
         TradingParameters,
         TradingStrategies,
     )
-    from bot_core.security import (  # noqa: WPS433
+except Exception:  # pragma: no cover
+    TradingEngine = TradingEngineFactory = TradingStrategies = None  # type: ignore[assignment]
+
+    class TradingParameters:  # type: ignore[override]
+        pass
+
+try:  # pragma: no cover - komponenty bezpieczeństwa mogą nie być obecne
+    from .security import (
         KeyringSecretStorage,
         SecretManager,
         SecretStorageError,
     )
+except Exception:  # pragma: no cover
+    KeyringSecretStorage = SecretManager = SecretStorageError = None  # type: ignore[assignment]
 
-    try:  # pragma: no cover - optional dependency
-        from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore # noqa: WPS433
-    except Exception:  # pragma: no cover - keep package import-safe without observability
-        MetricsRegistry = None  # type: ignore[assignment]
-        get_global_metrics_registry = None  # type: ignore[assignment]
 
 __all__ = [
     "AlertChannel",
     "AlertMessage",
-    "DatabaseManager",
+    "AutoTrader",
+    "BaseBackend",
     "BootstrapContext",
     "CoreConfig",
+    "DatabaseManager",
     "DEFAULT_SMS_PROVIDERS",
-    "BaseBackend",
+    "EmitterLike",
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
+    "RiskDecision",
     "SmsProviderConfig",
-    "bootstrap_environment",
-    "get_sms_provider",
-    "load_core_config",
     "SecretManager",
     "SecretStorageError",
     "TradingEngine",
     "TradingEngineFactory",
     "TradingParameters",
     "TradingStrategies",
+    "bootstrap_environment",
+    "get_sms_provider",
+    "load_core_config",
 ]
-    __all__ = [
-        "AlertChannel",
-        "AlertMessage",
-        "AutoTrader",
-        "DatabaseManager",
-        "BootstrapContext",
-        "CoreConfig",
-        "DEFAULT_SMS_PROVIDERS",
-        "EmitterLike",
-        "BaseBackend",
-        "Event",
-        "EventBus",
-        "ExchangeAdapter",
-        "ExchangeManager",
-        "KeyringSecretStorage",
-        "MarketRules",
-        "Mode",
-        "OrderDTO",
-        "OrderResult",
-        "OrderSide",
-        "OrderStatus",
-        "OrderType",
-        "PaperBackend",
-        "PositionDTO",
-        "RiskDecision",
-        "SmsProviderConfig",
-        "bootstrap_environment",
-        "get_sms_provider",
-        "load_core_config",
-        "SecretManager",
-        "SecretStorageError",
-        "TradingEngine",
-        "TradingEngineFactory",
-        "TradingParameters",
-        "TradingStrategies",
-    ]
-
-    if MetricsRegistry is not None and get_global_metrics_registry is not None:  # pragma: no cover
-        __all__ += ["MetricsRegistry", "get_global_metrics_registry"]
-if MetricsRegistry is not None and get_global_metrics_registry is not None:  # pragma: no cover
-    __all__ += ["MetricsRegistry", "get_global_metrics_registry"]
-
-if AutoTrader is not None and RiskDecision is not None:  # pragma: no cover - opcjonalna ekspozycja
-    __all__ += ["AutoTrader", "RiskDecision"]
-if EmitterLike is not object:  # pragma: no cover
-    __all__.append("EmitterLike")
-__all__: list[str] = []
 
EOF
)