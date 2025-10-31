"""Natywna implementacja fasady ExchangeManager."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, MutableMapping
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from datetime import datetime, timezone

import yaml
from pydantic import BaseModel, Field

from bot_core.database.manager import DatabaseManager
from bot_core.exchanges.core import (
    BaseBackend,
    Event,
    EventBus,
    MarketRules,
    Mode,
    OrderDTO,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionDTO,
    PaperBackend,
)
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.health import (
    CircuitBreaker,
    HealthCheck,
    HealthMonitor,
    RetryPolicy,
    Watchdog,
)
from bot_core.exchanges.paper_simulator import PaperFuturesSimulator, PaperMarginSimulator
from bot_core.strategies.catalog import StrategyCatalog, StrategyPresetDescriptor

try:  # pragma: no cover
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None


log = logging.getLogger(__name__)


ORDER_FILLED_EVENT = "ORDER_FILLED"
ACCOUNT_MARK_EVENT = "ACCOUNT_MARK"


def _enable_sandbox_mode(client: Any) -> bool:
    """Próbuje włączyć sandbox/testnet dla przekazanego klienta CCXT.

    Preferowanym mechanizmem jest wywołanie jednej z metod
    ``setSandboxMode``/``set_sandbox_mode``.  Dla starszych adapterów CCXT,
    które nie eksponują tych metod, stosujemy fallback polegający na
    przepięciu adresów końcowych z wariantów ``*Test``.
    """

    for attr in ("setSandboxMode", "set_sandbox_mode"):
        setter = getattr(client, attr, None)
        if callable(setter):
            try:
                setter(True)
            except Exception as exc:  # pragma: no cover - logowanie diagnostyczne
                log.warning("Nie udało się włączyć sandbox mode przez %s: %s", attr, exc)
                return False
            return True

    urls = getattr(client, "urls", None)
    if isinstance(urls, dict):
        remapped = False
        for key, value in list(urls.items()):
            if key.endswith("Test") and value:
                urls[key[:-4]] = value
                remapped = True
        test_url = urls.get("test")
        if test_url:
            urls["api"] = test_url
            remapped = True
        if remapped:
            return True

    return False


@dataclass(slots=True)
class _NativeAdapterRegistration:
    factory: Any
    default_settings: Mapping[str, object]
    supports_testnet: bool = True
    source: Path | None = None
    dynamic: bool = False


_NATIVE_ADAPTER_REGISTRY: Dict[Tuple[Mode, str], _NativeAdapterRegistration] = {}
_DYNAMIC_ADAPTER_KEYS: set[Tuple[Mode, str]] = set()

_EXCHANGE_PROFILE_CACHE: Dict[tuple[str, Path], Mapping[str, Any]] = {}
_REQUIRED_PROFILE_NAMES: tuple[str, ...] = ("paper", "testnet", "live")
_SUPPORTED_MANAGER_MODES: frozenset[str] = frozenset({"paper", "spot", "margin", "futures"})


@dataclass(slots=True)
class _StrategyBinding:
    """Aktualna konfiguracja strategii przypięta do danego kontekstu giełdowego."""

    environment: str
    preset_id: str
    name: str
    profile: str | None
    strategies: Mapping[str, Mapping[str, Any]]
    license_status: Mapping[str, Any]
    metadata: Mapping[str, Any]
    applied_at: datetime

    def as_dict(self) -> Mapping[str, object]:
        return {
            "environment": self.environment,
            "preset_id": self.preset_id,
            "name": self.name,
            "profile": self.profile,
            "strategies": {key: dict(value) for key, value in self.strategies.items()},
            "license_status": dict(self.license_status),
            "metadata": dict(self.metadata),
            "applied_at": self.applied_at.astimezone(timezone.utc).isoformat(),
        }


@dataclass(slots=True)
class _StrategyContextSnapshot:
    """Migawka katalogu strategii dostępnego dla określonego kontekstu."""

    environment: str
    mode: str
    presets: tuple[Mapping[str, Any], ...]
    engines: tuple[Mapping[str, Any], ...]
    assignments: tuple[_StrategyBinding, ...] = field(default_factory=tuple)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> Mapping[str, object]:
        return {
            "environment": self.environment,
            "mode": self.mode,
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "presets": [dict(entry) for entry in self.presets],
            "engines": [dict(entry) for entry in self.engines],
            "assignments": [binding.as_dict() for binding in self.assignments],
        }


class _LegacyAdapterMapping(MutableMapping[str, Any]):
    """Compatybilna z wcześniejszym API mapa natywnych adapterów."""

    def __init__(self, mode: Mode) -> None:
        self._mode = mode

    def __getitem__(self, key: str) -> Any:
        normalized = (key or "").strip().lower()
        entry = _NATIVE_ADAPTER_REGISTRY.get((self._mode, normalized))
        if entry is None:
            raise KeyError(normalized)
        return entry.factory

    def __setitem__(self, key: str, value: Any) -> None:
        register_native_adapter(
            exchange_id=str(key).strip().lower(),
            mode=self._mode,
            factory=value,
        )

    def __delitem__(self, key: str) -> None:
        normalized = (key or "").strip().lower()
        _NATIVE_ADAPTER_REGISTRY.pop((self._mode, normalized), None)

    def __iter__(self):
        for (mode, exchange_id), _ in _NATIVE_ADAPTER_REGISTRY.items():
            if mode == self._mode:
                yield exchange_id

    def __len__(self) -> int:
        return sum(1 for mode, _ in _NATIVE_ADAPTER_REGISTRY if mode == self._mode)


_NATIVE_MARGIN_ADAPTERS: MutableMapping[str, Any] = _LegacyAdapterMapping(Mode.MARGIN)
_NATIVE_FUTURES_ADAPTERS: MutableMapping[str, Any] = _LegacyAdapterMapping(Mode.FUTURES)


_DYNAMIC_ADAPTERS_INITIALIZED = False
_DYNAMIC_ADAPTERS_SOURCE: Path | None = None


@dataclass(frozen=True, slots=True)
class NativeAdapterInfo:
    """Informacje o zarejestrowanym natywnym adapterze giełdy."""

    exchange_id: str
    mode: Mode
    factory: Any
    default_settings: Mapping[str, object]
    supports_testnet: bool
    source: Path | None
    dynamic: bool


def register_native_adapter(
    *,
    exchange_id: str,
    mode: Mode | str,
    factory: Any,
    default_settings: Mapping[str, object] | None = None,
    supports_testnet: bool = True,
    source: str | os.PathLike[str] | None = None,
    dynamic: bool = False,
) -> None:
    """Rejestruje fabrykę adaptera wykorzystywaną przez :class:`ExchangeManager`.

    ``mode`` może być przekazany jako instancja :class:`Mode` lub napis
    ``"margin"``/``"futures"``.  Testy korzystają z tego helpera, aby
    wstrzykiwać lekkie atrapy bez modyfikowania globalnych struktur.
    Rejestracje są idempotentne – ostatnie wywołanie wygrywa, co pozwala
    nadpisywać domyślne ustawienia w razie potrzeby.
    """

    if mode not in {Mode.MARGIN, Mode.FUTURES}:
        raise ValueError("Native adapters are only supported for margin/futures modes")
    key = (mode, exchange_id)
    source_path = Path(source).expanduser() if source is not None else None
    registration = _NativeAdapterRegistration(
        factory=factory,
        default_settings=dict(default_settings or {}),
        supports_testnet=bool(supports_testnet),
        source=source_path,
        dynamic=bool(dynamic),
    )
    _NATIVE_ADAPTER_REGISTRY[key] = registration
    if registration.dynamic:
        _DYNAMIC_ADAPTER_KEYS.add(key)
    else:
        _DYNAMIC_ADAPTER_KEYS.discard(key)


def _import_adapter_factory(path: str) -> Any:
    """Importuje klasę adaptera z notacji modułowej ``module:Class``."""

    module_path, separator, attr_path = path.partition(":")
    if not separator:
        module_path, _, attr_path = path.rpartition(".")
    if not module_path or not attr_path:
        raise ImportError(f"Niepoprawna ścieżka klasy adaptera: {path}")
    module = import_module(module_path)
    target: Any = module
    for attr in attr_path.split("."):
        target = getattr(target, attr)
    return target


def _discover_core_config_candidates() -> list[Path]:
    """Zwraca listę potencjalnych ścieżek do pliku core.yaml."""

    candidates: list[Path] = []
    for env_var in (
        "BOT_CORE_ADAPTER_CONFIG",
        "DUDZIAN_CORE_CONFIG",
        "BOT_CORE_CORE_CONFIG",
    ):
        candidate = os.environ.get(env_var)
        if candidate:
            candidates.append(Path(candidate).expanduser())

    base_dir = Path(__file__).resolve().parents[2]
    candidates.append(base_dir / "config" / "core.yaml")
    candidates.append(Path("config/core.yaml").expanduser())

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _clear_dynamic_native_adapters() -> None:
    """Czyści rejestr adapterów pochodzących z konfiguracji."""

    for key in list(_NATIVE_ADAPTER_REGISTRY):
        registration = _NATIVE_ADAPTER_REGISTRY.get(key)
        if registration is None or not registration.dynamic:
            continue
        _NATIVE_ADAPTER_REGISTRY.pop(key, None)
    _DYNAMIC_ADAPTER_KEYS.clear()


def _resolve_exchange_config_path(
    exchange_id: str,
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> Path:
    normalized = (exchange_id or "").strip().lower()
    if not normalized:
        raise ValueError("exchange_id nie może być pusty")
    if config_dir is not None:
        base_dir = Path(config_dir).expanduser().resolve()
    else:
        override = os.environ.get("BOT_CORE_EXCHANGE_CONFIG_DIR")
        if override:
            base_dir = Path(override).expanduser().resolve()
        else:
            base_dir = Path(__file__).resolve().parents[2] / "config" / "exchanges"
    return base_dir / f"{normalized}.yaml"


def _load_exchange_profiles(
    exchange_id: str,
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> Mapping[str, Any]:
    path = _resolve_exchange_config_path(exchange_id, config_dir=config_dir)
    cache_key = (exchange_id.strip().lower(), path)
    cached = _EXCHANGE_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Nie znaleziono konfiguracji giełdy: {path}") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Plik konfiguracji {path} musi zawierać mapę profili")
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(value, Mapping):
            continue
        normalized[str(key).strip().lower()] = value
    _validate_exchange_profiles(exchange_id, path, normalized)
    _EXCHANGE_PROFILE_CACHE[cache_key] = normalized
    return normalized


def _validate_exchange_profiles(
    exchange_id: str,
    path: Path,
    profiles: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = [name for name in _REQUIRED_PROFILE_NAMES if name not in profiles]
    if missing:
        raise ValueError(
            "Konfiguracja %s dla giełdy %s nie posiada profili: %s (wymagane: %s)"
            % (
                path,
                exchange_id,
                ", ".join(sorted(missing)),
                ", ".join(_REQUIRED_PROFILE_NAMES),
            ),
        )

    for name in _REQUIRED_PROFILE_NAMES:
        profile = profiles.get(name)
        if not isinstance(profile, Mapping):
            raise ValueError(
                f"Profil '{name}' w konfiguracji {path} musi być słownikiem ustawień.",
            )

        manager_cfg = profile.get("exchange_manager")
        if not isinstance(manager_cfg, Mapping):
            raise ValueError(
                f"Profil '{name}' w konfiguracji {path} wymaga sekcji 'exchange_manager'.",
            )

        raw_mode = manager_cfg.get("mode")
        mode_value = str(raw_mode or "").strip().lower()
        if mode_value not in _SUPPORTED_MANAGER_MODES:
            raise ValueError(
                "Profil '%s' w konfiguracji %s zawiera nieobsługiwany tryb '%s'. Dozwolone: %s"
                % (name, path, raw_mode, ", ".join(sorted(_SUPPORTED_MANAGER_MODES))),
            )

        if name == "paper" and mode_value != "paper":
            raise ValueError(
                f"Profil 'paper' w konfiguracji {path} musi mieć mode ustawiony na 'paper'.",
            )

        if name != "paper":
            if mode_value == "paper":
                raise ValueError(
                    f"Profil '{name}' w konfiguracji {path} nie może używać trybu 'paper'.",
                )
            testnet_flag = manager_cfg.get("testnet")
            if not isinstance(testnet_flag, bool):
                raise ValueError(
                    f"Profil '{name}' w konfiguracji {path} wymaga boolowskiego pola 'testnet'.",
                )

        credentials_cfg = profile.get("credentials")
        if not isinstance(credentials_cfg, Mapping):
            raise ValueError(
                f"Profil '{name}' w konfiguracji {path} wymaga sekcji 'credentials'.",
            )
        for required_key in ("api_key", "secret"):
            if required_key not in credentials_cfg:
                raise ValueError(
                    "Profil '%s' w konfiguracji %s wymaga klucza credentials.%s"
                    % (name, path, required_key),
                )


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = value
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _expand_env_values(payload: Any) -> Any:
    if isinstance(payload, str):
        return os.path.expandvars(payload)
    if isinstance(payload, Mapping):
        return {key: _expand_env_values(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        expanded = [_expand_env_values(item) for item in payload]
        return type(payload)(expanded)  # type: ignore[call-arg]
    return payload


def _load_raw_exchange_adapters(candidate: Path) -> Mapping[str, Any] | None:
    """Wczytuje sekcję ``exchange_adapters`` bez pełnego parsera konfiguracji."""

    try:
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:  # pragma: no cover - raport diagnostyczny
        log.debug("Nie odnaleziono alternatywnej konfiguracji adapterów %s: %s", candidate, exc)
        return None
    except Exception as exc:  # pragma: no cover - diagnostyka formatu YAML
        log.debug("Nie udało się sparsować %s: %s", candidate, exc)
        return None

    adapters = payload.get("exchange_adapters") or {}
    if not isinstance(adapters, Mapping):
        return {}
    return adapters


def _iter_exchange_adapter_entries(
    adapters: Mapping[str, Any],
) -> Iterator[tuple[Mode, str, str, Mapping[str, object], bool]]:
    """Normalizuje wpisy adapterów niezależnie od użytego loadera."""

    for exchange_id, modes in adapters.items():
        if not isinstance(modes, Mapping):
            continue
        normalized_exchange = str(exchange_id or "").strip().lower()
        if not normalized_exchange:
            continue
        for raw_mode, entry in modes.items():
            mode: Mode
            if isinstance(raw_mode, Mode):
                mode = raw_mode
            else:
                try:
                    mode = Mode(str(raw_mode).lower())
                except Exception:
                    continue

            if hasattr(entry, "class_path"):
                class_path = getattr(entry, "class_path", "")
                default_settings = getattr(entry, "default_settings", {}) or {}
                supports_testnet = getattr(entry, "supports_testnet", True)
            elif isinstance(entry, Mapping):
                class_path = entry.get("class_path") or ""
                default_settings = entry.get("default_settings") or {}
                supports_testnet = entry.get("supports_testnet", True)
            else:
                continue

            class_path_text = str(class_path or "").strip()
            if not class_path_text:
                continue

            if isinstance(default_settings, Mapping):
                normalized_settings = dict(default_settings)
            else:
                try:
                    normalized_settings = dict(default_settings)
                except Exception:
                    normalized_settings = {}

            yield mode, normalized_exchange, class_path_text, normalized_settings, bool(supports_testnet)


def _load_dynamic_native_adapters(
    config_path: str | os.PathLike[str] | Path | None = None,
) -> None:
    """Ładuje definicje adapterów z config/core.yaml, jeśli dostępne."""

    global _DYNAMIC_ADAPTERS_INITIALIZED, _DYNAMIC_ADAPTERS_SOURCE
    if _DYNAMIC_ADAPTERS_INITIALIZED and config_path is None:
        return

    try:
        from bot_core.config.loader import load_core_config  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - opcjonalny loader
        log.debug("Pominięto dynamiczne adaptery – loader niedostępny: %s", exc)
        _DYNAMIC_ADAPTERS_INITIALIZED = True
        return

    if config_path is not None:
        candidates = [Path(config_path).expanduser()]
    else:
        candidates = _discover_core_config_candidates()

    for candidate in candidates:
        adapters_payload: Mapping[str, Any] | None = None
        try:
            config = load_core_config(candidate)
        except Exception as exc:  # pragma: no cover - diagnostyka konfiguracji
            log.debug("Nie udało się wczytać %s przez loader: %s", candidate, exc)
        else:
            adapters_payload = getattr(config, "exchange_adapters", None) or {}

        if adapters_payload is None:
            adapters_payload = _load_raw_exchange_adapters(candidate)
            if adapters_payload is None:
                continue

        seen_entries = False
        registered_any = False
        for mode, exchange, class_path, default_settings, supports_testnet in _iter_exchange_adapter_entries(adapters_payload):
            seen_entries = True
            key = (mode, exchange)
            if key in _NATIVE_ADAPTER_REGISTRY:
                continue
            try:
                factory = _import_adapter_factory(class_path)
            except Exception as exc:  # pragma: no cover - diagnostyka importu
                log.warning(
                    "Pominięto adapter %s/%s z powodu błędu importu: %s",
                    exchange,
                    mode.value,
                    exc,
                )
                continue
            register_native_adapter(
                exchange_id=exchange,
                mode=mode,
                factory=factory,
                default_settings=default_settings,
                supports_testnet=supports_testnet,
                source=candidate,
                dynamic=True,
            )
            registered_any = True

        if seen_entries:
            _DYNAMIC_ADAPTERS_SOURCE = candidate
        if registered_any or seen_entries:
            break

    _DYNAMIC_ADAPTERS_INITIALIZED = True


def reload_native_adapters(
    config_path: str | os.PathLike[str] | Path | None = None,
) -> None:
    """Czyści i ponownie ładuje adaptery konfiguracyjne."""

    global _DYNAMIC_ADAPTERS_INITIALIZED, _DYNAMIC_ADAPTERS_SOURCE

    _clear_dynamic_native_adapters()
    _DYNAMIC_ADAPTERS_INITIALIZED = False
    _DYNAMIC_ADAPTERS_SOURCE = None
    _load_dynamic_native_adapters(config_path)


def iter_registered_native_adapters(
    mode: Mode | None = None,
) -> Iterator[NativeAdapterInfo]:
    """Udostępnia metadane wszystkich zarejestrowanych natywnych adapterów."""

    if mode is not None and mode not in {Mode.MARGIN, Mode.FUTURES}:
        raise ValueError("Filtrowanie obsługuje tylko tryby margin lub futures")

    for (registered_mode, exchange_id), registration in sorted(
        _NATIVE_ADAPTER_REGISTRY.items(),
        key=lambda item: (item[0][0].value, item[0][1]),
    ):
        if mode is not None and registered_mode is not mode:
            continue
        yield NativeAdapterInfo(
            exchange_id=exchange_id,
            mode=registered_mode,
            factory=registration.factory,
            default_settings=dict(registration.default_settings),
            supports_testnet=registration.supports_testnet,
            source=registration.source,
            dynamic=registration.dynamic,
        )

def get_native_adapter_info(*, exchange_id: str, mode: Mode) -> NativeAdapterInfo | None:
    """Zwraca metadane pojedynczego zarejestrowanego adaptera."""

    if mode not in {Mode.MARGIN, Mode.FUTURES}:
        raise ValueError(
            "Informacje o adapterach dostępne są tylko dla trybów margin/futures",
        )

    normalized_exchange = str(exchange_id or "").strip().lower()
    if not normalized_exchange:
        raise ValueError("exchange_id nie może być pusty")

    registration = _NATIVE_ADAPTER_REGISTRY.get((mode, normalized_exchange))
    if registration is None:
        return None

    return NativeAdapterInfo(
        exchange_id=normalized_exchange,
        mode=mode,
        factory=registration.factory,
        default_settings=dict(registration.default_settings),
        supports_testnet=registration.supports_testnet,
        source=registration.source,
        dynamic=registration.dynamic,
    )


def unregister_native_adapter(
    *,
    exchange_id: str,
    mode: Mode,
    allow_dynamic: bool = False,
) -> bool:
    """Usuwa zarejestrowany adapter natywny, jeśli istnieje.

    Dynamiczne wpisy z konfiguracji są domyślnie chronione przed usunięciem –
    aby je skasować należy ustawić ``allow_dynamic=True`` i ewentualnie ponownie
    załadować konfigurację funkcją :func:`reload_native_adapters`.
    """

    if mode not in {Mode.MARGIN, Mode.FUTURES}:
        raise ValueError("Usuwanie adapterów wspiera tylko tryby margin/futures")

    normalized_exchange = str(exchange_id or "").strip().lower()
    if not normalized_exchange:
        raise ValueError("exchange_id nie może być pusty")

    key = (mode, normalized_exchange)
    registration = _NATIVE_ADAPTER_REGISTRY.get(key)
    if registration is None:
        return False
    if registration.dynamic and not allow_dynamic:
        return False

    _NATIVE_ADAPTER_REGISTRY.pop(key, None)
    _DYNAMIC_ADAPTER_KEYS.discard(key)
    return True


def _import_adapter_factory(path: str) -> Any:
    """Importuje klasę adaptera z notacji modułowej ``module:Class``."""

    module_path, separator, attr_path = path.partition(":")
    if not separator:
        module_path, _, attr_path = path.rpartition(".")
    if not module_path or not attr_path:
        raise ImportError(f"Niepoprawna ścieżka klasy adaptera: {path}")
    module = import_module(module_path)
    target: Any = module
    for attr in attr_path.split("."):
        target = getattr(target, attr)
    return target


def _discover_core_config_candidates() -> list[Path]:
    """Zwraca listę potencjalnych ścieżek do pliku core.yaml."""

    candidates: list[Path] = []
    for env_var in (
        "BOT_CORE_ADAPTER_CONFIG",
        "DUDZIAN_CORE_CONFIG",
        "BOT_CORE_CORE_CONFIG",
    ):
        candidate = os.environ.get(env_var)
        if candidate:
            candidates.append(Path(candidate).expanduser())

    base_dir = Path(__file__).resolve().parents[2]
    candidates.append(base_dir / "config" / "core.yaml")
    candidates.append(Path("config/core.yaml").expanduser())

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _clear_dynamic_native_adapters() -> None:
    """Czyści rejestr adapterów pochodzących z konfiguracji."""

    for key in list(_NATIVE_ADAPTER_REGISTRY):
        registration = _NATIVE_ADAPTER_REGISTRY.get(key)
        if registration is None or not registration.dynamic:
            continue
        _NATIVE_ADAPTER_REGISTRY.pop(key, None)
    _DYNAMIC_ADAPTER_KEYS.clear()


_STATUS_MAPPING = {
    "NEW": OrderStatus.OPEN,
    "OPEN": OrderStatus.OPEN,
    "PENDING_NEW": OrderStatus.OPEN,
    "PENDING": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "PARTIALLY": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "CANCELLED": OrderStatus.CANCELED,
    "PENDING_CANCEL": OrderStatus.CANCELED,
    "EXPIRED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
}


def _map_order_status(raw: object) -> OrderStatus:
    if isinstance(raw, OrderStatus):
        return raw
    value = str(raw or "").upper()
    return _STATUS_MAPPING.get(value, OrderStatus.OPEN)


def _map_order_side(raw: object) -> OrderSide:
    if isinstance(raw, OrderSide):
        return raw
    return OrderSide.BUY if str(raw or "").upper() == "BUY" else OrderSide.SELL


def _map_order_type(raw: object) -> OrderType:
    if isinstance(raw, OrderType):
        return raw
    value = str(raw or "").upper()
    if value == "LIMIT":
        return OrderType.LIMIT
    if value == "MARKET":
        return OrderType.MARKET
    return OrderType.MARKET if "MARKET" in value else OrderType.LIMIT


class _CCXTPublicFeed(BaseBackend):
    """Backend publiczny CCXT używany do cen oraz reguł rynku."""

    def __init__(
        self,
        exchange_id: str = "binance",
        testnet: bool = False,
        futures: bool = False,
        market_type: str | None = None,
        *,
        error_handler: Callable[[str, Exception], None] | None = None,
    ) -> None:
        super().__init__(event_bus=EventBus())
        if ccxt is None:
            raise RuntimeError("CCXT nie jest zainstalowane.")
        self.exchange_id = exchange_id
        self.testnet = bool(testnet)
        normalized_type = (market_type or ("future" if futures else "spot")).strip().lower()
        if normalized_type not in {"spot", "margin", "future"}:
            log.warning(
                "Nieznany market_type=%s – domyślnie użyjemy 'spot' dla %s",
                normalized_type,
                exchange_id,
            )
            normalized_type = "spot"
        self.market_type = normalized_type
        self.futures = self.market_type == "future"
        self.client = getattr(ccxt, exchange_id)(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": self.market_type,
                },
            }
        )
        if self.testnet:
            enabled = _enable_sandbox_mode(self.client)
            if not enabled:
                log.warning(
                    "Sandbox mode żądany dla %s, lecz klient CCXT nie udostępnia trybu testowego.",
                    exchange_id,
                )
        self._markets: Dict[str, Any] = {}
        self._rules: Dict[str, MarketRules] = {}
        self._error_handler = error_handler
        network_error_cls = getattr(ccxt, "NetworkError", None)
        if isinstance(network_error_cls, tuple):
            self._network_error_classes: Tuple[type[Exception], ...] = network_error_cls
        elif isinstance(network_error_cls, type):
            self._network_error_classes = (network_error_cls,)
        else:
            self._network_error_classes = tuple()

    def load_markets(self) -> Dict[str, MarketRules]:
        self._markets = self.client.load_markets()
        rules: Dict[str, MarketRules] = {}
        for symbol, meta in self._markets.items():
            limits = (meta.get("limits") or {})
            amount_limits = limits.get("amount") or {}
            price_limits = limits.get("price") or {}
            precision = meta.get("precision") or {}
            amount_step = amount_limits.get("step", 0.0) or (
                (10 ** -float(precision.get("amount", 8))) if precision.get("amount") is not None else 0.0
            )
            price_step = price_limits.get("step", 0.0) or (
                (10 ** -float(precision.get("price", 8))) if precision.get("price") is not None else 0.0
            )
            min_notional = (limits.get("cost") or {}).get("min", 0.0) or 0.0
            rules[symbol] = MarketRules(
                symbol=symbol,
                price_step=float(price_step or 0.0),
                amount_step=float(amount_step or 0.0),
                min_notional=float(min_notional or 0.0),
                min_amount=float(amount_limits.get("min") or 0.0),
                max_amount=float(amount_limits.get("max") or 0.0)
                if amount_limits.get("max") is not None
                else None,
                min_price=float(price_limits.get("min") or 0.0)
                if price_limits.get("min") is not None
                else None,
                max_price=float(price_limits.get("max") or 0.0)
                if price_limits.get("max") is not None
                else None,
            )
        self._rules = rules
        return rules

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        return self._rules.get(symbol)

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return self.client.fetch_ticker(symbol)
        except Exception as exc:
            log.warning("fetch_ticker failed: %s", exc)
            self._handle_network_error("fetch_ticker", exc)
            return None

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[List[List[float]]]:
        try:
            return self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            log.warning("fetch_ohlcv failed: %s", exc)
            self._handle_network_error("fetch_ohlcv", exc)
            return None

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        try:
            return self.client.fetch_order_book(symbol, limit=limit)
        except Exception as exc:
            log.warning("fetch_order_book failed: %s", exc)
            self._handle_network_error("fetch_order_book", exc)
            return None

    def _handle_network_error(self, operation: str, exc: Exception) -> None:
        if not self._error_handler or not self._network_error_classes:
            return
        if isinstance(exc, self._network_error_classes):
            try:
                self._error_handler(operation, exc)
            except Exception:  # pragma: no cover - defensywnie
                log.debug("Network error handler raised", exc_info=True)

    def create_order(self, *args, **kwargs):  # pragma: no cover - interfejs
        raise NotImplementedError

    def cancel_order(self, *args, **kwargs):  # pragma: no cover - interfejs
        raise NotImplementedError

    def fetch_open_orders(self, *args, **kwargs) -> List[OrderDTO]:  # pragma: no cover
        return []

    def fetch_positions(self, *args, **kwargs) -> List[PositionDTO]:  # pragma: no cover
        return []


class _CCXTPrivateBackend(_CCXTPublicFeed):
    """Prywatny backend CCXT dla trybów SPOT/MARGIN/FUTURES."""

    def __init__(
        self,
        exchange_id: str = "binance",
        testnet: bool = False,
        futures: bool = False,
        *,
        market_type: str | None = None,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        error_handler: Callable[[str, Exception], None] | None = None,
    ) -> None:
        resolved_market_type = market_type or ("future" if futures else "spot")
        super().__init__(
            exchange_id=exchange_id,
            testnet=testnet,
            futures=futures,
            market_type=resolved_market_type,
            error_handler=error_handler,
        )
        if ccxt is None:
            raise RuntimeError("CCXT nie jest zainstalowane.")
        options: Dict[str, Any] = {
            "enableRateLimit": True,
            "apiKey": api_key or "",
            "secret": secret or "",
            "options": {"defaultType": self.market_type},
        }
        if passphrase:
            options["password"] = passphrase
        self.client = getattr(ccxt, exchange_id)(options)
        if testnet and not _enable_sandbox_mode(self.client):
            log.warning(
                "Sandbox mode żądany dla %s (%s), lecz klient CCXT nie udostępnia trybu testowego.",
                exchange_id,
                self.market_type,
            )
        self.futures = self.market_type == "future"
        if self.market_type == "margin":
            self.mode = Mode.MARGIN
        elif self.futures:
            self.mode = Mode.FUTURES
        else:
            self.mode = Mode.SPOT

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        if not self._rules:
            self.load_markets()
        rules = self.get_market_rules(symbol)
        if not rules:
            raise RuntimeError(f"Brak reguł rynku dla {symbol}. Najpierw 'Load Markets'.")

        qty = rules.quantize_amount(float(quantity))
        if qty <= 0:
            raise ValueError("Ilość po kwantyzacji = 0.")

        px = None
        params: Dict[str, Any] = {}
        if type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Cena wymagana dla LIMIT.")
            px = rules.quantize_price(float(price))

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        if type == OrderType.MARKET:
            ticker = self.fetch_ticker(symbol) or {}
            last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
            if not last:
                raise RuntimeError(f"Brak ceny MARKET dla {symbol}.")
            notional = qty * float(last)
        else:
            notional = qty * float(px)

        min_notional = rules.min_notional or 0.0
        if min_notional and notional < min_notional:
            raise ValueError(
                f"Notional {notional:.8f} < minNotional {min_notional:.8f} dla {symbol}"
            )

        ccxt_type = "market" if type == OrderType.MARKET else "limit"
        ccxt_side = side.value.lower()
        response = self.client.create_order(symbol, ccxt_type, ccxt_side, qty, px, params)
        order_id = response.get("id") or response.get("orderId")
        status = response.get("status", "open").upper()
        if status == "CLOSED":
            status = "FILLED"

        return OrderDTO(
            id=int(order_id) if isinstance(order_id, str) and order_id.isdigit() else order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=type,
            quantity=qty,
            price=px,
            status=OrderStatus(status),
            mode=self.mode,
        )

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        try:
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            log.error("cancel_order failed: %s", exc)
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        try:
            orders = (
                self.client.fetch_open_orders(symbol)
                if symbol
                else self.client.fetch_open_orders()
            )
        except Exception as exc:
            log.error("fetch_open_orders failed: %s", exc)
            return []

        out: List[OrderDTO] = []
        for entry in orders:
            status = (entry.get("status") or "open").upper()
            if status == "CLOSED":
                status = "FILLED"
            out.append(
                OrderDTO(
                    id=entry.get("id") or entry.get("orderId"),
                    client_order_id=entry.get("clientOrderId")
                    or entry.get("info", {}).get("clientOrderId"),
                    symbol=entry.get("symbol"),
                    side=OrderSide.BUY
                    if (entry.get("side", "").lower() == "buy")
                    else OrderSide.SELL,
                    type=OrderType.MARKET
                    if (entry.get("type", "").lower() == "market")
                    else OrderType.LIMIT,
                    quantity=float(entry.get("amount") or entry.get("filled") or 0.0),
                    price=float(entry.get("price") or 0.0) if entry.get("price") else None,
                    status=OrderStatus(status),
                    mode=self.mode,
                )
            )
        return out

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if not self.futures:
            return []
        try:
            positions = self.client.fetch_positions([symbol] if symbol else None)
        except Exception as exc:
            log.error("fetch_positions failed: %s", exc)
            return []

        out: List[PositionDTO] = []
        for position in positions or []:
            amount = float(position.get("contracts") or position.get("amount") or 0.0)
            if abs(amount) < 1e-12:
                continue
            side = "LONG" if amount > 0 else "SHORT"
            out.append(
                PositionDTO(
                    symbol=position.get("symbol"),
                    side=side,
                    quantity=abs(amount),
                    avg_price=float(position.get("entryPrice") or 0.0),
                    unrealized_pnl=float(position.get("unrealizedPnl") or 0.0),
                    mode=Mode.FUTURES,
                )
            )
        return out

    def fetch_balance(self) -> Dict[str, Any]:
        try:
            return self.client.fetch_balance()
        except Exception as exc:
            log.error("fetch_balance failed: %s", exc)
            return {}


class ExchangeManager:
    """Fasada do obsługi wymiany w trybach paper/spot/futures."""

    def __init__(
        self,
        exchange_id: str = "binance",
        *,
        paper_initial_cash: float = 10_000.0,
        paper_cash_asset: str = "USDT",
        db_url: Optional[str] = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.mode: Mode = Mode.PAPER
        self._testnet: bool = False
        self._futures: bool = False
        self._api_key: Optional[str] = None
        self._secret: Optional[str] = None
        self._passphrase: Optional[str] = None

        self._event_bus = EventBus()
        self._public: Optional[_CCXTPublicFeed] = None
        self._private: Optional[_CCXTPrivateBackend] = None
        self._paper: Optional[PaperBackend] = None
        self._paper_initial_cash = float(paper_initial_cash)
        self._paper_cash_asset = paper_cash_asset.upper()
        self._paper_fee_rate = getattr(PaperBackend, "FEE_RATE", 0.001)
        self._paper_variant = "spot"
        self._paper_simulator_settings: Dict[str, object] = {}
        self._db_url = db_url or "sqlite+aiosqlite:///trading.db"
        self._db: Optional[DatabaseManager] = None
        self._db_failed: bool = False
        self._native_adapter = None
        self._native_adapter_settings: Dict[tuple[Mode, str], Dict[str, object]] = {}
        self._watchdog: Watchdog | None = None
        default_margin_type = os.getenv("BINANCE_MARGIN_TYPE")
        if self.exchange_id == "binance" and default_margin_type:
            self._native_adapter_settings[(Mode.MARGIN, self.exchange_id)] = {
                "margin_type": default_margin_type,
            }

        log.info("ExchangeManager initialized (bot_core)")
        self._network_error_counts: Counter[str] = Counter()
        self._environment_profile: Dict[str, Any] | None = None
        self._environment_profile_name: str | None = None
        self._last_mark_signature: str | None = None
        self._strategy_catalog: StrategyCatalog | None = None
        self._strategy_contexts: Dict[str, _StrategyContextSnapshot] = {}
        self._strategy_assignments: Dict[str, _StrategyBinding] = {}

    @property
    def event_bus(self) -> EventBus:
        """Udostępnia magistralę zdarzeń giełdy dla modułów runtime."""

        return self._event_bus

    def publish_event(self, event_type: str, payload: Mapping[str, Any] | None = None) -> None:
        """Publikuje zdarzenie w wewnętrznej magistrali ExchangeManagera."""

        event_payload = dict(payload or {})
        self._event_bus.publish(Event(type=event_type, payload=event_payload))

    def set_mode(
        self,
        *,
        paper: bool = False,
        spot: bool = False,
        margin: bool = False,
        futures: bool = False,
        testnet: bool = False,
    ) -> None:
        selected = [paper, spot, margin, futures]
        if sum(1 for flag in selected if flag) > 1:
            raise ValueError("Można wybrać tylko jeden tryb: paper, spot, margin lub futures.")

        if paper:
            self.mode = Mode.PAPER
            self._futures = False
            self._testnet = False
        elif futures:
            self.mode = Mode.FUTURES
            self._futures = True
            self._testnet = bool(testnet)
        elif margin:
            self.mode = Mode.MARGIN
            self._futures = False
            self._testnet = bool(testnet)
        else:
            self.mode = Mode.SPOT
            self._futures = False
            self._testnet = bool(testnet)

        log.info("Mode set to %s (futures=%s, testnet=%s)", self.mode.value, self._futures, self._testnet)
        self._private = None
        self._paper = None
        self._native_adapter = None
        self._rebuild_strategy_contexts()

    def set_paper_variant(self, variant: str) -> None:
        """Selects paper simulator flavour (``spot``/``margin``/``futures``)."""

        normalized = (variant or "").strip().lower()
        if normalized not in {"spot", "margin", "futures"}:
            raise ValueError("Paper variant must be one of: spot, margin, futures")
        if self._paper_variant != normalized:
            log.info("Switching paper simulator variant to %s", normalized)
        self._paper_variant = normalized
        self._paper = None

    def configure_paper_simulator(self, **settings: object) -> None:
        """Stores custom parameters used by margin/futures simulators."""

        normalized: Dict[str, float] = {}
        for key, value in settings.items():
            if value is None:
                continue
            try:
                float_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Parametr symulatora '{key}' wymaga liczbowej wartości."
                ) from exc
            if key == "funding_interval_seconds" and float_value <= 0:
                raise ValueError(
                    "Parametr funding_interval_seconds wymaga dodatniej wartości (sekundy)."
                )
            normalized[key] = float_value

        if not normalized:
            return

        merged = dict(self._paper_simulator_settings)
        merged.update(normalized)
        self._paper_simulator_settings = merged
        if self._paper is not None:
            log.info("Reconfiguring paper simulator with settings: %s", merged)
            self._paper = None
        self._paper = None

    def set_credentials(
        self,
        api_key: Optional[str],
        secret: Optional[str],
        *,
        passphrase: Optional[str] = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._secret = (secret or "").strip() or None
        self._passphrase = (passphrase or "").strip() or None
        api_key_length = len(self._api_key or "")
        secret_length = len(self._secret or "")
        passphrase_length = len(self._passphrase or "")
        log.info(
            "Credentials set (lengths): api_key=%d, secret=%d, passphrase=%d",
            api_key_length,
            secret_length,
            passphrase_length,
        )
        self._native_adapter = None
        self._private = None

    def configure_native_adapter(
        self,
        *,
        settings: Mapping[str, object],
        mode: Mode | None = None,
    ) -> None:
        if not isinstance(settings, Mapping):
            raise TypeError("Konfiguracja adaptera musi być mapowaniem.")
        target_mode = mode or self.mode
        if target_mode not in {Mode.MARGIN, Mode.FUTURES}:
            raise ValueError("Konfiguracja natywnego adaptera jest dostępna tylko dla trybów margin/futures.")
        self._native_adapter_settings[(target_mode, self.exchange_id)] = dict(settings)
        self._native_adapter = None

    def set_watchdog(self, watchdog: Watchdog | None) -> None:
        """Ustawia współdzielony watchdog dla natywnych adapterów margin/futures."""

        if watchdog is not None and not isinstance(watchdog, Watchdog):
            raise TypeError("Watchdog musi być instancją klasy bot_core.exchanges.health.Watchdog")
        self._watchdog = watchdog
        self._native_adapter = None

    def configure_watchdog(
        self,
        *,
        retry_policy: Mapping[str, object] | None = None,
        circuit_breaker: Mapping[str, object] | None = None,
        retry_exceptions: Sequence[type[Exception]] | None = None,
    ) -> None:
        """Buduje i ustawia watchdog na podstawie przekazanych parametrów."""

        kwargs: Dict[str, object] = {}
        if retry_policy is not None:
            if not isinstance(retry_policy, Mapping):
                raise TypeError("retry_policy musi być mapowaniem z parametrami RetryPolicy")
            kwargs["retry_policy"] = RetryPolicy(**dict(retry_policy))
        if circuit_breaker is not None:
            if not isinstance(circuit_breaker, Mapping):
                raise TypeError("circuit_breaker musi być mapowaniem z parametrami CircuitBreaker")
            kwargs["circuit_breaker"] = CircuitBreaker(**dict(circuit_breaker))
        if retry_exceptions is not None:
            if not isinstance(retry_exceptions, Sequence):
                raise TypeError("retry_exceptions musi być sekwencją klas wyjątków")
            normalized: list[type[Exception]] = []
            for exc in retry_exceptions:
                if not isinstance(exc, type) or not issubclass(exc, Exception):
                    raise TypeError("retry_exceptions musi zawierać klasy wyjątków")
                normalized.append(exc)
            kwargs["retry_exceptions"] = tuple(normalized)
        self._watchdog = Watchdog(**kwargs)
        self._native_adapter = None

    # ------------------------------------------------------------------
    # Environment profiles
    # ------------------------------------------------------------------
    def load_environment_profile(
        self,
        name: str,
        *,
        exchange: str | None = None,
        config_dir: str | os.PathLike[str] | None = None,
    ) -> Mapping[str, Any]:
        profiles = _load_exchange_profiles(exchange or self.exchange_id, config_dir=config_dir)
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("Nazwa profilu środowiska nie może być pusta")
        profile = profiles.get(key)
        if profile is None:
            raise KeyError(f"Nie znaleziono profilu '{key}' dla giełdy {self.exchange_id}")
        return profile

    def apply_environment_profile(
        self,
        name: str,
        *,
        exchange: str | None = None,
        config_dir: str | os.PathLike[str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        profile = dict(self.load_environment_profile(name, exchange=exchange, config_dir=config_dir))
        if overrides:
            profile = _deep_merge(profile, overrides)
        expanded = _expand_env_values(profile)
        manager_cfg = expanded.get("exchange_manager")
        if isinstance(manager_cfg, Mapping):
            self._apply_manager_profile(manager_cfg)
        credentials_cfg = expanded.get("credentials")
        if isinstance(credentials_cfg, Mapping):
            self.set_credentials(
                credentials_cfg.get("api_key"),
                credentials_cfg.get("secret"),
                passphrase=credentials_cfg.get("passphrase"),
            )
        self._environment_profile = {key: value for key, value in expanded.items() if key != "credentials"}
        self._environment_profile_name = (name or "").strip().lower()

    def describe_environment_profile(self) -> Mapping[str, Any] | None:
        if self._environment_profile is None:
            return None
        description = dict(self._environment_profile)
        if self._environment_profile_name:
            description.setdefault("name", self._environment_profile_name)
        return description

    # ------------------------------------------------------------------
    # Synchronizacja katalogu strategii z kontekstami giełdowymi
    # ------------------------------------------------------------------

    def set_strategy_catalog(self, catalog: StrategyCatalog | None) -> None:
        """Podpina katalog strategii wykorzystywany przez UI i runtime."""

        if catalog is self._strategy_catalog:
            return
        self._strategy_catalog = catalog
        self._rebuild_strategy_contexts()

    def activate_strategy_preset(
        self,
        environment: str,
        preset_id: str,
        *,
        parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Mapping[str, object]:
        """Przypina preset strategii do wskazanego kontekstu giełdowego.

        Preset wskazywany przez ``preset_id`` musi być zgodny ze schematem
        opisanym w ``bot_core.strategies.catalog.PRESET_SCHEMA_DOC``,
        tj. zawierać pola ``name``, ``metadata.id`` (lub ``metadata.preset_id``)
        oraz listę ``strategies`` z wpisami definiującymi ``engine`` i
        ``parameters``. Dzięki temu UI i runtime mogą jednoznacznie odtworzyć
        konfigurację strategii na rachunku live lub papierowym.
        """

        normalized_env = self._normalize_strategy_environment(environment)
        self._ensure_strategy_catalog()
        descriptor = self._resolve_preset(descriptor_id=preset_id)

        strategies: Dict[str, Dict[str, Any]] = {}
        for entry in descriptor.strategies:
            strategy_name = str(entry.get("name") or entry.get("engine") or descriptor.preset_id)
            base_params = dict(entry.get("parameters") or {})
            if parameter_overrides and strategy_name in parameter_overrides:
                override = parameter_overrides[strategy_name]
                base_params.update({str(key): value for key, value in override.items()})
            strategies[strategy_name] = base_params

        binding = _StrategyBinding(
            environment=normalized_env,
            preset_id=descriptor.preset_id,
            name=descriptor.name,
            profile=descriptor.profile.value if hasattr(descriptor.profile, "value") else str(descriptor.profile),
            strategies=strategies,
            license_status=descriptor.license_status.as_dict(),
            metadata=dict(descriptor.metadata),
            applied_at=datetime.now(timezone.utc),
        )
        self._strategy_assignments[normalized_env] = binding
        self._rebuild_strategy_contexts()
        return binding.as_dict()

    def clear_strategy_assignment(self, environment: str) -> bool:
        """Usuwa przypisanie presetu dla wybranego kontekstu."""

        normalized_env = self._normalize_strategy_environment(environment)
        removed = self._strategy_assignments.pop(normalized_env, None) is not None
        if removed:
            self._rebuild_strategy_contexts()
        return removed

    def describe_strategy_contexts(
        self,
        *,
        include_engines: bool = True,
    ) -> Mapping[str, object]:
        """Buduje migawkę katalogu strategii dla UI (paper/live/testnet)."""

        contexts: Dict[str, Mapping[str, object]] = {}
        for key, snapshot in self._strategy_contexts.items():
            payload = dict(snapshot.as_dict())
            if not include_engines:
                payload.pop("engines", None)
            contexts[key] = payload
        return {
            "mode": self.mode.value,
            "assignments": {
                key: binding.as_dict() for key, binding in self._strategy_assignments.items()
            },
            "contexts": contexts,
        }

    def strategy_assignment(self, environment: str) -> Mapping[str, object] | None:
        """Zwraca pojedyncze przypisanie presetu (jeśli istnieje)."""

        normalized_env = self._normalize_strategy_environment(environment)
        binding = self._strategy_assignments.get(normalized_env)
        return binding.as_dict() if binding else None

    def _ensure_strategy_catalog(self) -> StrategyCatalog:
        if not isinstance(self._strategy_catalog, StrategyCatalog):
            raise RuntimeError("StrategyCatalog is not configured for ExchangeManager")
        return self._strategy_catalog

    def _resolve_preset(self, descriptor_id: str) -> StrategyPresetDescriptor:
        catalog = self._ensure_strategy_catalog()
        try:
            return catalog.preset(descriptor_id)
        except KeyError as exc:  # pragma: no cover - diagnostyka błędów konfiguracji
            raise KeyError(f"Nie znaleziono presetu strategii: {descriptor_id}") from exc

    def _normalize_strategy_environment(self, environment: str) -> str:
        normalized = (environment or "").strip().lower()
        if normalized == "live":
            normalized = "spot"
        if normalized not in _SUPPORTED_MANAGER_MODES:
            raise ValueError(
                f"Nieobsługiwany kontekst strategii: {environment} (dozwolone: {sorted(_SUPPORTED_MANAGER_MODES)})"
            )
        return normalized

    def _rebuild_strategy_contexts(self) -> None:
        catalog = self._strategy_catalog
        if not isinstance(catalog, StrategyCatalog):
            self._strategy_contexts = {}
            return

        descriptors = [descriptor.as_dict(include_strategies=True) for descriptor in catalog.list_presets()]
        engines = tuple(catalog.describe_engines())
        environments = self._strategy_context_environments()

        snapshots: Dict[str, _StrategyContextSnapshot] = {}
        for environment in environments:
            assignments = ()
            binding = self._strategy_assignments.get(environment)
            if binding:
                assignments = (binding,)
            snapshots[environment] = _StrategyContextSnapshot(
                environment=environment,
                mode=environment,
                presets=tuple(dict(entry) for entry in descriptors),
                engines=engines,
                assignments=assignments,
            )
        if "spot" in snapshots and "live" not in snapshots:
            binding = self._strategy_assignments.get("spot")
            assignments = (binding,) if binding else ()
            snapshots["live"] = _StrategyContextSnapshot(
                environment="live",
                mode="spot",
                presets=tuple(dict(entry) for entry in descriptors),
                engines=engines,
                assignments=assignments,
            )
        self._strategy_contexts = snapshots

    def _strategy_context_environments(self) -> tuple[str, ...]:
        """Zwraca listę kontekstów giełdowych obsługujących katalog strategii."""

        sequence = ["paper", "spot", "margin", "futures"]
        return tuple(dict.fromkeys(sequence))

    def _apply_manager_profile(self, config: Mapping[str, Any]) -> None:
        mode_value = str(config.get("mode") or "").strip().lower()
        testnet = bool(config.get("testnet", False))
        if mode_value == "paper":
            self.set_mode(paper=True)
        elif mode_value == "margin":
            self.set_mode(margin=True, testnet=testnet)
        elif mode_value == "futures":
            self.set_mode(futures=True, testnet=testnet)
        elif mode_value == "spot":
            self.set_mode(spot=True, testnet=testnet)
        elif mode_value:
            raise ValueError(f"Nieobsługiwany tryb ExchangeManager: {mode_value}")

        variant = config.get("paper_variant")
        if variant:
            self.set_paper_variant(str(variant))

        initial_cash = config.get("paper_initial_cash")
        cash_asset = config.get("paper_cash_asset")
        if initial_cash is not None or cash_asset:
            amount = float(initial_cash if initial_cash is not None else self.get_paper_initial_cash())
            self.set_paper_balance(amount, asset=str(cash_asset) if cash_asset else None)

        fee_rate = config.get("paper_fee_rate")
        if fee_rate is not None:
            self.set_paper_fee_rate(float(fee_rate))

        simulator_cfg = config.get("simulator")
        if isinstance(simulator_cfg, Mapping) and simulator_cfg:
            self.configure_paper_simulator(**{key: value for key, value in simulator_cfg.items() if value is not None})

        native_cfg = config.get("native_adapter")
        if isinstance(native_cfg, Mapping):
            settings = native_cfg.get("settings")
            if isinstance(settings, Mapping):
                target_mode = self.mode
                raw_mode = native_cfg.get("mode")
                if raw_mode is not None:
                    if isinstance(raw_mode, Mode):
                        target_mode = raw_mode
                    else:
                        try:
                            target_mode = Mode(str(raw_mode).lower())
                        except Exception as exc:
                            raise ValueError(f"Niepoprawny tryb natywnego adaptera: {raw_mode}") from exc
                self.configure_native_adapter(settings=settings, mode=target_mode)

        watchdog_cfg = config.get("watchdog")
        if isinstance(watchdog_cfg, Mapping) and watchdog_cfg:
            kwargs: Dict[str, Any] = {}
            retry_policy = watchdog_cfg.get("retry_policy")
            if isinstance(retry_policy, Mapping):
                kwargs["retry_policy"] = retry_policy
            circuit_breaker = watchdog_cfg.get("circuit_breaker")
            if isinstance(circuit_breaker, Mapping):
                cb_kwargs = dict(circuit_breaker)
                if "recovery_time_seconds" in cb_kwargs and "recovery_timeout" not in cb_kwargs:
                    cb_kwargs["recovery_timeout"] = cb_kwargs.pop("recovery_time_seconds")
                kwargs["circuit_breaker"] = cb_kwargs
            retry_exceptions = watchdog_cfg.get("retry_exceptions")
            if isinstance(retry_exceptions, Sequence):
                kwargs["retry_exceptions"] = tuple(
                    exc for exc in retry_exceptions if isinstance(exc, type) and issubclass(exc, Exception)
                )
            if kwargs:
                self.configure_watchdog(**kwargs)

    def create_health_monitor(self, checks: Iterable[HealthCheck]) -> HealthMonitor:
        """Buduje `HealthMonitor` współdzielący strażnika z adapterami."""

        if not isinstance(checks, Iterable):
            raise TypeError("checks musi być iterowalną sekwencją HealthCheck")

        normalized: list[HealthCheck] = []
        for check in checks:
            if not isinstance(check, HealthCheck):
                raise TypeError("checks musi zawierać instancje HealthCheck")
            normalized.append(check)

        return HealthMonitor(normalized, watchdog=self._ensure_watchdog())

    def _ensure_public(self) -> _CCXTPublicFeed:
        if self._public is None:
            if self.mode is Mode.MARGIN:
                market_type = "margin"
            elif self._futures:
                market_type = "future"
            else:
                market_type = "spot"
            self._public = _CCXTPublicFeed(
                exchange_id=self.exchange_id,
                testnet=self._testnet,
                futures=self._futures,
                market_type=market_type,
                error_handler=self._record_network_error,
            )
        return self._public

    def _resolve_environment(self) -> Environment:
        if self.mode is Mode.PAPER:
            return Environment.PAPER

        candidates = []
        if self.exchange_id.startswith("binance"):
            candidates.append(os.getenv("BINANCE_ENVIRONMENT"))
        if self.exchange_id.startswith("kraken"):
            candidates.append(os.getenv("KRAKEN_ENVIRONMENT"))
        if self.exchange_id.startswith("zonda"):
            candidates.append(os.getenv("ZONDA_ENVIRONMENT"))
        if self.exchange_id.startswith("bybit"):
            candidates.append(os.getenv("BYBIT_ENVIRONMENT"))
        if self.exchange_id.startswith("okx"):
            candidates.append(os.getenv("OKX_ENVIRONMENT"))
        if self.exchange_id.startswith("coinbase"):
            candidates.append(os.getenv("COINBASE_ENVIRONMENT"))
        candidates.append(os.getenv("EXCHANGE_ENVIRONMENT"))

        for candidate in candidates:
            if not candidate:
                continue
            try:
                environment = Environment(candidate.strip().lower())
            except (ValueError, AttributeError):
                continue
            if environment is Environment.PAPER:
                return Environment.PAPER
            return environment

        return Environment.TESTNET if self._testnet else Environment.LIVE

    def _get_adapter_settings(self) -> Dict[str, object]:
        return dict(self._native_adapter_settings.get((self.mode, self.exchange_id), {}))

    def _ensure_watchdog(self) -> Watchdog:
        if self._watchdog is None:
            self._watchdog = Watchdog()
        return self._watchdog

    def _record_network_error(self, operation: str, exc: Exception) -> None:
        self._network_error_counts[operation] += 1
        log.warning("Network error during %s: %s", operation, exc)

    def get_network_error_counts(self) -> Dict[str, int]:
        """Zwraca liczniki błędów sieciowych zarejestrowanych przez backendy CCXT."""

        return dict(self._network_error_counts)

    def _ensure_native_adapter(self):
        if self.mode not in {Mode.MARGIN, Mode.FUTURES}:
            raise RuntimeError("Natywny adapter dostępny jest wyłącznie w trybach margin/futures.")
        if not self._api_key or not self._secret:
            raise RuntimeError("Brak API Key/Secret – ustaw je przed użyciem trybu live/testnet.")

        _load_dynamic_native_adapters()

        registration = _NATIVE_ADAPTER_REGISTRY.get((self.mode, self.exchange_id))

        if registration is None:
            raise RuntimeError(
                f"Brak natywnego adaptera dla giełdy {self.exchange_id} w trybie {self.mode.value}."
            )

        if self._testnet and not registration.supports_testnet:
            raise RuntimeError(
                f"Giełda {self.exchange_id} nie wspiera trybu testnet dla {self.mode.value}."
            )

        if self._native_adapter is None:
            environment = self._resolve_environment()
            credentials = ExchangeCredentials(
                key_id=self._api_key,
                secret=self._secret,
                passphrase=self._passphrase,
                environment=environment,
                permissions=("read", "trade"),
            )
            settings = dict(registration.default_settings)
            settings.update(self._get_adapter_settings())
            kwargs: Dict[str, object] = {"environment": environment}
            if settings:
                kwargs["settings"] = settings
            kwargs["watchdog"] = self._ensure_watchdog()
            self._native_adapter = registration.factory(credentials, **kwargs)

        return self._native_adapter

    def _ensure_private(self) -> _CCXTPrivateBackend:
        if not self._api_key or not self._secret:
            raise RuntimeError("Brak API Key/Secret – ustaw je przed użyciem trybu live/testnet.")
        if self._private is None:
            if self.mode is Mode.MARGIN:
                market_type = "margin"
            elif self._futures:
                market_type = "future"
            else:
                market_type = "spot"
            self._private = _CCXTPrivateBackend(
                exchange_id=self.exchange_id,
                testnet=self._testnet,
                futures=self._futures,
                market_type=market_type,
                api_key=self._api_key,
                secret=self._secret,
                passphrase=self._passphrase,
                error_handler=self._record_network_error,
            )
            self._private.load_markets()
        return self._private

    def _ensure_paper(self) -> PaperBackend:
        public = self._ensure_public()
        if self._paper is None:
            settings = dict(self._paper_simulator_settings)
            simulator: PaperBackend
            if self._paper_variant == "margin":
                defaults = self._default_paper_simulator_settings()
                defaults.update(settings)
                simulator = PaperMarginSimulator(
                    public,
                    event_bus=self._event_bus,
                    initial_cash=self._paper_initial_cash,
                    cash_asset=self._paper_cash_asset,
                    fee_rate=self._paper_fee_rate,
                    database=self._ensure_db(),
                    leverage_limit=float(defaults.get("leverage_limit", 3.0)),
                    maintenance_margin_ratio=float(defaults.get("maintenance_margin_ratio", 0.15)),
                    funding_rate=float(defaults.get("funding_rate", 0.0)),
                    funding_interval_seconds=float(defaults.get("funding_interval_seconds", 0.0)),
                )
            elif self._paper_variant == "futures":
                defaults = self._default_paper_simulator_settings()
                defaults.update(settings)
                simulator = PaperFuturesSimulator(
                    public,
                    event_bus=self._event_bus,
                    initial_cash=self._paper_initial_cash,
                    cash_asset=self._paper_cash_asset,
                    fee_rate=self._paper_fee_rate,
                    database=self._ensure_db(),
                    leverage_limit=float(defaults.get("leverage_limit", 10.0)),
                    maintenance_margin_ratio=float(defaults.get("maintenance_margin_ratio", 0.05)),
                    funding_rate=float(defaults.get("funding_rate", 0.0001)),
                    funding_interval_seconds=float(defaults.get("funding_interval_seconds", 0.0)),
                )
            else:
                simulator = PaperBackend(
                    price_feed_backend=public,
                    event_bus=self._event_bus,
                    initial_cash=self._paper_initial_cash,
                    cash_asset=self._paper_cash_asset,
                    fee_rate=self._paper_fee_rate,
                    database=self._ensure_db(),
                )
            self._paper = simulator
            self._paper.load_markets()
        return self._paper

    def _ensure_db(self) -> Optional[DatabaseManager]:
        if self._db_failed:
            return None
        if self._db is None:
            try:
                self._db = DatabaseManager(self._db_url)
                self._db.sync.init_db()
            except Exception as exc:  # pragma: no cover
                log.warning("DatabaseManager init failed (%s): %s", self._db_url, exc)
                self._db = None
                self._db_failed = True
        return self._db

    def set_paper_balance(self, amount: float, asset: Optional[str] = None) -> None:
        self._paper_initial_cash = float(amount)
        if asset:
            self._paper_cash_asset = asset.upper()
        if self._paper is not None:
            self._paper._cash_balance = max(0.0, float(amount))  # type: ignore[attr-defined]
            if asset:
                self._paper._cash_asset = self._paper_cash_asset  # type: ignore[attr-defined]

    def fetch_account_snapshot(self) -> AccountSnapshot:
        """Pobiera snapshot konta i publikuje zdarzenie mark-to-market."""

        timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        if self.mode == Mode.PAPER:
            backend = self._ensure_paper()
        elif self.mode in {Mode.MARGIN, Mode.FUTURES}:
            backend = self._ensure_native_adapter()
        else:
            backend = self._ensure_private()

        if not hasattr(backend, "fetch_account_snapshot"):
            raise RuntimeError("Wybrany backend nie udostępnia fetch_account_snapshot")

        snapshot: AccountSnapshot = backend.fetch_account_snapshot()  # type: ignore[call-arg]

        positions: Sequence[PositionDTO] = ()
        fetch_positions = getattr(backend, "fetch_positions", None)
        if callable(fetch_positions):
            try:
                positions = tuple(fetch_positions())  # type: ignore[assignment]
            except Exception:  # pragma: no cover - diagnostyka adaptera
                log.debug("fetch_positions failed", exc_info=True)

        payload_positions: list[dict[str, object]] = []
        for position in positions or ():
            try:
                avg_price = float(position.avg_price)
            except Exception:
                avg_price = 0.0
            try:
                quantity = float(position.quantity)
            except Exception:
                quantity = 0.0
            notional = abs(quantity * avg_price)
            payload_positions.append(
                {
                    "symbol": position.symbol,
                    "side": str(getattr(position, "side", "LONG")).lower(),
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "notional": notional,
                    "unrealized_pnl": float(getattr(position, "unrealized_pnl", 0.0) or 0.0),
                }
            )

        payload = {
            "snapshot": {
                "total_equity": float(snapshot.total_equity),
                "available_margin": float(snapshot.available_margin),
                "maintenance_margin": float(snapshot.maintenance_margin),
                "balances": dict(snapshot.balances),
            },
            "positions": payload_positions,
            "mode": self.mode.value,
            "timestamp": timestamp.isoformat(),
        }

        signature = json.dumps(payload, sort_keys=True, default=str)
        if signature != self._last_mark_signature:
            self._last_mark_signature = signature
            self.publish_event(ACCOUNT_MARK_EVENT, payload)

        return snapshot

    def get_paper_cash_asset(self) -> Optional[str]:
        """Zwraca aktualny symbol waluty gotówkowej w symulatorze paper."""

        return self._paper_cash_asset

    def get_paper_initial_cash(self) -> float:
        """Zwraca aktualną wartość początkowego kapitału w symulatorze paper."""

        return float(self._paper_initial_cash)

    def get_paper_variant(self) -> str:
        """Zwraca aktywny wariant symulatora paper."""

        return self._paper_variant

    def set_paper_fee_rate(self, fee_rate: float) -> None:
        self._paper_fee_rate = max(0.0, float(fee_rate))
        if self._paper is not None:
            self._paper.set_fee_rate(self._paper_fee_rate)

    def get_paper_fee_rate(self) -> float:
        if self._paper is not None:
            return self._paper.get_fee_rate()
        return self._paper_fee_rate

    def get_paper_simulator_settings(self) -> Dict[str, float]:
        """Zwraca aktualne parametry symulatora margin/futures."""

        if self._paper_variant not in {"margin", "futures"}:
            return {}

        if isinstance(self._paper, PaperMarginSimulator):
            return dict(self._paper.describe_configuration())

        settings = self._default_paper_simulator_settings()
        for key, value in self._paper_simulator_settings.items():
            settings[key] = float(value)
        return settings

    def _default_paper_simulator_settings(self) -> Dict[str, float]:
        if self._paper_variant == "margin":
            return {
                "leverage_limit": 3.0,
                "maintenance_margin_ratio": 0.15,
                "funding_rate": 0.0,
                "funding_interval_seconds": 0.0,
            }
        if self._paper_variant == "futures":
            return {
                "leverage_limit": 10.0,
                "maintenance_margin_ratio": 0.05,
                "funding_rate": 0.0001,
                "funding_interval_seconds": 0.0,
            }
        return {}

    def load_markets(self) -> Dict[str, MarketRules]:
        public = self._ensure_public()
        rules = public.load_markets()
        log.info("Loaded %s markets (public)", len(rules))
        if self.mode == Mode.PAPER and self._paper:
            self._paper.load_markets()
        return rules

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._ensure_public().fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[List[List[float]]]:
        return self._ensure_public().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        return self._ensure_public().fetch_order_book(symbol, limit=limit)

    def fetch_batch(
        self,
        symbols: Iterable[str],
        *,
        timeframe: str = "1m",
        use_orderbook: bool = False,
        limit_ohlcv: int = 500,
    ) -> List[Tuple[str, Optional[List[List[float]]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]]:
        results: List[
            Tuple[
                str,
                Optional[List[List[float]]],
                Optional[Dict[str, Any]],
                Optional[Dict[str, Any]],
                Optional[str],
            ]
        ] = []
        for symbol in symbols:
            ohlcv: Optional[List[List[float]]] = None
            ticker: Optional[Dict[str, Any]] = None
            orderbook: Optional[Dict[str, Any]] = None
            errors: List[str] = []

            try:
                ohlcv = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit_ohlcv)
            except Exception as exc:  # pragma: no cover - defensywnie
                errors.append(f"ohlcv: {exc}")

            try:
                ticker = self.fetch_ticker(symbol)
            except Exception as exc:  # pragma: no cover - defensywnie
                errors.append(f"ticker: {exc}")

            if use_orderbook:
                try:
                    orderbook = self.fetch_order_book(symbol, limit=50)
                except Exception as exc:  # pragma: no cover - defensywnie
                    errors.append(f"orderbook: {exc}")

            error_msg = "; ".join(errors) if errors else None
            results.append((symbol, ohlcv, ticker, orderbook, error_msg))

        return results

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        public = self._ensure_public()
        if not public.get_market_rules(symbol):
            public.load_markets()
        return public.get_market_rules(symbol)

    def quantize_amount(self, symbol: str, amount: float) -> float:
        rules = self.get_market_rules(symbol)
        return rules.quantize_amount(amount) if rules else float(f"{amount:.8f}")

    def quantize_price(self, symbol: str, price: float) -> float:
        rules = self.get_market_rules(symbol)
        return rules.quantize_price(price) if rules else float(f"{price:.8f}")

    def min_notional(self, symbol: str) -> float:
        rules = self.get_market_rules(symbol)
        return float(rules.min_notional) if rules else 0.0

    def simulate_vwap_price(
        self,
        symbol: str,
        side: str,
        amount: Optional[float],
        fallback_bps: float = 5.0,
        limit: int = 50,
    ) -> Tuple[Optional[float], float]:
        try:
            ticker = self.fetch_ticker(symbol) or {}
            last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
            mid = float(last) if last else None
            if amount is None or amount <= 0:
                return (mid, float(fallback_bps))

            order_book = self.fetch_order_book(symbol, limit=limit) or {}
            side_lower = side.lower().strip()
            levels = order_book.get("asks") if side_lower == "buy" else order_book.get("bids")
            if not levels:
                return (mid, float(fallback_bps))

            remaining = float(amount)
            taken = 0.0
            cost = 0.0
            for price, qty in levels:
                take_qty = min(remaining - taken, float(qty))
                if take_qty <= 0:
                    break
                cost += take_qty * float(price)
                taken += take_qty
                if taken >= remaining - 1e-12:
                    break

            if taken <= 0:
                return (mid, float(fallback_bps))

            vwap = cost / taken
            if mid:
                slip_bps = abs(vwap - mid) / mid * 10_000.0
            else:
                slip_bps = float(fallback_bps)
            return (float(vwap), float(slip_bps))
        except Exception as exc:
            log.warning("simulate_vwap_price failed for %s: %s", symbol, exc)
            try:
                fallback = self.fetch_ticker(symbol) or {}
                last = fallback.get("last") or fallback.get("close")
                return (float(last) if last else None, float(fallback_bps))
            except Exception:
                return (None, float(fallback_bps))

    def fetch_balance(self) -> Dict[str, Any]:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().fetch_balance()

        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            adapter = self._ensure_native_adapter()
            snapshot = adapter.fetch_account_snapshot()
            return {
                "balances": dict(snapshot.balances),
                "total_equity": snapshot.total_equity,
                "available_margin": snapshot.available_margin,
                "maintenance_margin": snapshot.maintenance_margin,
            }

        backend = self._ensure_private()
        raw = backend.fetch_balance()
        return self._normalize_balance(raw)

    @staticmethod
    def _normalize_balance(balance: Any) -> Dict[str, Any]:
        if not isinstance(balance, dict):
            return {}
        result: Dict[str, Any] = dict(balance)
        for key in ("free", "total", "used"):
            section = balance.get(key)
            if not isinstance(section, dict):
                continue
            normalized: Dict[str, float] = {}
            for asset, amount in section.items():
                try:
                    normalized[asset] = float(amount)
                except Exception:
                    continue
                result.setdefault(asset, normalized[asset])
            result[key] = normalized
        return result

    def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        type_enum = OrderType.MARKET if type.upper() == "MARKET" else OrderType.LIMIT

        if self.mode == Mode.PAPER:
            return self._ensure_paper().create_order(symbol, side_enum, type_enum, quantity, price, client_order_id)

        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            rules = self.get_market_rules(symbol)
            if not rules:
                self.load_markets()
                rules = self.get_market_rules(symbol)
            if not rules:
                raise RuntimeError(f"Brak reguł rynku dla {symbol}. Najpierw załaduj rynek.")

            qty = rules.quantize_amount(float(quantity))
            if qty <= 0:
                raise ValueError("Ilość po kwantyzacji = 0.")

            price_value: Optional[float] = None
            if type_enum is OrderType.LIMIT:
                if price is None:
                    raise ValueError("Cena wymagana dla LIMIT.")
                price_value = rules.quantize_price(float(price))

            if type_enum is OrderType.MARKET:
                ticker = self.fetch_ticker(symbol) or {}
                last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
                if not last:
                    raise RuntimeError(f"Brak ceny MARKET dla {symbol}.")
                notional = qty * float(last)
            else:
                notional = qty * float(price_value or 0.0)

            min_notional = rules.min_notional or 0.0
            if min_notional and notional < min_notional:
                raise ValueError(
                    f"Notional {notional:.8f} < minNotional {min_notional:.8f} dla {symbol}"
                )

            adapter = self._ensure_native_adapter()
            request = OrderRequest(
                symbol=symbol,
                side=side_enum.value,
                quantity=qty,
                order_type=type_enum.value,
                price=price_value,
                client_order_id=client_order_id,
            )
            result = adapter.place_order(request)
            raw_payload = result.raw_response if isinstance(result.raw_response, Mapping) else {}
            resolved_client_id = client_order_id
            if not resolved_client_id and isinstance(raw_payload, Mapping):
                candidate = (
                    raw_payload.get("clientOrderId")
                    or raw_payload.get("client_order_id")
                    or raw_payload.get("userref")
                )
                if isinstance(candidate, str) and candidate:
                    resolved_client_id = candidate
            order_identifier = result.order_id
            try:
                parsed_id = int(order_identifier) if order_identifier is not None else None
            except (TypeError, ValueError):
                parsed_id = None

            return OrderDTO(
                id=parsed_id,
                client_order_id=resolved_client_id,
                symbol=symbol,
                side=side_enum,
                type=type_enum,
                quantity=qty,
                price=price_value,
                status=_map_order_status(result.status),
                mode=self.mode,
                extra={
                    "order_id": order_identifier,
                    "filled_quantity": result.filled_quantity,
                    "avg_price": result.avg_price,
                    "raw_response": raw_payload,
                },
            )

        backend = self._ensure_private()
        return backend.create_order(symbol, side_enum, type_enum, quantity, price, client_order_id)

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        if self.mode == Mode.PAPER:
            return False
        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            try:
                adapter = self._ensure_native_adapter()
                adapter.cancel_order(str(order_id), symbol=symbol)
                return True
            except Exception as exc:
                log.error("cancel_order failed (native): %s", exc)
                return False
        return self._ensure_private().cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        if self.mode == Mode.PAPER:
            return []
        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            try:
                adapter = self._ensure_native_adapter()
                native_orders = adapter.fetch_open_orders()
            except Exception as exc:
                log.error("fetch_open_orders failed (native): %s", exc)
                return []

            result: List[OrderDTO] = []
            for entry in native_orders or []:
                raw_symbol = getattr(entry, "symbol", symbol or "")
                order_symbol = raw_symbol if isinstance(raw_symbol, str) else symbol or ""
                price_value = getattr(entry, "price", None)
                if price_value in (None, ""):
                    resolved_price = None
                else:
                    try:
                        resolved_price = float(price_value)
                    except Exception:
                        resolved_price = None
                quantity_value = getattr(entry, "orig_quantity", getattr(entry, "quantity", 0.0))
                try:
                    resolved_quantity = float(quantity_value)
                except Exception:
                    resolved_quantity = 0.0
                order_identifier = getattr(entry, "order_id", None)
                try:
                    parsed_id = int(order_identifier) if order_identifier is not None else None
                except (TypeError, ValueError):
                    parsed_id = None
                result.append(
                    OrderDTO(
                        id=parsed_id,
                        client_order_id=getattr(entry, "client_order_id", None),
                        symbol=order_symbol,
                        side=_map_order_side(getattr(entry, "side", "BUY")),
                        type=_map_order_type(getattr(entry, "order_type", "LIMIT")),
                        quantity=resolved_quantity,
                        price=resolved_price,
                        status=_map_order_status(getattr(entry, "status", "OPEN")),
                        mode=self.mode,
                        extra={"order_id": order_identifier},
                    )
                )
            return result
        return self._ensure_private().fetch_open_orders(symbol)

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().fetch_positions(symbol)

        if self.mode == Mode.SPOT:
            try:
                db = self._ensure_db()
                if db:
                    positions = db.sync.get_open_positions(mode=Mode.SPOT.value)
                else:
                    positions = []
            except Exception as exc:
                log.debug("DB fallback failed: %s", exc)
                positions = []

            if positions:
                out: List[PositionDTO] = []
                for entry in positions:
                    try:
                        qty = float(entry.get("quantity") or 0.0)
                    except Exception:
                        continue
                    if qty <= 0:
                        continue
                    side_val = entry.get("side") or "LONG"
                    avg_price = float(entry.get("avg_price") or 0.0)
                    unreal = float(entry.get("unrealized_pnl") or 0.0)
                    sym = entry.get("symbol") or ""
                    out.append(
                        PositionDTO(
                            symbol=sym,
                            side=side_val,
                            quantity=qty,
                            avg_price=avg_price,
                            unrealized_pnl=unreal,
                            mode=Mode.SPOT,
                        )
                    )
                if out:
                    return out

            try:
                backend = self._ensure_private()
                balance = backend.fetch_balance()
            except Exception as exc:
                log.warning("Spot balance fallback failed: %s", exc)
                return []
            normalized = self._normalize_balance(balance)
            return self._positions_from_balance(normalized, symbol)

        if self.mode == Mode.FUTURES:
            try:
                adapter = self._ensure_native_adapter()
                native_positions = adapter.fetch_positions()
            except RuntimeError as exc:
                log.warning("Fallback to CCXT futures backend: %s", exc)
            except Exception as exc:
                log.error("fetch_positions failed (native): %s", exc)
                return []
            else:
                result: List[PositionDTO] = []
                for entry in native_positions or []:
                    try:
                        quantity = float(getattr(entry, "quantity", 0.0) or 0.0)
                    except Exception:
                        quantity = 0.0
                    if abs(quantity) < 1e-12:
                        continue
                    avg_price = getattr(entry, "entry_price", getattr(entry, "avg_price", 0.0))
                    try:
                        resolved_avg = float(avg_price)
                    except Exception:
                        resolved_avg = 0.0
                    try:
                        pnl = float(getattr(entry, "unrealized_pnl", 0.0) or 0.0)
                    except Exception:
                        pnl = 0.0
                    result.append(
                        PositionDTO(
                            symbol=str(getattr(entry, "symbol", "")),
                            side=str(getattr(entry, "side", "LONG")),
                            quantity=abs(quantity),
                            avg_price=resolved_avg,
                            unrealized_pnl=pnl,
                            mode=Mode.FUTURES,
                        )
                    )
                return result

        return self._ensure_private().fetch_positions(symbol)

    def _positions_from_balance(
        self,
        balance: Dict[str, Any],
        symbol: Optional[str],
    ) -> List[PositionDTO]:
        if isinstance(balance.get("total"), dict):
            totals = dict(balance.get("total") or {})
        elif isinstance(balance.get("free"), dict):
            totals = dict(balance.get("free") or {})
        else:
            totals = {
                key: value
                for key, value in balance.items()
                if key not in {"free", "used", "total", "info"}
            }

        preferred_quotes = {
            self._paper_cash_asset.upper(),
            "USDT",
            "USD",
            "USDC",
            "BUSD",
            "EUR",
        }
        fallback_quote = self._paper_cash_asset.upper()
        markets: Dict[str, Any] = {}
        try:
            public = self._ensure_public()
            markets = public._markets or public.load_markets()
        except Exception as exc:  # pragma: no cover - informacyjne
            log.debug("Market load failed for balance conversion: %s", exc)

        symbol_filter = symbol
        base_filter: Optional[str] = None
        if symbol_filter:
            try:
                base_filter = symbol_filter.split("/")[0].upper()
            except Exception:
                base_filter = symbol_filter.upper()

        out: List[PositionDTO] = []
        for asset, amount in totals.items():
            try:
                qty = float(amount)
            except Exception:
                continue
            if qty <= 0:
                continue
            base = asset.upper()
            if base_filter and base != base_filter:
                continue
            if base in preferred_quotes:
                continue

            resolved_symbol = None
            if symbol_filter and base_filter == base:
                resolved_symbol = symbol_filter
            else:
                resolved_symbol = self._resolve_symbol_from_markets(
                    base, markets, preferred_quotes, fallback_quote
                )
            if resolved_symbol is None:
                resolved_symbol = base

            price = 0.0
            if resolved_symbol and "/" in resolved_symbol:
                try:
                    ticker = self.fetch_ticker(resolved_symbol) or {}
                    price = float(
                        ticker.get("last")
                        or ticker.get("close")
                        or ticker.get("bid")
                        or ticker.get("ask")
                        or 0.0
                    )
                except Exception:
                    price = 0.0

            out.append(
                PositionDTO(
                    symbol=resolved_symbol,
                    side="LONG",
                    quantity=qty,
                    avg_price=price,
                    unrealized_pnl=0.0,
                    mode=Mode.SPOT,
                )
            )
        return out

    def process_paper_tick(
        self,
        symbol: str,
        price: float,
        *,
        timestamp: Optional[dt.datetime] = None,
    ) -> None:
        if self.mode != Mode.PAPER:
            raise RuntimeError("process_paper_tick dostępne tylko w trybie paper")
        backend = self._ensure_paper()
        processor = getattr(backend, "process_tick", None)
        if not callable(processor):
            raise RuntimeError("Paper backend nie obsługuje process_tick")
        processor(symbol, price, timestamp=timestamp)

    def _resolve_symbol_from_markets(
        self,
        base: str,
        markets: Dict[str, Any],
        preferred_quotes: set,
        fallback_quote: str,
    ) -> Optional[str]:
        candidates: List[Tuple[str, str]] = []
        for symbol in markets.keys():
            if not isinstance(symbol, str) or "/" not in symbol:
                continue
            base_part, quote_part = symbol.split("/", 1)
            if base_part.upper() != base.upper():
                continue
            candidates.append((quote_part.upper(), symbol))

        for quote, candidate in candidates:
            if quote in preferred_quotes:
                return candidate
        if candidates:
            return candidates[0][1]
        if fallback_quote:
            return f"{base}/{fallback_quote}"
        return None

    def on(self, event_type: str, callback) -> None:
        self._event_bus.subscribe(event_type, callback)


__all__ = [
    "ExchangeManager",
    "Mode",
    "OrderDTO",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionDTO",
    "register_native_adapter",
    "reload_native_adapters",
    "iter_registered_native_adapters",
    "NativeAdapterInfo",
]

