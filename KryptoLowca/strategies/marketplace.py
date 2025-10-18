# -*- coding: utf-8 -*-
"""Prosty menedżer konfiguracji kompatybilny ze starszym API testów."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Mapping

import pandas as pd
import yaml
from cryptography.fernet import Fernet, InvalidToken

if TYPE_CHECKING:  # pragma: no cover - tylko dla typowania
    from KryptoLowca.backtest.simulation import BacktestReport, MatchingConfig
    from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest

__all__ = [
    "ConfigManager",
    "ConfigError",
    "ValidationError",
    "AIConfig",
    "DBConfig",
    "TradeConfig",
    "ExchangeConfig",
    "StrategyConfig",
    "StrategyPreset",
    "load_marketplace_presets",
    "load_preset",
]


class ConfigError(RuntimeError):
    """Ogólny błąd operacji na konfiguracji."""


class ValidationError(ValueError):
    """Nieprawidłowe wartości w konfiguracji."""


_DEFAULT_MARKETPLACE_DIR = Path(__file__).with_name("marketplace")


def _ensure_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _ensure_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as exc:  # pragma: no cover - defensywne
            raise ConfigError("Wartość musi być liczbą zmiennoprzecinkową") from exc
    raise ConfigError("Wartość musi być liczbą zmiennoprzecinkową lub None")


def _ensure_str_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                result.append(stripped)
        elif isinstance(item, (int, float)):
            result.append(str(item))
    return result


def _ensure_mapping(value: object) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _parse_last_updated(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensywne
            raise ConfigError("last_updated musi być w formacie ISO 8601") from exc
    else:
        raise ConfigError("last_updated musi być napisem lub datetime")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


@dataclass(slots=True)
class StrategyPreset:
    preset_id: str
    name: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    risk_level: str | None = None
    recommended_min_balance: float | None = None
    timeframe: str | None = None
    exchanges: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str | None = None
    last_updated: datetime | None = None
    compatibility: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StrategyPreset":
        preset_id = _ensure_optional_str(payload.get("id")) or _ensure_optional_str(
            payload.get("preset_id")
        )
        if not preset_id:
            raise ConfigError("Preset JSON musi zawierać pole 'id'")

        name = _ensure_optional_str(payload.get("name")) or preset_id
        description = _ensure_optional_str(payload.get("description")) or ""
        config_payload = payload.get("config")
        if config_payload is None:
            config_data: Dict[str, Any] = {}
        elif isinstance(config_payload, Mapping):
            config_data = dict(config_payload)
        else:
            raise ConfigError("Pole 'config' musi być słownikiem")

        risk_level = _ensure_optional_str(payload.get("risk_level"))
        recommended = _ensure_float(payload.get("recommended_min_balance"))
        timeframe = _ensure_optional_str(payload.get("timeframe"))
        exchanges = _ensure_str_list(payload.get("exchanges"))
        tags = _ensure_str_list(payload.get("tags"))
        version = _ensure_optional_str(payload.get("version"))
        last_updated = _parse_last_updated(payload.get("last_updated"))
        compatibility = _ensure_mapping(payload.get("compatibility"))
        compliance = _ensure_mapping(payload.get("compliance"))

        return cls(
            preset_id=preset_id,
            name=name,
            description=description,
            config=config_data,
            risk_level=risk_level,
            recommended_min_balance=recommended,
            timeframe=timeframe,
            exchanges=exchanges,
            tags=tags,
            version=version,
            last_updated=last_updated,
            compatibility=compatibility,
            compliance=compliance,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.last_updated is not None:
            data["last_updated"] = self.last_updated.isoformat()
        return data


def _resolve_marketplace_dir(base_path: str | Path | None) -> Path:
    if base_path is None:
        return _DEFAULT_MARKETPLACE_DIR
    return Path(base_path)


def _load_preset_from_path(file_path: Path) -> StrategyPreset:
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Nie można odczytać pliku presetu: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Nieprawidłowy JSON w pliku {file_path}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("Plik presetu musi zawierać obiekt JSON")
    return StrategyPreset.from_payload(raw)


def load_marketplace_presets(*, base_path: str | Path | None = None) -> List[StrategyPreset]:
    directory = _resolve_marketplace_dir(base_path)
    if not directory.exists():
        return []
    presets: List[StrategyPreset] = []
    for file_path in sorted(directory.glob("*.json")):
        if not file_path.is_file():
            continue
        try:
            presets.append(_load_preset_from_path(file_path))
        except ConfigError:
            continue
    return presets


def load_preset(preset_id: str, *, base_path: str | Path | None = None) -> StrategyPreset:
    directory = _resolve_marketplace_dir(base_path)
    file_path = directory / f"{preset_id}.json"
    if not file_path.exists():
        raise ConfigError(f"Brak presetu o identyfikatorze '{preset_id}'")
    return _load_preset_from_path(file_path)


@dataclass(slots=True)
class AIConfig:
    threshold_bps: float = 5.0
    model_types: List[str] = field(default_factory=lambda: ["rf"])
    seq_len: int = 64
    epochs: int = 10
    batch_size: int = 32
    model_dir: str = "models"

    def validate(self) -> "AIConfig":
        if self.threshold_bps < 0:
            raise ValidationError("threshold_bps musi być >= 0")
        if not self.model_types:
            raise ValidationError("model_types musi zawierać co najmniej jedną nazwę")
        if self.seq_len <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValidationError("seq_len, epochs i batch_size muszą być dodatnie")
        return self


@dataclass(slots=True)
class DBConfig:
    db_url: str = "sqlite+aiosqlite:///trading.db"
    timeout_s: float = 30.0
    pool_size: int = 5
    max_overflow: int = 5

    def validate(self) -> "DBConfig":
        if not self.db_url:
            raise ValidationError("db_url nie może być puste")
        if self.timeout_s <= 0:
            raise ValidationError("timeout_s musi być dodatnie")
        if self.pool_size < 0 or self.max_overflow < 0:
            raise ValidationError("Rozmiary puli muszą być nieujemne")
        return self


@dataclass(slots=True)
class TradeConfig:
    risk_per_trade: float = 0.01
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_open_positions: int = 5

    def validate(self) -> "TradeConfig":
        if not (0.0 <= self.risk_per_trade <= 1.0):
            raise ValidationError("risk_per_trade musi być w zakresie 0-1")
        if self.max_leverage <= 0:
            raise ValidationError("max_leverage musi być dodatnie")
        if self.stop_loss_pct < 0 or self.take_profit_pct < 0:
            raise ValidationError("Poziomy SL/TP muszą być nieujemne")
        if self.max_open_positions <= 0:
            raise ValidationError("max_open_positions musi być dodatnie")
        return self


@dataclass(slots=True)
class ExchangeConfig:
    exchange_name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    # rozszerzenia: limity/alerty/retry
    rate_limit_per_minute: int = 1200
    rate_limit_window_seconds: float = 60.0
    rate_limit_alert_threshold: float = 0.85
    error_alert_threshold: int = 3
    rate_limit_buckets: List[Dict[str, Any]] = field(default_factory=list)
    retry_attempts: int = 1
    retry_delay: float = 0.05
    require_demo_mode: bool = True
    # rozszerzenia telemetryczne (opcjonalne)
    telemetry_log_interval_s: float = 30.0
    telemetry_schema_version: int = 1
    telemetry_storage_path: Optional[str] = None
    telemetry_grpc_target: Optional[str] = None

    def validate(self) -> "ExchangeConfig":
        if not self.exchange_name:
            raise ValidationError("exchange_name nie może być puste")
        if self.rate_limit_per_minute < 0:
            raise ValidationError("rate_limit_per_minute musi być nieujemne")
        if self.rate_limit_window_seconds <= 0:
            raise ValidationError("rate_limit_window_seconds musi być dodatnie")
        if not (0.0 < self.rate_limit_alert_threshold <= 1.0):
            raise ValidationError("rate_limit_alert_threshold musi być w zakresie (0, 1]")
        if self.error_alert_threshold <= 0:
            raise ValidationError("error_alert_threshold musi być dodatnie")
        if self.retry_attempts < 0:
            raise ValidationError("retry_attempts musi być >= 0")
        if self.retry_delay < 0:
            raise ValidationError("retry_delay musi być >= 0")
        if self.telemetry_log_interval_s <= 0:
            raise ValidationError("telemetry_log_interval_s musi być dodatnie")
        if self.telemetry_schema_version <= 0:
            raise ValidationError("telemetry_schema_version musi być dodatnie")
        if self.telemetry_storage_path is not None and not isinstance(self.telemetry_storage_path, str):
            raise ValidationError("telemetry_storage_path musi być ścieżką lub None")
        if self.telemetry_grpc_target is not None and not isinstance(self.telemetry_grpc_target, str):
            raise ValidationError("telemetry_grpc_target musi być tekstem lub None")
        if not isinstance(self.rate_limit_buckets, list):
            raise ValidationError("rate_limit_buckets musi być listą")

        # normalizacja kubełków
        cleaned_buckets: List[Dict[str, Any]] = []
        for bucket in self.rate_limit_buckets:
            if not isinstance(bucket, dict):
                continue
            capacity = int(bucket.get("capacity", 0))
            window = float(bucket.get("window_seconds", 0.0))
            if capacity <= 0 or window <= 0:
                continue
            name = str(bucket.get("name") or f"bucket_{len(cleaned_buckets) + 1}")
            cleaned_buckets.append({"name": name, "capacity": capacity, "window_seconds": window})
        self.rate_limit_buckets = cleaned_buckets

        # rzutowania
        self.retry_attempts = int(self.retry_attempts)

        retry_delay_raw = self.retry_delay
        if retry_delay_raw is None:
            retry_delay_value = 0.0
        elif isinstance(retry_delay_raw, (int, float)):
            retry_delay_value = float(retry_delay_raw)
        elif isinstance(retry_delay_raw, str):
            try:
                retry_delay_value = float(retry_delay_raw.strip())
            except ValueError as exc:
                raise ValidationError("retry_delay musi być liczbą") from exc
        else:
            raise ValidationError("retry_delay musi być liczbą lub None")
        self.retry_delay = retry_delay_value
        self.require_demo_mode = bool(self.require_demo_mode)
        self.telemetry_log_interval_s = float(self.telemetry_log_interval_s)
        self.telemetry_schema_version = int(self.telemetry_schema_version)
        if self.telemetry_storage_path is not None:
            self.telemetry_storage_path = str(self.telemetry_storage_path)
        if self.telemetry_grpc_target is not None:
            self.telemetry_grpc_target = str(self.telemetry_grpc_target)
        return self


@dataclass(slots=True)
class StrategyConfig:
    """Parametry strategii oraz sztywne limity ryzyka dla auto-tradingu."""

    preset: str = "SAFE"
    mode: str = "demo"
    max_leverage: float = 1.0
    max_position_notional_pct: float = 0.02  # 2% kapitału na pojedynczą pozycję
    trade_risk_pct: float = 0.01
    default_sl: float = 0.02
    default_tp: float = 0.04
    violation_cooldown_s: int = 300
    reduce_only_after_violation: bool = True
    compliance_confirmed: bool = False
    api_keys_configured: bool = False
    acknowledged_risk_disclaimer: bool = False

    def validate(self) -> "StrategyConfig":
        mode = (self.mode or "demo").strip().lower()
        if mode not in {"demo", "live"}:
            raise ValidationError("mode musi mieć wartość 'demo' lub 'live'")
        self.mode = mode

        if self.max_leverage <= 0:
            raise ValidationError("max_leverage musi być dodatnie")
        if not (0.0 <= self.max_position_notional_pct <= 1.0):
            raise ValidationError("max_position_notional_pct musi być w zakresie [0, 1]")
        if not (0.0 <= self.trade_risk_pct <= 1.0):
            raise ValidationError("trade_risk_pct musi być w zakresie [0, 1]")
        if self.default_sl < 0 or self.default_tp < 0:
            raise ValidationError("default_sl i default_tp muszą być nieujemne")
        if self.violation_cooldown_s <= 0:
            raise ValidationError("violation_cooldown_s musi być dodatnie")
        self.reduce_only_after_violation = bool(self.reduce_only_after_violation)
        self.preset = (self.preset or "").strip().upper() or "CUSTOM"

        # Normalizacja flag zgodności (bool)
        for field_name in (
            "compliance_confirmed",
            "api_keys_configured",
            "acknowledged_risk_disclaimer",
        ):
            value = getattr(self, field_name, None)
            if value is None:
                normalized = False
            elif isinstance(value, bool):
                normalized = value
            elif isinstance(value, (int, float)) and value in (0, 1):
                normalized = bool(value)
            else:
                raise ValidationError(
                    f"{field_name} musi być wartością boolowską (domyślnie False)"
                )
            setattr(self, field_name, normalized)

        return self

    @classmethod
    def presets(cls) -> Dict[str, "StrategyConfig"]:
        return {
            "SAFE": cls(
                preset="SAFE",
                mode="demo",
                max_leverage=1.0,
                max_position_notional_pct=0.02,
                trade_risk_pct=0.01,
                default_sl=0.02,
                default_tp=0.04,
                violation_cooldown_s=300,
                reduce_only_after_violation=True,
            ),
            "BALANCED": cls(
                preset="BALANCED",
                mode="demo",
                max_leverage=2.0,
                max_position_notional_pct=0.05,
                trade_risk_pct=0.02,
                default_sl=0.03,
                default_tp=0.06,
                violation_cooldown_s=240,
                reduce_only_after_violation=True,
            ),
        }

    @classmethod
    def from_preset(cls, name: str) -> "StrategyConfig":
        presets = cls.presets()
        preset = presets.get((name or "").strip().upper())
        return (preset or cls(preset="CUSTOM")).validate()


_STRATEGY_FIELD_NAMES = {f.name for f in fields(StrategyConfig)}
_COMPLIANCE_FLAGS = (
    "compliance_confirmed",
    "api_keys_configured",
    "acknowledged_risk_disclaimer",
)


class _InMemoryDB:
    def __init__(self) -> None:
        self._users: Dict[str, int] = {}
        self._next_id = 1
        self._user_configs: Dict[int, Dict[str, Dict[str, Any]]] = {}

    async def ensure_user(self, email: str) -> int:
        email = (email or "").strip().lower()
        if not email:
            raise ValidationError("email nie może być pusty")
        if email not in self._users:
            self._users[email] = self._next_id
            self._next_id += 1
        return self._users[email]

    async def save_user_config(self, user_id: int, name: str, config: Dict[str, Any]) -> None:
        if user_id <= 0:
            raise ValidationError("user_id musi być dodatnie")
        name = (name or "").strip()
        if not name:
            raise ValidationError("Nazwa presetu nie może być pusta")
        self._user_configs.setdefault(user_id, {})[name] = config

    async def load_user_config(self, user_id: int, name: str) -> Dict[str, Any]:
        try:
            return self._user_configs[user_id][name]
        except KeyError as exc:
            raise ConfigError("Nie znaleziono konfiguracji użytkownika") from exc


class ConfigManager:
    """Prosta implementacja asynchroniczna zgodna z testami."""

    def __init__(self, config_path: Path, encryption_key: Optional[bytes] = None) -> None:
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(encryption_key) if encryption_key else None
        self.db_manager = _InMemoryDB()
        self._current_config: Dict[str, Any] = self._default_config()
        self._marketplace_dir: Optional[Path] = None

    @classmethod
    async def create(
        cls,
        *,
        config_path: str,
        encryption_key: Optional[bytes] = None,
    ) -> "ConfigManager":
        return cls(Path(config_path), encryption_key=encryption_key)

    # -------------------------- obsługa konfiguracji --------------------------
    def _default_config(self) -> Dict[str, Any]:
        strategy_cfg = StrategyConfig.presets()["SAFE"].validate()
        strategy_section = asdict(strategy_cfg)
        for flag in _COMPLIANCE_FLAGS:
            strategy_section.setdefault(flag, False)
        return {
            "ai": asdict(AIConfig().validate()),
            "db": asdict(DBConfig().validate()),
            "trade": asdict(TradeConfig().validate()),
            "exchange": asdict(ExchangeConfig().validate()),
            "strategy": strategy_section,
        }

    def _encrypt_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._fernet:
            return data
        encrypted = dict(data)
        for key in ("api_key", "api_secret"):
            value = encrypted.get(key)
            if isinstance(value, str) and value:
                encrypted[key] = self._fernet.encrypt(value.encode()).decode()
        return encrypted

    def _decrypt_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._fernet:
            return data
        decrypted = dict(data)
        for key in ("api_key", "api_secret"):
            value = decrypted.get(key)
            if isinstance(value, str) and value:
                try:
                    decrypted[key] = self._fernet.decrypt(value.encode()).decode()
                except InvalidToken:
                    # jeśli nie jest zaszyfrowane lub klucz się zmienił – pozostaw oryginał
                    pass
        return decrypted

    def _validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ai = AIConfig(**payload.get("ai", {})).validate()
        db = DBConfig(**payload.get("db", {})).validate()
        trade = TradeConfig(**payload.get("trade", {})).validate()
        exchange = ExchangeConfig(**payload.get("exchange", {})).validate()

        strategy_payload = payload.get("strategy", {})
        if isinstance(strategy_payload, str):
            strategy = StrategyConfig.from_preset(strategy_payload)
        elif isinstance(strategy_payload, dict):
            default_strategy = StrategyConfig().validate()
            merged_strategy = asdict(default_strategy)
            filtered_payload = {
                key: strategy_payload[key]
                for key in strategy_payload
                if key in _STRATEGY_FIELD_NAMES
            }
            merged_strategy.update(filtered_payload)
            strategy = StrategyConfig(**merged_strategy).validate()
        else:
            strategy = StrategyConfig().validate()

        strategy_section = asdict(strategy)
        for flag in _COMPLIANCE_FLAGS:
            strategy_section[flag] = bool(strategy_section.get(flag, False))

        return {
            "ai": asdict(ai),
            "db": asdict(db),
            "trade": asdict(trade),
            "exchange": asdict(exchange),
            "strategy": strategy_section,
        }

    def set_marketplace_directory(self, directory: str | Path) -> None:
        self._marketplace_dir = Path(directory)

    def list_marketplace_catalog(self) -> List[StrategyPreset]:
        return load_marketplace_presets(base_path=self._marketplace_dir)

    def get_marketplace_preset(self, preset_id: str) -> StrategyPreset:
        return load_preset(preset_id, base_path=self._marketplace_dir)

    def apply_marketplace_preset(self, preset_id: str) -> Dict[str, Any]:
        preset = self.get_marketplace_preset(preset_id)
        merged = dict(self._current_config)
        for section, payload in preset.config.items():
            if not isinstance(payload, dict):
                continue
            existing = dict(merged.get(section, {}))
            existing.update(payload)
            merged[section] = existing
        self._current_config = self._validate_payload(merged)
        return self._current_config

    async def save_config(self, config: Dict[str, Any]) -> None:
        sanitized = self._validate_payload(config)
        to_disk = dict(sanitized)
        to_disk["exchange"] = self._encrypt_section(to_disk["exchange"])
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(to_disk, fh, sort_keys=True)
        self._current_config = sanitized

    async def load_config(
        self,
        *,
        preset_name: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if preset_name and user_id:
            data = await self.db_manager.load_user_config(user_id, preset_name)
            self._current_config = self._validate_payload(data)
            return self._current_config

        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            data["exchange"] = self._decrypt_section(data.get("exchange", {}))
            self._current_config = self._validate_payload(data)
        else:
            self._current_config = self._default_config()
        return self._current_config

    async def save_user_config(self, user_id: int, name: str, config: Dict[str, Any]) -> None:
        sanitized = self._validate_payload(config)
        await self.db_manager.save_user_config(user_id, name, sanitized)

    def load_ai_config(self) -> AIConfig:
        return AIConfig(**self._current_config.get("ai", {})).validate()

    def load_db_config(self) -> DBConfig:
        return DBConfig(**self._current_config.get("db", {})).validate()

    def load_trade_config(self) -> TradeConfig:
        return TradeConfig(**self._current_config.get("trade", {})).validate()

    def load_exchange_config(self) -> ExchangeConfig:
        return ExchangeConfig(**self._current_config.get("exchange", {})).validate()

    def load_strategy_config(self) -> StrategyConfig:
        strategy = self._current_config.get("strategy", {})
        if isinstance(strategy, dict):
            return StrategyConfig(**strategy).validate()
        return StrategyConfig.from_preset(str(strategy))

    def list_strategy_presets(self) -> Dict[str, Dict[str, Any]]:
        return {name: asdict(cfg.validate()) for name, cfg in StrategyConfig.presets().items()}

    # -------------------------- Backtest helpers --------------------------
    def run_backtest_on_dataframe(
        self,
        data: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        initial_balance: float = 10_000.0,
        matching: Optional["MatchingConfig"] = None,
        allow_short: bool = False,
        context_extra: Optional[Dict[str, Any]] = None,
    ) -> "BacktestReport":
        """
        Uruchamia backtest na dostarczonym DataFrame korzystając z BacktestEngine.
        Importy runtime-owe, by uniknąć twardej zależności i cykli importu.
        """
        if data is None or data.empty:
            raise ValidationError("Backtest wymaga niepustych danych historycznych")

        # runtime import
        from KryptoLowca.backtest.simulation import BacktestEngine, MatchingConfig  # noqa

        strategy_cfg = self.load_strategy_config()
        extra = dict(context_extra or {})
        # przekaż ograniczenia strategii jako kontekst
        extra.setdefault("trade_risk_pct", strategy_cfg.trade_risk_pct)
        extra.setdefault("max_position_notional_pct", strategy_cfg.max_position_notional_pct)
        extra.setdefault("max_leverage", strategy_cfg.max_leverage)

        match_cfg = matching or MatchingConfig()
        engine = BacktestEngine(
            strategy_name=strategy_name,
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=float(initial_balance),
            matching=match_cfg,
            allow_short=bool(allow_short),
            context_extra=extra,
        )
        report = engine.run()
        return report

    async def preflight_backtest(
        self,
        provider: "MarketDataProvider",
        request: "MarketDataRequest",
        *,
        strategy_name: str,
        initial_balance: float = 10_000.0,
        matching: Optional["MatchingConfig"] = None,
        allow_short: bool = False,
        context_extra: Optional[Dict[str, Any]] = None,
    ) -> "BacktestReport":
        """
        Pobiera dane z providera i uruchamia szybki backtest sanity-check.
        """
        # MarketDataProvider może mieć sync/async API; użyjemy ewentualnie async wersji jeśli istnieje.
        df: pd.DataFrame
        getter = getattr(provider, "get_historical_async", None)
        if callable(getter):
            df = await getter(request)
        else:
            df = provider.get_historical(request)  # type: ignore[attr-defined]
        if "close" not in df.columns:
            raise ValidationError("Dane z provider'a nie zawierają kolumny ceny 'close'")

        return self.run_backtest_on_dataframe(
            df,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy_name=strategy_name,
            initial_balance=initial_balance,
            matching=matching,
            allow_short=allow_short,
            context_extra=context_extra,
        )
