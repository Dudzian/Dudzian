# -*- coding: utf-8 -*-
"""Prosty menedżer konfiguracji kompatybilny ze starszym API testów."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml
from cryptography.fernet import Fernet, InvalidToken
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - tylko dla typowania
    from KryptoLowca.backtest.simulation import BacktestReport, MatchingConfig
    from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest

from KryptoLowca.strategies.marketplace import (
    StrategyPreset,
    load_marketplace_presets,
    load_preset,
)

__all__ = [
    "ConfigManager",
    "ConfigError",
    "ValidationError",
    "AIConfig",
    "DBConfig",
    "TradeConfig",
    "ExchangeConfig",
    "StrategyConfig",
]


class ConfigError(RuntimeError):
    """Ogólny błąd operacji na konfiguracji."""


class ValidationError(ValueError):
    """Nieprawidłowe wartości w konfiguracji."""


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
        self.retry_delay = float(self.retry_delay)
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
        if not (0.0 < self.max_position_notional_pct <= 1.0):
            raise ValidationError("max_position_notional_pct musi być w zakresie (0, 1]")
        if not (0.0 < self.trade_risk_pct <= 1.0):
            raise ValidationError("trade_risk_pct musi być w zakresie (0, 1]")
        if self.default_sl < 0 or self.default_tp < 0:
            raise ValidationError("default_sl i default_tp muszą być nieujemne")
        if self.violation_cooldown_s <= 0:
            raise ValidationError("violation_cooldown_s musi być dodatnie")
        self.reduce_only_after_violation = bool(self.reduce_only_after_violation)

        # Normalizacja flag zgodności (zachowuje kompatybilność ze starszym API)
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

        # W trybie LIVE wymagamy spełnienia wszystkich potwierdzeń
        if self.mode == "live":
            missing: List[str] = []
            if not self.api_keys_configured:
                missing.append("api_keys_configured")
            if not self.compliance_confirmed:
                missing.append("compliance_confirmed")
            if not self.acknowledged_risk_disclaimer:
                missing.append("acknowledged_risk_disclaimer")
            if missing:
                raise ValidationError(
                    "Nie można przełączyć strategii w tryb LIVE bez potwierdzenia: "
                    + ", ".join(missing)
                )

        self.preset = (self.preset or "").strip().upper() or "CUSTOM"
        return self

    @classmethod
    def presets(cls) -> Dict[str, "StrategyConfig"]:
        base_presets = {
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
                compliance_confirmed=False,
                api_keys_configured=False,
                acknowledged_risk_disclaimer=False,
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
                compliance_confirmed=False,
                api_keys_configured=False,
                acknowledged_risk_disclaimer=False,
            ),
        }
        try:
            from KryptoLowca.strategies.presets import load_builtin_presets

            for preset in load_builtin_presets():
                raw_section = preset.config.get("strategy", {})
                strategy_section: Dict[str, Any]
                if isinstance(raw_section, dict):
                    strategy_section = dict(raw_section)
                else:
                    strategy_section = {}
                if not strategy_section:
                    continue
                name = preset.preset_id.upper()
                merged = asdict(cls())
                merged.update(strategy_section)
                base_presets[name] = cls(**merged)
        except Exception:
            pass
        return base_presets

    @classmethod
    def from_preset(cls, name: str) -> "StrategyConfig":
        presets = cls.presets()
        preset = presets.get((name or "").strip().upper())
        return (preset or cls(preset="CUSTOM")).validate()

    def guard_backtest(self, report: "BacktestReport") -> "StrategyConfig":
        """Sprawdza wyniki backtestu i zwraca konfigurację, jeśli test został zaliczony."""
        from KryptoLowca.backtest.simulation import evaluate_strategy_backtest

        evaluate_strategy_backtest(asdict(self), report)
        return self


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

    ENCRYPTED_FIELDS = {
        "exchange": {"api_key", "api_secret"},
    }

    def __init__(self, config_path: Path, encryption_key: Optional[bytes] = None) -> None:
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(encryption_key) if encryption_key else None
        self.db_manager = _InMemoryDB()
        self._current_config: Dict[str, Any] = self._default_config()
        self._marketplace_dir: Optional[Path] = None
        self._versions_dir: Path = self.config_path.parent / "versions"

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

    def _encrypt_section(self, section: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._fernet:
            return data
        encrypted = dict(data)
        for key in self.ENCRYPTED_FIELDS.get(section, set()):
            value = encrypted.get(key)
            if isinstance(value, str) and value:
                encrypted[key] = self._fernet.encrypt(value.encode()).decode()
        return encrypted

    def _decrypt_section(self, section: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._fernet:
            return data
        decrypted = dict(data)
        for key in self.ENCRYPTED_FIELDS.get(section, set()):
            value = decrypted.get(key)
            if isinstance(value, str) and value:
                try:
                    decrypted[key] = self._fernet.decrypt(value.encode()).decode()
                except InvalidToken:
                    # jeśli nie jest zaszyfrowane lub klucz się zmienił – pozostaw oryginał
                    pass
        return decrypted

    def _record_snapshot(
        self,
        *,
        preset_id: str,
        marketplace_version: Optional[str],
        config: Dict[str, Any],
        actor: Optional[str],
        note: Optional[str],
        source: str,
    ) -> str:
        timestamp = datetime.now(timezone.utc)
        version_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        actor_name = (actor or "unknown").strip() or "unknown"
        directory = self._versions_dir / preset_id
        directory.mkdir(parents=True, exist_ok=True)

        try:
            serialized_config = json.loads(json.dumps(config, default=str))
        except TypeError:
            serialized_config = json.loads(json.dumps(self._validate_payload(config)))
        exchange_section = serialized_config.get("exchange", {})
        if isinstance(exchange_section, dict):
            serialized_config["exchange"] = self._encrypt_section("exchange", exchange_section)

        metadata = {
            "preset_id": preset_id,
            "marketplace_version": marketplace_version,
            "applied_at": timestamp.isoformat(),
            "actor": actor_name,
            "note": note,
            "source": source,
            "config": serialized_config,
        }

        file_path = directory / f"{version_id}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

        self._append_audit_entry(
            timestamp=timestamp,
            preset_id=preset_id,
            actor=actor_name,
            marketplace_version=marketplace_version,
            source=source,
            note=note,
        )
        return version_id

    def _append_audit_entry(
        self,
        *,
        timestamp: datetime,
        preset_id: str,
        actor: str,
        marketplace_version: Optional[str],
        source: str,
        note: Optional[str],
    ) -> None:
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self._versions_dir / "audit.log"
        line = (
            f"{timestamp.isoformat()}|{actor}|{preset_id}|"
            f"{marketplace_version or '-'}|{source}|{note or ''}\n"
        )
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def _ensure_live_mode_requirements(
        self, config: Dict[str, Any], *, user_confirmed: bool
    ) -> None:
        strategy_section = config.get("strategy", {})
        if not isinstance(strategy_section, dict):
            return
        mode = str(strategy_section.get("mode", "demo")).strip().lower()
        if mode != "live":
            return
        missing = [
            flag
            for flag in _COMPLIANCE_FLAGS
            if not bool(strategy_section.get(flag))
        ]
        if missing:
            raise ValidationError(
                "Tryb LIVE wymaga potwierdzonych flag zgodności: "
                + ", ".join(missing)
            )
        if not user_confirmed:
            raise ValidationError(
                "Aktywacja trybu LIVE wymaga potwierdzenia użytkownika (ustaw opcję confirm)."
            )

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

    def apply_marketplace_preset(
        self,
        preset_id: str,
        *,
        actor: str | None = None,
        user_confirmed: bool = False,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        preset = self.get_marketplace_preset(preset_id)
        merged = dict(self._current_config)
        for section, payload in preset.config.items():
            if not isinstance(payload, dict):
                continue
            existing = dict(merged.get(section, {}))
            existing.update(payload)
            merged[section] = existing
        sanitized = self._validate_payload(merged)
        self._ensure_live_mode_requirements(sanitized, user_confirmed=user_confirmed)
        self._current_config = sanitized
        self._record_snapshot(
            preset_id=preset.preset_id,
            marketplace_version=preset.version,
            config=sanitized,
            actor=actor,
            note=note,
            source="marketplace",
        )
        return self._current_config

    def get_preset_history(self, preset_id: str) -> List[Dict[str, Any]]:
        directory = self._versions_dir / preset_id
        if not directory.exists():
            return []
        history: List[Dict[str, Any]] = []
        for file in sorted(directory.glob("*.json")):
            try:
                with file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            history.append(
                {
                    "version_id": file.stem,
                    "preset_id": preset_id,
                    "marketplace_version": data.get("marketplace_version"),
                    "applied_at": data.get("applied_at"),
                    "actor": data.get("actor"),
                    "note": data.get("note"),
                    "source": data.get("source"),
                }
            )
        return history

    def rollback_preset(
        self,
        preset_id: str,
        version_id: str,
        *,
        actor: str | None = None,
        user_confirmed: bool = False,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        directory = self._versions_dir / preset_id
        file_path = directory / f"{version_id}.json"
        if not file_path.exists():
            raise ConfigError(
                f"Brak wersji '{version_id}' dla presetu '{preset_id}'"
            )
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        payload = data.get("config", {})
        if isinstance(payload, dict):
            exchange_section = payload.get("exchange", {})
            payload["exchange"] = self._decrypt_section("exchange", exchange_section)
        sanitized = self._validate_payload(payload)
        self._ensure_live_mode_requirements(sanitized, user_confirmed=user_confirmed)
        self._current_config = sanitized
        self._record_snapshot(
            preset_id=preset_id,
            marketplace_version=data.get("marketplace_version"),
            config=sanitized,
            actor=actor,
            note=note or f"rollback to {version_id}",
            source="rollback",
        )
        return self._current_config

    async def save_config(
        self,
        config: Dict[str, Any],
        *,
        actor: Optional[str] = None,
        preset_id: Optional[str] = None,
        note: Optional[str] = None,
        source: str = "manual",
    ) -> None:
        sanitized = self._validate_payload(config)
        to_disk = dict(sanitized)
        to_disk["exchange"] = self._encrypt_section("exchange", to_disk["exchange"])
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(to_disk, fh, sort_keys=True)
        self._current_config = sanitized
        if preset_id:
            self._record_snapshot(
                preset_id=preset_id,
                marketplace_version=None,
                config=sanitized,
                actor=actor,
                note=note,
                source=source,
            )

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
            exchange_section = data.get("exchange", {})
            data["exchange"] = self._decrypt_section("exchange", exchange_section)
            self._current_config = self._validate_payload(data)
        else:
            self._current_config = self._default_config()
        return self._current_config

    def write_template(self, *, force: bool = False) -> Path:
        """Generuje plik YAML z domyślną konfiguracją."""

        if self.config_path.exists() and not force:
            return self.config_path
        template = self._default_config()
        template["exchange"] = {
            key: ("" if key in self.ENCRYPTED_FIELDS.get("exchange", set()) else value)
            for key, value in template["exchange"].items()
        }
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(template, fh, sort_keys=True)
        return self.config_path

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

    def run_backtest_on_dataframe(
        self,
        data: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        strategy_name: Optional[str] = None,
        initial_balance: float = 10_000.0,
        matching: Optional[MatchingConfig] = None,
        allow_short: bool = False,
        report_dir: Optional[Path] = None,
    ) -> "BacktestReport":
        from KryptoLowca.backtest.reporting import export_report
        from KryptoLowca.backtest.simulation import BacktestEngine, MatchingConfig

        if data.empty:
            raise ValidationError("Backtest wymaga niepustego zbioru danych")
        if "close" not in data.columns:
            raise ValidationError("Dane wymagają kolumny 'close'")

        strategy = self.load_strategy_config()
        context_extra = {
            "mode": "demo",
            "trade_risk_pct": strategy.trade_risk_pct,
            "max_position_notional_pct": strategy.max_position_notional_pct,
            "max_leverage": strategy.max_leverage,
        }
        engine = BacktestEngine(
            strategy_name=(strategy_name or strategy.preset or "SAFE"),
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=initial_balance,
            matching=matching or MatchingConfig(),
            allow_short=allow_short,
            context_extra=context_extra,
        )
        report = engine.run()
        strategy.guard_backtest(report)
        if report_dir is not None:
            export_report(report, Path(report_dir))
        return report

    async def preflight_backtest(
        self,
        provider: "MarketDataProvider",
        request: "MarketDataRequest",
        *,
        strategy_name: Optional[str] = None,
        initial_balance: float = 10_000.0,
        allow_short: bool = False,
        report_dir: Optional[Path] = None,
    ) -> "BacktestReport":
        df = await provider.get_historical_async(request)
        if df.empty:
            raise ValidationError("Provider zwrócił pusty zbiór danych do backtestu")
        return self.run_backtest_on_dataframe(
            df,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy_name=strategy_name,
            initial_balance=initial_balance,
            matching=None,
            allow_short=allow_short,
            report_dir=report_dir,
        )
