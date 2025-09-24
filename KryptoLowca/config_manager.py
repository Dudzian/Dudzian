# -*- coding: utf-8 -*-
"""Prosty menedżer konfiguracji kompatybilny ze starszym API testów."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from cryptography.fernet import Fernet, InvalidToken

__all__ = [
    "ConfigManager",
    "ConfigError",
    "ValidationError",
    "AIConfig",
    "DBConfig",
    "TradeConfig",
    "ExchangeConfig",
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

    def validate(self) -> "ExchangeConfig":
        if not self.exchange_name:
            raise ValidationError("exchange_name nie może być puste")
        return self


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
        return {
            "ai": AIConfig().validate().__dict__.copy(),
            "db": DBConfig().validate().__dict__.copy(),
            "trade": TradeConfig().validate().__dict__.copy(),
            "exchange": ExchangeConfig().validate().__dict__.copy(),
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
                    pass
        return decrypted

    def _validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ai = AIConfig(**payload.get("ai", {})).validate()
        db = DBConfig(**payload.get("db", {})).validate()
        trade = TradeConfig(**payload.get("trade", {})).validate()
        exchange = ExchangeConfig(**payload.get("exchange", {})).validate()
        return {
            "ai": ai.__dict__,
            "db": db.__dict__,
            "trade": trade.__dict__,
            "exchange": exchange.__dict__,
        }

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
