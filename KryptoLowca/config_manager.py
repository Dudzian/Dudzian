# -*- coding: utf-8 -*-
"""Prosty menedżer konfiguracji kompatybilny ze starszym API testów."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, ClassVar, Mapping

from bot_core.runtime.metadata import (
    RiskManagerSettings,
    derive_risk_manager_settings,
)

import yaml
from cryptography.fernet import Fernet, InvalidToken
import pandas as pd

try:
    # Unikamy importów cyklicznych i ciężkich w czasie testów
    if TYPE_CHECKING:
        from KryptoLowca.backtest.reports import MatchingConfig, BacktestReport  # noqa: F401
    else:
        # definiujemy nazwę dla lintów bez importu cyklicznego
        MatchingConfig = object  # type: ignore[assignment]
        BacktestReport = object  # type: ignore[assignment]
    from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Brak wymaganych zależności do działania config_manager") from exc

try:  # pragma: no cover - zależność opcjonalna w środowisku testowym
    from KryptoLowca.strategies.marketplace import (
        StrategyPreset,
        load_marketplace_presets,
        load_preset,
    )
except Exception:  # pragma: no cover - fallback dla testów bez marketplace
    StrategyPreset = Any  # type: ignore

    from typing import List
    from pathlib import Path

    def load_marketplace_presets(*, base_path: str | Path | None = None) -> List[StrategyPreset]:
        return []

    def load_preset(preset_id: str, *, base_path: str | Path | None = None) -> StrategyPreset:
        raise RuntimeError("Marketplace presets are unavailable in this environment")

__all__ = [
    "AIConfig",
    "StrategyConfig",
    "ConfigManager",
    "encrypt_config",
    "decrypt_config",
    "save_encrypted_config",
    "load_encrypted_config",
    "ValidationError",
    "ConfigError",
]

DEFAULT_CONFIG_PATH = Path("./kl_config.yaml")
BACKTEST_VALIDITY_WINDOW_S = 24 * 60 * 60  # 24h


class ConfigError(RuntimeError):
    """Ogólny błąd operacji na konfiguracji."""


class ValidationError(ValueError):
    """Nieprawidłowe wartości w konfiguracji."""


@dataclass(slots=True)
class AIConfig:
    threshold_bps: float = 5.0
    model_types: List[str] = field(default_factory=lambda: ["light", "pro"])
    max_inference_per_min: int = 60
    window_seconds: float = 30.0

    def validate(self) -> "AIConfig":
        if self.threshold_bps < 0:
            raise ValidationError("threshold_bps nie może być ujemne")
        if not self.model_types:
            raise ValidationError("model_types nie może być puste")
        if self.max_inference_per_min <= 0:
            raise ValidationError("max_inference_per_min musi być dodatnie")
        if self.window_seconds < 0:
            raise ValidationError("window_seconds nie może być ujemne")
        return self

    def as_buckets(self) -> List[Dict[str, Any]]:
        buckets: List[Dict[str, Any]] = []
        step = max(1, self.max_inference_per_min // max(1, len(self.model_types)))
        for i, model in enumerate(self.model_types):
            bucket = {
                "model": model,
                "max_calls": step,
                "window_seconds": self.window_seconds if i == 0 else self.window_seconds / 2,
            }
            # jawne rzutowania
            window_val = bucket.get("window_seconds", 0.0)
            if isinstance(window_val, (str, int, float)):
                window = float(window_val)
            else:
                window = 0.0
            bucket["window_seconds"] = window
            buckets.append(bucket)
        return buckets


@dataclass(slots=True)
class TelemetryConfig:
    enabled: bool = False
    grpc_target: Optional[str] = None
    storage_path: Optional[str] = None
    log_interval_s: float = 10.0
    retry_delay: float | str = 1.0  # sekundowa liczba lub "100ms", "1s", "2m"

    def validate(self) -> "TelemetryConfig":
        self.enabled = bool(self.enabled)

        # retry_delay może być floatem lub napisem z sufiksem
        retry_delay_raw = self.retry_delay
        if isinstance(retry_delay_raw, (int, float)):
            retry_delay_value = float(retry_delay_raw)
        elif isinstance(retry_delay_raw, str):
            retry_delay_value = float(retry_delay_raw.strip())
        else:
            raise ValidationError("retry_delay musi być floatem lub napisem")
        if retry_delay_value <= 0:
            raise ValidationError("retry_delay musi być dodatni")
        self.retry_delay = retry_delay_value

        self.log_interval_s = float(self.log_interval_s)
        if self.log_interval_s <= 0:
            raise ValidationError("log_interval_s musi być dodatnie")

        if self.storage_path is not None:
            self.storage_path = str(self.storage_path)
        if self.grpc_target is not None:
            self.grpc_target = str(self.grpc_target)
        return self


@dataclass(slots=True)
class StrategyConfig:
    """Parametry strategii oraz sztywne limity ryzyka dla auto-tradingu."""

    BACKTEST_VALIDITY_WINDOW_S: ClassVar[float] = BACKTEST_VALIDITY_WINDOW_S

    preset: str = "SAFE"
    mode: str = "demo"
    max_leverage: float = 1.0
    max_position_usd: float = 100.0
    max_position_notional_pct: float = 0.02
    max_daily_loss_usd: float = 50.0
    trade_risk_pct: float = 0.01
    default_sl: float = 0.01
    default_tp: float = 0.02
    violation_cooldown_s: float = 300.0
    reduce_only_after_violation: bool = True

    # potwierdzenia i zgodności
    compliance_confirmed: bool = False
    api_keys_configured: bool = False
    acknowledged_risk_disclaimer: bool = False

    # znacznik pozytywnego backtestu (epoch seconds)
    backtest_passed_at: Optional[float] = None

    def validate(self) -> "StrategyConfig":
        self.max_leverage = float(self.max_leverage)
        self.max_position_usd = float(self.max_position_usd)
        self.max_position_notional_pct = float(self.max_position_notional_pct)
        self.max_daily_loss_usd = float(self.max_daily_loss_usd)
        self.trade_risk_pct = float(self.trade_risk_pct)
        self.default_sl = float(self.default_sl)
        self.default_tp = float(self.default_tp)
        self.violation_cooldown_s = float(self.violation_cooldown_s)

        if self.max_leverage <= 0:
            raise ValidationError("max_leverage musi być dodatnie")
        if self.max_position_usd <= 0:
            raise ValidationError("max_position_usd musi być dodatnie")
        if self.max_position_notional_pct < 0 or self.max_position_notional_pct > 1:
            raise ValidationError(
                "max_position_notional_pct musi być w zakresie [0, 1]"
            )
        if self.max_daily_loss_usd <= 0:
            raise ValidationError("max_daily_loss_usd musi być dodatnie")
        if not (0 <= self.trade_risk_pct <= 1):
            raise ValidationError("trade_risk_pct musi być w zakresie [0, 1]")
        if not (0 <= self.default_sl < 1) or not (0 <= self.default_tp < 1):
            raise ValidationError("default_sl i default_tp muszą być w [0,1)")
        if self.default_sl < 0 or self.default_tp < 0:
            raise ValidationError("default_sl i default_tp muszą być nieujemne")
        if self.violation_cooldown_s <= 0:
            raise ValidationError("violation_cooldown_s musi być dodatnie")
        self.reduce_only_after_violation = bool(self.reduce_only_after_violation)

        # Normalizacja znacznika zaliczonego backtestu
        raw_backtest_ts = getattr(self, "backtest_passed_at", None)
        normalized_backtest_ts: Optional[float]
        if raw_backtest_ts is None:
            normalized_backtest_ts = None
        elif isinstance(raw_backtest_ts, (int, float)):
            normalized_backtest_ts = float(raw_backtest_ts)
        elif isinstance(raw_backtest_ts, str):
            s = raw_backtest_ts.strip()
            if not s or s in {"0", "0.0"}:
                normalized_backtest_ts = None
            else:
                try:
                    normalized_backtest_ts = float(s)
                except ValueError as exc:
                    raise ValidationError(
                        "backtest_passed_at musi być znacznikiem czasu w sekundach"
                    ) from exc
        else:
            raise ValidationError(
                "backtest_passed_at musi być znacznikiem czasu w sekundach"
            )
        if normalized_backtest_ts is not None and normalized_backtest_ts <= 0:
            raise ValidationError("backtest_passed_at musi być dodatnie")
        self.backtest_passed_at = normalized_backtest_ts

        # Normalizacja flag zgodności (zachowuje kompatybilność ze starszym API)
        for field_name in (
            "compliance_confirmed",
            "api_keys_configured",
            "acknowledged_risk_disclaimer",
        ):
            setattr(self, field_name, bool(getattr(self, field_name)))

        self.preset = (self.preset or "").strip().upper() or "CUSTOM"
        return self

    def derive_risk_manager_settings(
        self,
        profile: Mapping[str, Any] | Any | None,
        *,
        profile_name: str | None = None,
        defaults: Mapping[str, Any] | RiskManagerSettings | None = None,
    ) -> RiskManagerSettings:
        """Buduje ``RiskManagerSettings`` na bazie profilu runtime i lokalnych limitów."""

        default_mapping: Mapping[str, Any]
        if isinstance(defaults, RiskManagerSettings):
            return derive_risk_manager_settings(
                profile,
                profile_name=profile_name,
                defaults=defaults,
            )
        if isinstance(defaults, Mapping):
            merged: Dict[str, Any] = dict(defaults)
            merged.setdefault("max_risk_per_trade", self.max_position_notional_pct)
            merged.setdefault(
                "max_daily_loss_pct",
                self.trade_risk_pct or self.max_position_notional_pct,
            )
            default_mapping = merged
        else:
            default_mapping = {
                "max_risk_per_trade": self.max_position_notional_pct,
                "max_daily_loss_pct": self.trade_risk_pct or self.max_position_notional_pct,
            }
        return derive_risk_manager_settings(
            profile,
            profile_name=profile_name,
            defaults=default_mapping,
        )

    def apply_risk_profile(
        self,
        profile: Mapping[str, Any] | Any,
        *,
        prefer_profile_leverage: bool = True,
        profile_name: str | None = None,
    ) -> "StrategyConfig":
        """Zwraca kopię konfiguracji zaktualizowaną o limity z profilu ryzyka."""

        clone = replace(self)

        leverage: float = float(clone.max_leverage)
        if hasattr(profile, "max_leverage"):
            try:
                leverage = float(getattr(profile, "max_leverage"))
            except Exception:
                leverage = float(clone.max_leverage)
        elif isinstance(profile, Mapping) and "max_leverage" in profile:
            try:
                leverage = float(profile["max_leverage"])
            except Exception:
                leverage = float(clone.max_leverage)

        settings = self.derive_risk_manager_settings(
            profile,
            profile_name=profile_name,
        )

        clone.max_position_notional_pct = min(1.0, max(0.0, settings.max_risk_per_trade))
        per_trade = min(settings.max_risk_per_trade, settings.max_daily_loss_pct)
        clone.trade_risk_pct = min(1.0, max(0.0, per_trade))

        leverage_value = max(0.0, leverage)
        if leverage_value > 0:
            if prefer_profile_leverage:
                clone.max_leverage = leverage_value
            else:
                clone.max_leverage = max(clone.max_leverage, leverage_value)

        return clone.validate()

    @classmethod
    def presets(cls) -> Dict[str, "StrategyConfig"]:
        base_presets = {
            "SAFE": cls(
                preset="SAFE",
                mode="demo",
                max_leverage=1.0,
                max_position_usd=100.0,
                max_daily_loss_usd=50.0,
                default_sl=0.01,
                default_tp=0.02,
                violation_cooldown_s=300.0,
                reduce_only_after_violation=True,
            ),
            "AGGRO": cls(
                preset="AGGRO",
                mode="demo",
                max_leverage=3.0,
                max_position_usd=500.0,
                max_daily_loss_usd=250.0,
                default_sl=0.02,
                default_tp=0.04,
                violation_cooldown_s=120.0,
                reduce_only_after_violation=False,
            ),
        }
        return base_presets

    def mark_backtest_passed(self) -> "StrategyConfig":
        self.backtest_passed_at = time.time()
        return self

    def has_fresh_backtest(
        self,
        *,
        freshness_window: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Zwraca True, jeśli konfiguracja posiada aktualne potwierdzenie backtestu."""
        if not self.backtest_passed_at:
            return False
        window = self.BACKTEST_VALIDITY_WINDOW_S if freshness_window is None else float(freshness_window)
        if window <= 0:
            return True
        reference_ts = float(now if now is not None else time.time())
        return (reference_ts - float(self.backtest_passed_at)) <= window


class UserConfigStore:
    """Prosta pamięć konfiguracji użytkowników (w RAM dla testów)."""

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
        cfg = dict(config)
        if "strategy" in cfg and isinstance(cfg["strategy"], StrategyConfig):
            cfg["strategy"] = asdict(cfg["strategy"])
        self._user_configs.setdefault(user_id, {})[name] = cfg

    async def get_user_config(self, user_id: int, name: str) -> Dict[str, Any]:
        try:
            return self._user_configs[user_id][name]
        except KeyError as exc:  # pragma: no cover - łatwe do testów
            raise ConfigError(f"Nie znaleziono konfiguracji {name!r} dla user_id={user_id}") from exc


def encrypt_config(data: Dict[str, Any], key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(json.dumps(data).encode("utf-8"))


def decrypt_config(token: bytes, key: bytes) -> Dict[str, Any]:
    f = Fernet(key)
    try:
        payload = f.decrypt(token)
    except InvalidToken as exc:
        raise ConfigError("Zły klucz lub uszkodzony plik konfiguracyjny") from exc
    try:
        text_payload = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError("Nieprawidłowe kodowanie konfiguracji") from exc

    try:
        data = json.loads(text_payload)
    except json.JSONDecodeError as exc:
        raise ConfigError("Nie można odczytać konfiguracji") from exc

    if not isinstance(data, dict):
        raise ConfigError("Nieprawidłowy format konfiguracji: oczekiwano obiektu JSON")

    return data


def save_encrypted_config(path: str | Path, data: Dict[str, Any], key: bytes) -> None:
    path = Path(path)
    path.write_bytes(encrypt_config(data, key))


def load_encrypted_config(path: str | Path, key: bytes) -> Dict[str, Any]:
    path = Path(path)
    return decrypt_config(path.read_bytes(), key)


class ConfigManager:
    """Menedżer konfiguracji — ładowanie/presety/łączenie/telemetria/marketplace."""

    def __init__(self, *, marketplace_dir: str | Path | None = None) -> None:
        self._marketplace_dir = Path(marketplace_dir) if marketplace_dir else None
        self._current_config: Dict[str, Any] = {}
        self._telemetry: Optional[TelemetryConfig] = None

    # ---- TELEMETRIA ----

    def configure_telemetry(self, cfg: TelemetryConfig) -> None:
        self._telemetry = cfg.validate()

    async def ping_telemetry(self) -> None:
        if not self._telemetry or not self._telemetry.enabled:
            return
        # w testach tylko symulujemy opóźnienie itd.
        await asyncio.sleep(0)

    # ---- MARKETPLACE PRESETS ----

    def list_marketplace_presets(self) -> List[StrategyPreset]:
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

        config_mapping: Mapping[str, Any]
        if isinstance(getattr(preset, "config", None), Mapping):
            config_mapping = preset.config  # type: ignore[assignment]
        elif isinstance(preset, Mapping):
            config_mapping = preset
        else:
            raise ConfigError("Preset does not expose a valid config mapping")

        for section, payload in dict(config_mapping).items():
            merged[section] = payload

        # metadane
        meta = dict(merged.get("meta", {}))
        meta["last_preset"] = preset_id
        meta["actor"] = actor or "system"
        if user_confirmed:
            meta["user_confirmed"] = True
        if note:
            meta["note"] = note
        merged["meta"] = meta
        self._current_config = merged
        return merged

    # ---- STRATEGIA / LIVE WYMOGI ----

    @staticmethod
    def _ensure_live_mode_requirements(strategy_section: Dict[str, Any]) -> None:
        """Prosta walidacja wymogów trybu LIVE (mypy-friendly)."""
        mode = str(strategy_section.get("mode", "demo")).strip().lower()
        if mode != "live":
            return

        backtest_ts = strategy_section.get("backtest_passed_at")
        if not isinstance(backtest_ts, (int, float)) or backtest_ts <= 0:
            raise ValidationError(
                "Tryb LIVE wymaga aktualnego potwierdzenia backtestu (brak wpisu)."
            )

        freshness_window = StrategyConfig.BACKTEST_VALIDITY_WINDOW_S
        if freshness_window > 0:
            if (time.time() - float(backtest_ts)) > float(freshness_window):
                raise ValidationError(
                    "Wynik backtestu jest przeterminowany – uruchom backtest ponownie."
                )

    # ---- ŁADOWANIE Z PLIKU ----

    def load_from_path(self, path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Plik {str(path)!r} nie istnieje")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError("Plik konfiguracyjny musi być słownikiem (YAML mapping)")

        # sekcja strategy do dataclass
        strat_raw = data.get("strategy", {})
        if not isinstance(strat_raw, dict):
            raise ConfigError("Sekcja 'strategy' musi być słownikiem")
        strategy = StrategyConfig(**strat_raw).validate()
        data["strategy"] = strategy

        # sekcja telemetry do dataclass
        telemetry_raw = data.get("telemetry", None)
        if telemetry_raw is not None:
            if not isinstance(telemetry_raw, dict):
                raise ConfigError("Sekcja 'telemetry' musi być słownikiem")
            telemetry = TelemetryConfig(**telemetry_raw).validate()
            data["telemetry"] = telemetry
            self._telemetry = telemetry

        # prosta walidacja LIVE
        self._ensure_live_mode_requirements(asdict(strategy))

        self._current_config = data
        return data

    # ---- ZAPIS/JWT/DATAFRAME itp. (reszta pomocnicza) ----

    @staticmethod
    def to_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
        flat = {}
        for k, v in config.items():
            if hasattr(v, "__dataclass_fields__"):
                flat[k] = asdict(v)
            else:
                flat[k] = v
        return pd.json_normalize(flat, sep=".")

    def dump(self) -> Dict[str, Any]:
        out = dict(self._current_config)
        if "strategy" in out and isinstance(out["strategy"], StrategyConfig):
            out["strategy"] = asdict(out["strategy"])
        if "telemetry" in out and isinstance(out["telemetry"], TelemetryConfig):
            out["telemetry"] = asdict(out["telemetry"])
        return out

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.dump(), f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

    @staticmethod
    def _iso_to_ts(iso_s: Optional[str]) -> Optional[float]:
        if iso_s is None:
            return None
        return datetime.fromisoformat(iso_s).replace(tzinfo=timezone.utc).timestamp()
