# -*- coding: utf-8 -*-
"""Prosty menedżer konfiguracji kompatybilny ze starszym API testów."""
from __future__ import annotations

import asyncio
import base64
import json
import math
import statistics
import time
import uuid
from copy import deepcopy
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
    from KryptoLowca.security.api_key_manager import APIKeyManager
    from KryptoLowca.strategies.marketplace import (
        StrategyPreset,
        load_marketplace_index,
        load_marketplace_presets,
        load_preset,
    )
except Exception:  # pragma: no cover - fallback dla testów bez marketplace
    StrategyPreset = Any  # type: ignore
    APIKeyManager = Any  # type: ignore

    from typing import List
    from pathlib import Path

    def load_marketplace_presets(*, base_path: str | Path | None = None) -> List[StrategyPreset]:
        return []

    def load_preset(preset_id: str, *, base_path: str | Path | None = None) -> StrategyPreset:
        raise RuntimeError("Marketplace presets are unavailable in this environment")

    def load_marketplace_index(
        *, base_path: str | Path | None = None
    ) -> Dict[str, StrategyPreset]:
        return {}

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

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        encryption_key: bytes | None = None,
        marketplace_dir: str | Path | None = None,
    ) -> None:
        base_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        if base_path.is_dir():
            base_path = base_path / DEFAULT_CONFIG_PATH.name
        self.config_path: Path = base_path
        self.encryption_key = encryption_key
        self._marketplace_dir = Path(marketplace_dir) if marketplace_dir else None
        self._current_config: Dict[str, Any] = deepcopy(self._default_config())
        self._telemetry: Optional[TelemetryConfig] = None
        self._versions_dir = self.config_path.parent / "versions"
        self._history_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._api_key_manager: APIKeyManager | None = None
        self._marketplace_cache: Dict[str, StrategyPreset] | None = None
        self._marketplace_cache_mtime: float | None = None
        self._history_enabled = config_path is not None

    @classmethod
    async def create(
        cls,
        *,
        config_path: str | Path,
        encryption_key: bytes,
        marketplace_dir: str | Path | None = None,
    ) -> "ConfigManager":
        manager = cls(config_path, encryption_key=encryption_key, marketplace_dir=marketplace_dir)
        await manager.load_config()
        return manager

    def configure_telemetry(self, cfg: TelemetryConfig) -> None:
        self._telemetry = cfg.validate()

    async def ping_telemetry(self) -> None:
        if not self._telemetry or not self._telemetry.enabled:
            return
        await asyncio.sleep(0)

    def set_marketplace_directory(self, path: Path | str | None) -> None:
        self._marketplace_dir = Path(path) if path else None
        self.invalidate_marketplace_cache()

    def invalidate_marketplace_cache(self) -> None:
        self._marketplace_cache = None
        self._marketplace_cache_mtime = None

    def _load_marketplace_index(self) -> Dict[str, StrategyPreset]:
        directory = self._marketplace_dir
        mtime: float | None = None
        if directory is not None:
            try:
                mtime = directory.stat().st_mtime
            except OSError:
                mtime = None
        if (
            self._marketplace_cache is not None
            and (mtime is None or self._marketplace_cache_mtime == mtime)
        ):
            return dict(self._marketplace_cache)
        index = load_marketplace_index(base_path=directory)
        if not index:
            try:
                from KryptoLowca.strategies import marketplace as marketplace_module  # type: ignore

                presets = marketplace_module.load_marketplace_presets(base_path=directory)
                index = {preset.preset_id: preset for preset in presets}
            except Exception:
                index = {}
        self._marketplace_cache = dict(index)
        self._marketplace_cache_mtime = mtime
        return dict(index)

    def list_marketplace_presets(self) -> List[StrategyPreset]:
        return list(self._load_marketplace_index().values())

    def get_marketplace_preset(self, preset_id: str) -> StrategyPreset:
        index = self._load_marketplace_index()
        try:
            return index[preset_id]
        except KeyError as exc:
            raise ConfigError(f"Brak presetu o identyfikatorze '{preset_id}'") from exc

    def get_marketplace_ranking(self) -> List[Dict[str, Any]]:
        presets = self.list_marketplace_presets()
        sorted_presets = sorted(
            presets,
            key=lambda preset: (
                preset.evaluation_rank() if preset.evaluation_rank() is not None else float("inf"),
                preset.name.lower(),
            ),
        )
        ranking: List[Dict[str, Any]] = []
        for preset in sorted_presets:
            evaluation = preset.evaluation
            backtest = evaluation.backtest.to_dict() if evaluation and evaluation.backtest else None
            ranking.append(
                {
                    "preset_id": preset.preset_id,
                    "name": preset.name,
                    "rank": evaluation.rank if evaluation else None,
                    "risk_label": preset.effective_risk_label(),
                    "risk_score": evaluation.risk_score if evaluation else None,
                    "backtest": backtest,
                }
            )
        return ranking

    def get_marketplace_risk_labels(self) -> Dict[str, str]:
        return {
            preset.preset_id: preset.effective_risk_label() or "unknown"
            for preset in self.list_marketplace_presets()
        }

    def get_marketplace_risk_summary(self) -> Dict[str, Dict[str, Any]]:
        """Zwraca zagregowane statystyki ryzyka dla presetów marketplace."""

        summary: Dict[str, Dict[str, Any]] = {}
        total_presets = 0
        def _percentile(values: List[float], percentile: float) -> float | None:
            if not values:
                return None
            if len(values) == 1:
                return float(values[0])
            sorted_vals = sorted(float(v) for v in values)
            if percentile <= 0:
                return sorted_vals[0]
            if percentile >= 1:
                return sorted_vals[-1]
            position = (len(sorted_vals) - 1) * percentile
            lower_index = math.floor(position)
            upper_index = math.ceil(position)
            if lower_index == upper_index:
                return sorted_vals[int(position)]
            lower_value = sorted_vals[lower_index]
            upper_value = sorted_vals[upper_index]
            fraction = position - lower_index
            return lower_value + (upper_value - lower_value) * fraction

        def _shape(values: List[float]) -> tuple[float | None, float | None]:
            if not values:
                return (None, None)
            if len(values) == 1:
                return (0.0, 0.0)
            numeric = [float(v) for v in values]
            mean_value = sum(numeric) / len(numeric)
            centered = [value - mean_value for value in numeric]
            moment2 = sum(diff * diff for diff in centered) / len(centered)
            if moment2 <= 0:
                return (0.0, 0.0)
            moment3 = sum(diff ** 3 for diff in centered) / len(centered)
            moment4 = sum(diff ** 4 for diff in centered) / len(centered)
            std_cubed = moment2 ** 1.5
            if std_cubed == 0:
                return (0.0, 0.0)
            skewness = moment3 / std_cubed
            kurtosis = (moment4 / (moment2 * moment2)) - 3.0
            return (skewness, kurtosis)

        def _jarque_bera(count: int, skewness: float | None, kurtosis: float | None) -> float | None:
            if count <= 0 or skewness is None or kurtosis is None:
                return None
            return (count / 6.0) * ((skewness ** 2) + ((kurtosis ** 2) / 4.0))

        def _pearson_covariance(pairs: List[tuple[float, float]]) -> tuple[float | None, float | None]:
            if len(pairs) < 2:
                return (None, None)
            scores = [float(score) for score, _ in pairs]
            ranks = [float(rank) for _, rank in pairs]
            mean_score = sum(scores) / len(scores)
            mean_rank = sum(ranks) / len(ranks)
            covariance = sum(
                (score - mean_score) * (rank - mean_rank)
                for score, rank in zip(scores, ranks)
            ) / len(pairs)
            score_variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            rank_variance = sum((rank - mean_rank) ** 2 for rank in ranks) / len(ranks)
            if score_variance <= 0 or rank_variance <= 0:
                return (covariance, None)
            correlation = covariance / math.sqrt(score_variance * rank_variance)
            return (covariance, correlation)

        def _spearman(pairs: List[tuple[float, float]]) -> float | None:
            if len(pairs) < 2:
                return None

            def _rank(values: List[float]) -> List[float]:
                sorted_indices = sorted(range(len(values)), key=lambda idx: values[idx])
                ranks = [0.0] * len(values)
                index = 0
                while index < len(values):
                    start = index
                    current_value = values[sorted_indices[index]]
                    while index + 1 < len(values) and values[sorted_indices[index + 1]] == current_value:
                        index += 1
                    end = index
                    avg_rank = (start + end + 2) / 2.0
                    for position in range(start, end + 1):
                        ranks[sorted_indices[position]] = avg_rank
                    index += 1
                return ranks

            scores = [float(score) for score, _ in pairs]
            ranks = [float(rank) for _, rank in pairs]
            score_ranks = _rank(scores)
            rank_ranks = _rank(ranks)
            covariance, correlation = _pearson_covariance(list(zip(score_ranks, rank_ranks)))
            if covariance is None:
                return None
            return correlation

        for preset in self.list_marketplace_presets():
            risk_label = preset.effective_risk_label() or preset.risk_level or "unknown"
            entry = summary.setdefault(
                risk_label,
                {
                    "count": 0,
                    "_score_sum": 0.0,
                    "_score_count": 0,
                    "_score_sq_sum": 0.0,
                    "_score_min": None,
                    "_score_max": None,
                    "_scores": [],
                    "best_rank": None,
                    "worst_rank": None,
                    "top_preset": None,
                    "_worst_preset": None,
                    "_rank_sum": 0.0,
                    "_rank_count": 0,
                    "_ranks": [],
                    "_rank_sq_sum": 0.0,
                    "_score_rank_pairs": [],
                },
            )
            entry["count"] += 1
            total_presets += 1

            evaluation = preset.evaluation
            if evaluation and evaluation.risk_score is not None:
                score = float(evaluation.risk_score)
                entry["_score_sum"] += score
                entry["_score_count"] += 1
                entry["_score_sq_sum"] += score * score
                entry["_scores"].append(score)
                if entry["_score_min"] is None or score < entry["_score_min"]:
                    entry["_score_min"] = score
                if entry["_score_max"] is None or score > entry["_score_max"]:
                    entry["_score_max"] = score

            rank = preset.evaluation_rank()
            if rank is not None:
                entry["_rank_sum"] += float(rank)
                entry["_rank_count"] += 1
                entry["_ranks"].append(float(rank))
                entry["_rank_sq_sum"] += float(rank) * float(rank)
                best_rank = entry.get("best_rank")
                if best_rank is None or rank < best_rank:
                    entry["best_rank"] = rank
                    entry["top_preset"] = preset.preset_id
                worst_rank = entry.get("worst_rank")
                if worst_rank is None or rank > worst_rank:
                    entry["worst_rank"] = rank
                    entry["_worst_preset"] = preset.preset_id
                if evaluation and evaluation.risk_score is not None:
                    entry["_score_rank_pairs"].append((float(evaluation.risk_score), float(rank)))

        if summary:
            overall: Dict[str, Any] = {
                "count": 0,
                "_score_sum": 0.0,
                "_score_count": 0,
                "_score_sq_sum": 0.0,
                "_score_min": None,
                "_score_max": None,
                "_scores": [],
                "best_rank": None,
                "worst_rank": None,
                "top_preset": None,
                "_worst_preset": None,
                "_rank_sum": 0.0,
                "_rank_count": 0,
                "_ranks": [],
                "_rank_sq_sum": 0.0,
                "_score_rank_pairs": [],
            }

            for data in summary.values():
                overall["count"] += data.get("count", 0)
                overall["_score_sum"] += float(data.get("_score_sum", 0.0))
                overall["_score_count"] += int(data.get("_score_count", 0))
                overall["_score_sq_sum"] += float(data.get("_score_sq_sum", 0.0))
                if data.get("_score_min") is not None:
                    if overall["_score_min"] is None or data["_score_min"] < overall["_score_min"]:
                        overall["_score_min"] = data["_score_min"]
                if data.get("_score_max") is not None:
                    if overall["_score_max"] is None or data["_score_max"] > overall["_score_max"]:
                        overall["_score_max"] = data["_score_max"]
                overall["_scores"].extend(data.get("_scores", []))
                best_rank = data.get("best_rank")
                if best_rank is not None and (
                    overall["best_rank"] is None or best_rank < overall["best_rank"]
                ):
                    overall["best_rank"] = best_rank
                    overall["top_preset"] = data.get("top_preset")
                worst_rank = data.get("worst_rank")
                if worst_rank is not None and (
                    overall["worst_rank"] is None or worst_rank > overall["worst_rank"]
                ):
                    overall["worst_rank"] = worst_rank
                    overall["_worst_preset"] = data.get("_worst_preset")
                overall["_rank_sum"] += float(data.get("_rank_sum", 0.0))
                overall["_rank_count"] += int(data.get("_rank_count", 0))
                overall["_ranks"].extend(data.get("_ranks", []))
                overall["_rank_sq_sum"] += float(data.get("_rank_sq_sum", 0.0))
                overall["_score_rank_pairs"].extend(data.get("_score_rank_pairs", []))

            summary["overall"] = overall

        result: Dict[str, Dict[str, Any]] = {}
        for label, data in summary.items():
            score_count = data.pop("_score_count")
            score_sum = data.pop("_score_sum")
            score_sq_sum = data.pop("_score_sq_sum")
            scores = data.pop("_scores")
            score_rank_pairs = data.pop("_score_rank_pairs")
            avg_score = score_sum / score_count if score_count else None
            std_dev: float | None
            variance: float | None
            if score_count == 0:
                std_dev = None
                variance = None
            elif score_count == 1:
                std_dev = 0.0
                variance = 0.0
            else:
                avg_value = avg_score if isinstance(avg_score, (int, float)) else 0.0
                variance = max(0.0, (score_sq_sum / score_count) - (avg_value ** 2))
                std_dev = math.sqrt(variance)
            if (
                std_dev is not None
                and isinstance(avg_score, (int, float))
                and float(avg_score) != 0.0
            ):
                score_cv: float | None = std_dev / abs(float(avg_score))
            elif std_dev == 0.0 and isinstance(avg_score, (int, float)) and float(avg_score) == 0.0:
                score_cv = 0.0
            else:
                score_cv = None
            score_median = statistics.median(scores) if scores else None
            score_p10 = _percentile(scores, 0.10)
            score_p25 = _percentile(scores, 0.25)
            score_p75 = _percentile(scores, 0.75)
            score_p90 = _percentile(scores, 0.90)
            score_skewness, score_kurtosis = _shape(scores)
            score_jarque_bera = _jarque_bera(score_count, score_skewness, score_kurtosis)
            if score_count and isinstance(avg_score, (int, float)):
                score_mad = sum(abs(float(s) - float(avg_score)) for s in scores) / score_count
            else:
                score_mad = None
            rank_count = data.pop("_rank_count")
            rank_sum = data.pop("_rank_sum")
            ranks = data.pop("_ranks")
            rank_sq_sum = data.pop("_rank_sq_sum")
            avg_rank = rank_sum / rank_count if rank_count else None
            rank_median = statistics.median(ranks) if ranks else None
            rank_p10 = _percentile(ranks, 0.10)
            rank_p25 = _percentile(ranks, 0.25)
            rank_p75 = _percentile(ranks, 0.75)
            rank_p90 = _percentile(ranks, 0.90)
            rank_skewness, rank_kurtosis = _shape(ranks)
            rank_jarque_bera = _jarque_bera(rank_count, rank_skewness, rank_kurtosis)
            if rank_count == 0:
                rank_stddev: float | None = None
                rank_variance: float | None = None
            elif rank_count == 1:
                rank_stddev = 0.0
                rank_variance = 0.0
            else:
                avg_rank_value = avg_rank if isinstance(avg_rank, (int, float)) else 0.0
                rank_variance = max(0.0, (rank_sq_sum / rank_count) - (avg_rank_value ** 2))
                rank_stddev = math.sqrt(rank_variance)
            if (
                rank_stddev is not None
                and isinstance(avg_rank, (int, float))
                and float(avg_rank) != 0.0
            ):
                rank_cv: float | None = rank_stddev / abs(float(avg_rank))
            elif rank_stddev == 0.0 and isinstance(avg_rank, (int, float)) and float(avg_rank) == 0.0:
                rank_cv = 0.0
            else:
                rank_cv = None
            if rank_count and isinstance(avg_rank, (int, float)):
                rank_mad = sum(abs(float(r) - float(avg_rank)) for r in ranks) / rank_count
            else:
                rank_mad = None
            bottom_preset = data.pop("_worst_preset")
            count = data["count"]
            score_min = data.get("_score_min")
            score_max = data.get("_score_max")
            score_range = (
                float(score_max) - float(score_min)
                if isinstance(score_max, (int, float)) and isinstance(score_min, (int, float))
                else None
            )
            best_rank_value = data["best_rank"]
            worst_rank_value = data["worst_rank"]
            rank_range = (
                float(worst_rank_value) - float(best_rank_value)
                if isinstance(best_rank_value, (int, float)) and isinstance(worst_rank_value, (int, float))
                else None
            )
            score_rank_covariance, score_rank_pearson = _pearson_covariance(score_rank_pairs)
            score_rank_spearman = _spearman(score_rank_pairs)
            if (
                score_rank_covariance is not None
                and isinstance(variance, (int, float))
            ):
                if variance > 0:
                    score_rank_slope: float | None = score_rank_covariance / variance
                else:
                    score_rank_slope = 0.0
            else:
                score_rank_slope = None
            if (
                score_rank_slope is not None
                and isinstance(avg_rank, (int, float))
                and isinstance(avg_score, (int, float))
            ):
                score_rank_intercept: float | None = float(avg_rank) - (
                    score_rank_slope * float(avg_score)
                )
            else:
                score_rank_intercept = None
            score_rank_r_squared: float | None
            if isinstance(score_rank_pearson, (int, float)):
                score_rank_r_squared = float(score_rank_pearson) ** 2
            else:
                score_rank_r_squared = None
            if (
                score_rank_slope is not None
                and score_rank_intercept is not None
                and score_rank_pairs
            ):
                residuals = [
                    float(actual_rank)
                    - (
                        (float(score) * float(score_rank_slope))
                        + float(score_rank_intercept)
                    )
                    for score, actual_rank in score_rank_pairs
                ]
                residual_count = len(residuals)
                residual_bias = sum(residuals) / residual_count if residual_count else None
                residual_mae = (
                    sum(abs(residual) for residual in residuals) / residual_count
                    if residual_count
                    else None
                )
                residual_mse = (
                    sum(residual * residual for residual in residuals) / residual_count
                    if residual_count
                    else None
                )
                residual_rmse = (
                    math.sqrt(residual_mse) if isinstance(residual_mse, (int, float)) else None
                )
                if residual_count == 0:
                    residual_variance = None
                    residual_std_error = None
                elif residual_count == 1:
                    residual_variance = 0.0
                    residual_std_error = 0.0
                else:
                    mean_residual = residual_bias if isinstance(residual_bias, (int, float)) else 0.0
                    residual_variance = (
                        sum((residual - mean_residual) ** 2 for residual in residuals)
                        / (residual_count - 1)
                    )
                    residual_std_error = math.sqrt(residual_variance)
            else:
                residual_bias = None
                residual_mae = None
                residual_mse = None
                residual_rmse = None
                residual_variance = None
                residual_std_error = None
            result[label] = {
                "count": count,
                "presets_with_score": score_count,
                "avg_risk_score": avg_score,
                "min_risk_score": score_min,
                "max_risk_score": score_max,
                "best_rank": data["best_rank"],
                "worst_rank": data["worst_rank"],
                "top_preset": data["top_preset"],
                "bottom_preset": bottom_preset,
                "presets_with_rank": rank_count,
                "avg_rank": avg_rank,
                "risk_score_median": score_median,
                "rank_median": rank_median,
                "risk_score_stddev": std_dev,
                "risk_score_variance": variance,
                "risk_score_cv": score_cv,
                "risk_score_p10": score_p10,
                "risk_score_p25": score_p25,
                "risk_score_p75": score_p75,
                "risk_score_p90": score_p90,
                "risk_score_iqr": (score_p75 - score_p25) if score_p25 is not None and score_p75 is not None else None,
                "risk_score_mad": score_mad,
                "risk_score_range": score_range,
                "risk_score_skewness": score_skewness,
                "risk_score_kurtosis": score_kurtosis,
                "risk_score_jarque_bera": score_jarque_bera,
                "count_share": (data["count"] / total_presets) if total_presets else 0.0,
                "score_coverage": (score_count / count) if count else 0.0,
                "rank_coverage": (rank_count / count) if count else 0.0,
                "rank_p10": rank_p10,
                "rank_p25": rank_p25,
                "rank_p75": rank_p75,
                "rank_p90": rank_p90,
                "rank_iqr": (rank_p75 - rank_p25) if rank_p25 is not None and rank_p75 is not None else None,
                "rank_stddev": rank_stddev,
                "rank_variance": rank_variance,
                "rank_cv": rank_cv,
                "rank_mad": rank_mad,
                "rank_range": rank_range,
                "rank_skewness": rank_skewness,
                "rank_kurtosis": rank_kurtosis,
                "rank_jarque_bera": rank_jarque_bera,
                "score_rank_count": len(score_rank_pairs),
                "score_rank_covariance": score_rank_covariance,
                "score_rank_pearson": score_rank_pearson,
                "score_rank_spearman": score_rank_spearman,
                "score_rank_regression_slope": score_rank_slope,
                "score_rank_regression_intercept": score_rank_intercept,
                "score_rank_r_squared": score_rank_r_squared,
                "score_rank_regression_bias": residual_bias,
                "score_rank_regression_mae": residual_mae,
                "score_rank_regression_mse": residual_mse,
                "score_rank_regression_rmse": residual_rmse,
                "score_rank_regression_residual_variance": residual_variance,
                "score_rank_regression_residual_std_error": residual_std_error,
            }
        return result

    def apply_marketplace_preset(
        self,
        preset_id: str,
        *,
        actor: str | None = None,
        user_confirmed: bool = False,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        preset = self.get_marketplace_preset(preset_id)
        config_mapping: Mapping[str, Any]
        if isinstance(getattr(preset, "config", None), Mapping):
            config_mapping = preset.config  # type: ignore[assignment]
        elif isinstance(preset, Mapping):
            config_mapping = preset
        else:
            raise ConfigError("Preset does not expose a valid config mapping")

        merged = self._merge_config(self._current_config, config_mapping)
        merged["meta"] = self._update_meta(merged.get("meta", {}), preset_id, actor, note, user_confirmed)

        strategy_section = merged.get("strategy", {})
        if not isinstance(strategy_section, dict):
            strategy_section = {}
        if str(strategy_section.get("mode", "demo")).strip().lower() == "live":
            if not user_confirmed:
                raise ValidationError("Aktywacja trybu LIVE wymaga potwierdzenia użytkownika")
            strategy_section.setdefault("backtest_passed_at", time.time())

        self._ensure_live_mode_requirements(strategy_section)
        merged["strategy"] = strategy_section

        self._current_config = merged
        self._record_preset_snapshot(
            preset_id=preset_id,
            config=merged,
            actor=actor or "system",
            note=note,
            source="marketplace",
        )
        return merged

    @staticmethod
    def _ensure_live_mode_requirements(strategy_section: Dict[str, Any]) -> None:
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

    async def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            self._current_config = deepcopy(self._default_config())
            return self._current_config

        raw = self._read_config_file()
        if not isinstance(raw, dict):
            raise ConfigError("Plik konfiguracyjny musi być słownikiem")

        config = self._normalise_loaded_config(raw)
        strategy = config.get("strategy", {})
        if isinstance(strategy, dict):
            self._ensure_live_mode_requirements(strategy)
        telemetry_raw = config.get("telemetry")
        if isinstance(telemetry_raw, dict):
            self._telemetry = TelemetryConfig(**telemetry_raw).validate()
            config["telemetry"] = self._telemetry

        self._current_config = config
        return config

    async def save_config(
        self,
        config: Mapping[str, Any],
        *,
        actor: str,
        preset_id: str,
        note: Optional[str] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        prepared = self._prepare_config_for_storage(config)
        strategy_section = prepared.get("strategy", {})
        if isinstance(strategy_section, dict):
            self._ensure_live_mode_requirements(strategy_section)

        self._write_config_file(prepared)
        self._current_config = prepared
        self._record_preset_snapshot(
            preset_id=preset_id,
            config=prepared,
            actor=actor,
            note=note,
            source=source,
        )
        return prepared

    def _read_config_file(self) -> Dict[str, Any]:
        if self.encryption_key:
            payload = self.config_path.read_bytes()
            return decrypt_config(payload, self.encryption_key)

        text = self.config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ConfigError("Plik konfiguracyjny musi być słownikiem")
        return data

    def _write_config_file(self, config: Mapping[str, Any]) -> None:
        if not self._history_enabled:
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if self.encryption_key:
            token = encrypt_config(dict(config), self.encryption_key)
            self.config_path.write_bytes(token)
            return
        text = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
        self.config_path.write_text(text, encoding="utf-8")

    def _normalise_loaded_config(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        config = deepcopy(dict(payload))
        if "strategy" not in config or not isinstance(config.get("strategy"), dict):
            config["strategy"] = deepcopy(self._default_config()["strategy"])
        if "trade" not in config or not isinstance(config.get("trade"), dict):
            config["trade"] = deepcopy(self._default_config()["trade"])
        config.setdefault("meta", {})
        return config

    def _merge_config(
        self,
        base_config: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = deepcopy(dict(base_config))
        for section, payload in overrides.items():
            result[section] = deepcopy(payload)
        return result

    def _update_meta(
        self,
        meta: Mapping[str, Any],
        preset_id: str,
        actor: str | None,
        note: str | None,
        confirmed: bool,
    ) -> Dict[str, Any]:
        updated = dict(meta)
        updated["last_preset"] = preset_id
        updated["actor"] = actor or "system"
        if confirmed:
            updated["user_confirmed"] = True
        if note:
            updated["note"] = note
        return updated

    def _ensure_history_dir(self) -> None:
        if not self._history_enabled:
            return
        self._versions_dir.mkdir(parents=True, exist_ok=True)

    def _record_preset_snapshot(
        self,
        *,
        preset_id: str,
        config: Mapping[str, Any],
        actor: str,
        note: str | None,
        source: str,
    ) -> None:
        if not preset_id or not self._history_enabled:
            return
        self._ensure_history_dir()
        preset_dir = self._versions_dir / preset_id
        preset_dir.mkdir(parents=True, exist_ok=True)
        version_id = uuid.uuid4().hex
        payload = {
            "version_id": version_id,
            "preset_id": preset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "note": note,
            "source": source,
            "config": self._sanitize_config_for_history(config),
        }
        snapshot_path = preset_dir / f"{version_id}.json"
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._history_cache.pop(preset_id, None)

    def get_preset_history(self, preset_id: str) -> List[Dict[str, Any]]:
        if not preset_id or not self._history_enabled:
            return []
        if preset_id in self._history_cache:
            return list(self._history_cache[preset_id])

        preset_dir = self._versions_dir / preset_id
        history: List[Dict[str, Any]] = []
        if preset_dir.exists():
            for path in preset_dir.glob("*.json"):
                entry = self._read_history_entry(path)
                if entry:
                    history.append(entry)
        history.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        self._history_cache[preset_id] = history
        return list(history)

    def _read_history_entry(self, path: Path) -> Dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if "version_id" not in payload:
            payload["version_id"] = path.stem
        return payload

    def rollback_preset(
        self,
        preset_id: str,
        version_id: str,
        *,
        actor: str | None = None,
        note: str | None = None,
    ) -> Dict[str, Any]:
        if not self._history_enabled:
            raise ConfigError("Historia presetów jest niedostępna bez ścieżki konfiguracji")
        snapshot_path = self._versions_dir / preset_id / f"{version_id}.json"
        if not snapshot_path.exists():
            raise ConfigError(f"Brak wersji {version_id!r} dla presetu {preset_id!r}")
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ConfigError("Nieprawidłowa struktura snapshotu konfiguracji")
        self._current_config = deepcopy(config)
        strategy_section = self._current_config.get("strategy", {})
        if isinstance(strategy_section, dict):
            self._ensure_live_mode_requirements(strategy_section)
        self._write_config_file(self._current_config)
        self._record_preset_snapshot(
            preset_id=preset_id,
            config=self._current_config,
            actor=actor or "system",
            note=note or f"rollback:{version_id}",
            source="rollback",
        )
        return self._current_config

    @property
    def api_key_manager(self) -> APIKeyManager:
        if not self._history_enabled:
            raise ConfigError("Przechowywanie kluczy API wymaga ścieżki konfiguracji")
        if self._api_key_manager is None:
            manager_cls = APIKeyManager
            if manager_cls is Any:  # pragma: no cover - fallback na środowiska testowe
                from KryptoLowca.security.api_key_manager import (  # type: ignore
                    APIKeyManager as RealAPIKeyManager,
                )

                manager_cls = RealAPIKeyManager
            storage_path = self.config_path.parent / "api_keys_store.json"
            self._api_key_manager = manager_cls(
                storage_path=storage_path,
                encryptor=self._encrypt_secret_payload,
                decryptor=self._decrypt_secret_payload,
            )
        return self._api_key_manager

    def _encrypt_secret_payload(self, namespace: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.encryption_key:
            raise ConfigError("Brak klucza szyfrującego – nie można zapisać danych wrażliwych")
        token = encrypt_config({"namespace": namespace, "data": payload}, self.encryption_key)
        masked = {}
        for key, value in payload.items():
            if key in {"api_key", "api_secret", "passphrase"}:
                masked[key] = "***"
            else:
                masked[key] = value
        masked["namespace"] = namespace
        masked["token"] = base64.b64encode(token).decode("ascii")
        return masked

    def _decrypt_secret_payload(self, namespace: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.encryption_key:
            raise ConfigError("Brak klucza szyfrującego – nie można odczytać danych wrażliwych")
        token_b64 = payload.get("token")
        if not token_b64:
            return {}
        token = base64.b64decode(token_b64)
        data = decrypt_config(token, self.encryption_key)
        decrypted = data.get("data", {})
        if isinstance(decrypted, dict):
            return decrypted
        return {}

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "strategy": {
                "preset": "DEFAULT",
                "mode": "demo",
                "max_leverage": 1.0,
                "default_tp": 0.02,
                "default_sl": 0.01,
            },
            "trade": {
                "max_open_positions": 3,
                "risk_per_trade": 0.02,
            },
            "exchange": {},
            "meta": {},
        }

    def _prepare_config_for_storage(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in dict(config).items():
            if hasattr(value, "__dataclass_fields__"):
                prepared[key] = asdict(value)
            else:
                prepared[key] = deepcopy(value)
        return prepared

    def _sanitize_config_for_history(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        def mask(value: Any) -> Any:
            if isinstance(value, Mapping):
                masked: Dict[str, Any] = {}
                for key, inner in value.items():
                    if any(token in key.lower() for token in ("key", "secret", "token", "passphrase")):
                        masked[key] = "***"
                    else:
                        masked[key] = mask(inner)
                return masked
            if isinstance(value, list):
                return [mask(item) for item in value]
            return value

        return mask(dict(config))

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
        if "strategy" in out and hasattr(out["strategy"], "__dataclass_fields__"):
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

