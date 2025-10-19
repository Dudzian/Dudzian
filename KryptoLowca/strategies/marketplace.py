"""Ładowanie presetów strategii wraz z metadanymi oceny."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

__all__ = [
    "StrategyPreset",
    "StrategyEvaluation",
    "StrategyBacktestMetrics",
    "load_marketplace_presets",
    "load_preset",
    "load_marketplace_index",
]


class ConfigError(RuntimeError):
    """Ogólny błąd operacji na presetach marketplace."""


_DEFAULT_MARKETPLACE_DIR = Path(__file__).with_name("marketplace")


# ---------------------------------------------------------------------------
# Funkcje pomocnicze do normalizacji pól JSON
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Modele danych marketplace
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrategyBacktestMetrics:
    period: str
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int
    profit_factor: float | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StrategyBacktestMetrics":
        period = _ensure_optional_str(payload.get("period")) or ""
        return cls(
            period=period,
            cagr=float(payload.get("cagr", 0.0) or 0.0),
            sharpe=float(payload.get("sharpe", 0.0) or 0.0),
            max_drawdown=float(payload.get("max_drawdown", 0.0) or 0.0),
            win_rate=float(payload.get("win_rate", 0.0) or 0.0),
            trades=int(payload.get("trades", 0) or 0),
            profit_factor=(
                float(payload.get("profit_factor"))
                if payload.get("profit_factor") is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "period": self.period,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "trades": self.trades,
        }
        if self.profit_factor is not None:
            data["profit_factor"] = self.profit_factor
        return data


@dataclass(slots=True)
class StrategyEvaluation:
    rank: int | None = None
    risk_label: str | None = None
    risk_score: float | None = None
    highlights: List[str] = field(default_factory=list)
    backtest: StrategyBacktestMetrics | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StrategyEvaluation":
        rank_value = payload.get("rank")
        rank = int(rank_value) if isinstance(rank_value, (int, float)) else None
        risk_label = _ensure_optional_str(payload.get("risk_label"))
        risk_score_val = payload.get("risk_score")
        risk_score = (
            float(risk_score_val)
            if isinstance(risk_score_val, (int, float, str))
            and risk_score_val not in (None, "")
            else None
        )
        highlights = _ensure_str_list(payload.get("highlights"))
        backtest_payload = payload.get("backtest")
        backtest = (
            StrategyBacktestMetrics.from_payload(backtest_payload)
            if isinstance(backtest_payload, Mapping)
            else None
        )
        return cls(
            rank=rank,
            risk_label=risk_label,
            risk_score=risk_score,
            highlights=highlights,
            backtest=backtest,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "rank": self.rank,
            "risk_label": self.risk_label,
            "risk_score": self.risk_score,
            "highlights": list(self.highlights),
        }
        if self.backtest is not None:
            data["backtest"] = self.backtest.to_dict()
        return data


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
    evaluation: StrategyEvaluation | None = None

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
        evaluation_payload = payload.get("evaluation")
        evaluation = (
            StrategyEvaluation.from_payload(evaluation_payload)
            if isinstance(evaluation_payload, Mapping)
            else None
        )

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
            evaluation=evaluation,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "preset_id": self.preset_id,
            "name": self.name,
            "description": self.description,
            "config": dict(self.config),
            "risk_level": self.risk_level,
            "recommended_min_balance": self.recommended_min_balance,
            "timeframe": self.timeframe,
            "exchanges": list(self.exchanges),
            "tags": list(self.tags),
            "version": self.version,
            "compatibility": dict(self.compatibility),
            "compliance": dict(self.compliance),
        }
        if self.last_updated is not None:
            data["last_updated"] = self.last_updated.isoformat()
        if self.evaluation is not None:
            data["evaluation"] = self.evaluation.to_dict()
        return data

    def effective_risk_label(self) -> str | None:
        if self.evaluation and self.evaluation.risk_label:
            return self.evaluation.risk_label
        return self.risk_level

    def evaluation_rank(self) -> int | None:
        if self.evaluation:
            return self.evaluation.rank
        return None


# ---------------------------------------------------------------------------
# API do ładowania presetów z dysku
# ---------------------------------------------------------------------------


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


def load_marketplace_index(
    *, base_path: str | Path | None = None
) -> Dict[str, StrategyPreset]:
    """Buduje indeks presetów według identyfikatora.

    Funkcja jest wygodna w warstwach zarządzających (ConfigManager/UI),
    ponieważ unika wielokrotnego parsowania tych samych plików JSON.
    """

    presets = load_marketplace_presets(base_path=base_path)
    index: Dict[str, StrategyPreset] = {}
    for preset in presets:
        index[preset.preset_id] = preset
    return index
