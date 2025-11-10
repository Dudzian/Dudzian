"""Adapter konwersji presetów GUI do konfiguracji Core Stage6."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from enum import Enum

import yaml

from bot_core.config.loader import load_core_config
from bot_core.config.models import (
    CoreConfig,
    DecisionOrchestratorThresholds,
    RiskProfileConfig,
    RiskServiceConfig,
    RuntimeEntrypointConfig,
)

__all__ = ["PresetConfigService", "load_preset", "flatten_secret_payload"]


def _normalise_profile_name(name: str | None) -> str:
    candidate = (name or "gui_default").strip()
    return candidate.replace(" ", "_").lower() or "gui_default"


class PresetConfigService:
    """Mapuje presety GUI na konfigurację Stage6 ``CoreConfig``.

    Serwis wczytuje istniejący plik YAML przy pomocy ``load_core_config`` i
    udostępnia metody aktualizacji profili ryzyka na podstawie danych z GUI.
    """

    def __init__(self, core_config_path: str | os.PathLike[str]) -> None:
        self._path = Path(core_config_path).expanduser()
        self._core_config: CoreConfig | None = None
        self._raw_payload: MutableMapping[str, Any] = self._load_raw_payload()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def core_config(self) -> CoreConfig:
        if self._core_config is None:
            self._core_config = load_core_config(self._path)
        return self._core_config

    def reload(self) -> CoreConfig:
        self._core_config = load_core_config(self._path)
        self._raw_payload = self._load_raw_payload()
        return self._core_config

    def ensure_runtime_entrypoint(self, name: str) -> RuntimeEntrypointConfig:
        config = self.core_config
        entrypoints = dict(config.runtime_entrypoints)
        if name in entrypoints:
            return entrypoints[name]
        raise KeyError(f"Runtime entrypoint '{name}' nie jest zdefiniowany w core.yaml")

    def import_gui_preset(
        self,
        preset: Mapping[str, Any],
        *,
        profile_name: str | None = None,
        template_profile: str | None = None,
        runtime_entrypoint: str = "trading_gui",
    ) -> RiskProfileConfig:
        """Aktualizuje konfigurację na bazie presetu GUI.

        Tworzy lub aktualizuje profil ryzyka i przypina go do wskazanego
        entrypointu runtime. Wartości brakujące są uzupełniane z profilu
        wzorcowego.
        """

        config = self.core_config
        risk_profiles = dict(config.risk_profiles)
        if not risk_profiles:
            raise ValueError("Brak zdefiniowanych profili ryzyka w core.yaml")

        resolved_name = _normalise_profile_name(profile_name or preset.get("risk_profile"))
        template_name = template_profile or resolved_name
        base_profile = risk_profiles.get(template_name)
        if base_profile is None:
            # jeśli nie znaleziono profilu wzorcowego, wykorzystaj pierwszy dostępny
            base_profile = next(iter(risk_profiles.values()))

        gui_risk = preset.get("risk")
        if isinstance(gui_risk, Mapping):
            max_daily_loss = _coerce_float(gui_risk.get("max_daily_loss_pct"), base_profile.max_daily_loss_pct)
            max_position = _coerce_float(
                gui_risk.get("risk_per_trade"),
                _coerce_float(preset.get("fraction"), base_profile.max_position_pct),
            )
            target_volatility = _coerce_float(gui_risk.get("portfolio_risk"), base_profile.target_volatility)
            max_leverage = _coerce_float(gui_risk.get("max_leverage"), base_profile.max_leverage)
            stop_loss_atr = _coerce_float(gui_risk.get("stop_loss_atr_multiple"), base_profile.stop_loss_atr_multiple)
            max_open_positions = _coerce_int(gui_risk.get("max_open_positions"), base_profile.max_open_positions)
            hard_drawdown = _coerce_float(gui_risk.get("hard_drawdown_pct"), base_profile.hard_drawdown_pct)
        else:
            max_daily_loss = base_profile.max_daily_loss_pct
            max_position = _coerce_float(preset.get("fraction"), base_profile.max_position_pct)
            target_volatility = base_profile.target_volatility
            max_leverage = base_profile.max_leverage
            stop_loss_atr = base_profile.stop_loss_atr_multiple
            max_open_positions = base_profile.max_open_positions
            hard_drawdown = base_profile.hard_drawdown_pct

        updated_profile = replace(
            base_profile,
            name=resolved_name,
            max_daily_loss_pct=max_daily_loss,
            max_position_pct=max_position,
            target_volatility=target_volatility,
            max_leverage=max_leverage,
            stop_loss_atr_multiple=stop_loss_atr,
            max_open_positions=max_open_positions,
            hard_drawdown_pct=hard_drawdown,
        )

        risk_profiles[resolved_name] = updated_profile
        config.risk_profiles = risk_profiles  # type: ignore[assignment]

        entrypoints = dict(config.runtime_entrypoints)
        entry = entrypoints.get(runtime_entrypoint)
        if entry is None:
            raise KeyError(
                f"Runtime entrypoint '{runtime_entrypoint}' nie istnieje w konfiguracji"
            )
        entry = replace(entry, risk_profile=resolved_name)
        entrypoints[runtime_entrypoint] = entry
        config.runtime_entrypoints = entrypoints  # type: ignore[assignment]

        self._propagate_decision_engine(updated_profile, template_name)
        self._propagate_risk_service(updated_profile)
        self._propagate_portfolio_governors_passthrough(updated_profile, template_name)

        self._core_config = config
        return updated_profile

    def to_dict(self) -> MutableMapping[str, Any]:
        payload = _serialise(asdict(self.core_config))
        payload.pop("source_path", None)
        payload.pop("source_directory", None)
        if isinstance(self._raw_payload, Mapping) and self._raw_payload:
            merged = _merge_mappings(self._raw_payload, payload)
        else:
            merged = payload
        merged.pop("source_path", None)
        merged.pop("source_directory", None)
        return merged

    def save(self, *, destination: str | os.PathLike[str] | None = None, dry_run: bool = False) -> str:
        path = Path(destination) if destination else self._path
        payload = self.to_dict()
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        if dry_run:
            return text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------------
    def _load_raw_payload(self) -> MutableMapping[str, Any]:
        if not self._path.exists():
            return {}
        try:
            content = self._path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            parsed = yaml.safe_load(content) or {}
        except yaml.YAMLError:
            return {}
        if not isinstance(parsed, Mapping):
            return {}
        return deepcopy(parsed)

    # ------------------------------------------------------------------
    def _propagate_decision_engine(
        self, profile: RiskProfileConfig, template_name: str
    ) -> None:
        config = self.core_config
        engine = getattr(config, "decision_engine", None)
        if engine is None:
            return

        overrides = dict(getattr(engine, "profile_overrides", {}) or {})
        base_override = overrides.get(profile.name)
        if base_override is None:
            base_override = overrides.get(template_name)
        if base_override is None and isinstance(engine.orchestrator, DecisionOrchestratorThresholds):
            base_override = engine.orchestrator
        if base_override is None:
            return

        max_daily_loss = _coerce_float(
            profile.max_daily_loss_pct,
            getattr(base_override, "max_daily_loss_pct", profile.max_daily_loss_pct),
        )
        max_position_ratio = _coerce_float(
            profile.max_position_pct,
            getattr(base_override, "max_position_ratio", profile.max_position_pct),
        )
        max_open_positions = _coerce_int(
            profile.max_open_positions,
            getattr(base_override, "max_open_positions", profile.max_open_positions),
        )
        max_drawdown = _coerce_float(
            profile.hard_drawdown_pct,
            getattr(base_override, "max_drawdown_pct", profile.hard_drawdown_pct),
        )

        overrides[profile.name] = replace(
            base_override,
            max_daily_loss_pct=max_daily_loss,
            max_position_ratio=max_position_ratio,
            max_open_positions=max_open_positions,
            max_drawdown_pct=max_drawdown,
        )

        self.core_config.decision_engine = replace(engine, profile_overrides=overrides)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def _propagate_risk_service(self, profile: RiskProfileConfig) -> None:
        config = self.core_config
        risk_service = getattr(config, "risk_service", None)
        if not isinstance(risk_service, RiskServiceConfig):
            return

        existing = list(risk_service.profiles or ())
        if profile.name not in existing:
            existing.append(profile.name)
            config.risk_service = replace(risk_service, profiles=tuple(existing))  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def _propagate_portfolio_governors_passthrough(
        self, profile: RiskProfileConfig, template_name: str
    ) -> None:
        if not isinstance(self._raw_payload, MutableMapping):
            return
        governors = self._raw_payload.get("portfolio_governors")
        if not isinstance(governors, MutableMapping):
            return

        for entry in governors.values():
            if not isinstance(entry, MutableMapping):
                continue
            budgets = entry.get("risk_budgets")
            if not isinstance(budgets, MutableMapping):
                continue

            template_budget: Mapping[str, Any] | None = budgets.get(profile.name)
            if not isinstance(template_budget, Mapping):
                template_budget = budgets.get(template_name)
            if not isinstance(template_budget, Mapping):
                for candidate in budgets.values():
                    if isinstance(candidate, Mapping):
                        template_budget = candidate
                        break
            if not isinstance(template_budget, Mapping):
                continue

            new_budget = dict(template_budget)
            new_budget["name"] = profile.name
            new_budget["max_drawdown_pct"] = _coerce_float(
                profile.hard_drawdown_pct,
                template_budget.get("max_drawdown_pct", profile.hard_drawdown_pct),
            )
            if profile.target_volatility is not None:
                new_budget["max_var_pct"] = _coerce_float(
                    profile.target_volatility,
                    template_budget.get("max_var_pct", profile.target_volatility),
                )
            if profile.max_leverage is not None:
                new_budget["max_leverage"] = _coerce_float(
                    profile.max_leverage,
                    template_budget.get("max_leverage", profile.max_leverage),
                )
            budgets[profile.name] = new_budget


def load_preset(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Wczytuje preset GUI zapisany jako JSON/YAML."""

    data_path = Path(path)
    try:
        content = data_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - informacyjnie
        raise ValueError(f"Nie można odczytać pliku presetu: {exc}") from exc

    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError("Plik presetu ma niepoprawny format JSON/YAML") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Preset musi być słownikiem klucz-wartość")
    return payload


def flatten_secret_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    """Spłaszcza strukturę sekretów do par klucz → tekst."""

    flattened: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            flattened[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            flattened[str(key)] = str(value)
    return flattened


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        if value is None:
            raise TypeError
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _serialise(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _serialise(item)
            if cleaned in (None, {}, [], ()):  # pomijamy puste wartości
                continue
            result[str(key)] = cleaned
        return result
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned = _serialise(item)
            if cleaned in (None, {}, [], ()):  # pragma: no cover - spójność filtracji
                continue
            cleaned_items.append(cleaned)
        return cleaned_items
    return value


def _merge_mappings(base: Mapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    if not isinstance(base, Mapping):
        return dict(updates)
    result: MutableMapping[str, Any] = deepcopy(base)
    for key, value in updates.items():
        current = result.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            result[key] = _merge_mappings(current, value)
        else:
            result[key] = value
    return result
