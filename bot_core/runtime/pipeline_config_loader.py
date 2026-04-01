"""Ładowanie i walidacja konfiguracji wykorzystywanej przez runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, MultiStrategySchedulerConfig
from bot_core.portfolio import CopyTradingFollowerConfig, PortfolioBinding


@dataclass(frozen=True, slots=True)
class ResolvedMultiStrategyConfig:
    """Wynik walidacji i wyboru scheduler-a multi-strategy."""

    core_config: CoreConfig
    scheduler_name: str
    scheduler_config: MultiStrategySchedulerConfig


class PipelineConfigLoader:
    """Komponent odpowiedzialny za loading/normalizację/walidację konfiguracji pipeline."""

    def load_core_config(self, config_path: str | Path) -> CoreConfig:
        return load_core_config(config_path)

    def resolve_multi_strategy_scheduler(
        self,
        *,
        core_config: CoreConfig,
        scheduler_name: str | None,
    ) -> ResolvedMultiStrategyConfig:
        scheduler_configs = getattr(core_config, "multi_strategy_schedulers", {})
        if not scheduler_configs:
            raise ValueError("Brak zdefiniowanych schedulerów multi-strategy w konfiguracji")

        resolved_scheduler_name = scheduler_name or next(iter(scheduler_configs))
        scheduler_cfg = scheduler_configs.get(resolved_scheduler_name)
        if scheduler_cfg is None:
            raise KeyError(f"Nie znaleziono scheduler-a {resolved_scheduler_name}")

        return ResolvedMultiStrategyConfig(
            core_config=core_config,
            scheduler_name=resolved_scheduler_name,
            scheduler_config=scheduler_cfg,
        )

    def resolve_multi_portfolio_entries(self, definition: object) -> Sequence[Mapping[str, object]]:
        if definition is None:
            return ()
        if isinstance(definition, Mapping):
            if "portfolios" in definition:
                payload = definition["portfolios"]
                if isinstance(payload, Mapping):
                    return [dict(value) for value in payload.values() if isinstance(value, Mapping)]
                if isinstance(payload, Sequence):
                    return [dict(entry) for entry in payload if isinstance(entry, Mapping)]
            return [dict(definition)]
        if isinstance(definition, Sequence):
            entries: list[Mapping[str, object]] = []
            for item in definition:
                if isinstance(item, Mapping):
                    entries.append(dict(item))
            return entries
        if hasattr(definition, "portfolios"):
            payload = getattr(definition, "portfolios")
            return self.resolve_multi_portfolio_entries(payload)
        raise TypeError("Unsupported portfolio definition structure")

    def build_portfolio_binding(self, entry: Mapping[str, object]) -> PortfolioBinding:
        portfolio_id = str(entry.get("portfolio_id") or entry.get("id") or "").strip()
        if not portfolio_id:
            raise ValueError("Portfolio binding must define 'portfolio_id'")
        primary = str(entry.get("primary_preset") or entry.get("preset") or "").strip()
        if not primary:
            raise ValueError(f"Portfolio {portfolio_id} missing primary preset")
        fallback = self.normalize_fallbacks(entry.get("fallback_presets") or entry.get("fallback"))
        followers = self.build_follower_configs(entry.get("followers"))
        cooldown_value = entry.get("rebalance_cooldown_seconds")
        if cooldown_value in (None, ""):
            cooldown = timedelta(minutes=5)
        else:
            cooldown = timedelta(seconds=float(cooldown_value))
        return PortfolioBinding(
            portfolio_id=portfolio_id,
            primary_preset=primary,
            fallback_presets=fallback,
            followers=followers,
            rebalance_cooldown=cooldown,
        )

    def build_follower_configs(
        self, raw_followers: object
    ) -> tuple[CopyTradingFollowerConfig, ...]:
        if raw_followers in (None, ""):
            return ()
        if isinstance(raw_followers, Mapping):
            raw_sequence = [raw_followers]
        elif isinstance(raw_followers, Sequence) and not isinstance(raw_followers, (str, bytes)):
            raw_sequence = list(raw_followers)
        else:
            raise TypeError("Followers must be a mapping or a sequence of mappings")

        followers: list[CopyTradingFollowerConfig] = []
        for entry in raw_sequence:
            if not isinstance(entry, Mapping):
                continue
            portfolio_id = str(entry.get("portfolio_id") or entry.get("id") or "").strip()
            if not portfolio_id:
                continue
            scaling = float(entry.get("scaling", 1.0))
            risk_multiplier = float(entry.get("risk_multiplier", entry.get("risk", 1.0)))
            enabled = bool(entry.get("enabled", True))
            allow_partial = bool(entry.get("allow_partial", True))
            max_position_value = entry.get("max_position_value")
            follower = CopyTradingFollowerConfig(
                portfolio_id=portfolio_id,
                scaling=scaling,
                risk_multiplier=risk_multiplier,
                enabled=enabled,
                max_position_value=float(max_position_value)
                if max_position_value not in (None, "")
                else None,
                allow_partial=allow_partial,
            )
            followers.append(follower)
        return tuple(followers)

    def normalize_fallbacks(self, raw_fallbacks: object) -> tuple[str, ...]:
        if raw_fallbacks in (None, ""):
            return ()
        if isinstance(raw_fallbacks, str):
            entries = [raw_fallbacks]
        elif isinstance(raw_fallbacks, Sequence):
            entries = list(raw_fallbacks)
        else:
            raise TypeError("Fallback presets must be a string or sequence of strings")
        normalized: list[str] = []
        for item in entries:
            text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)
        return tuple(normalized)
