"""Funkcje pomocnicze do budowy profili ryzyka z konfiguracji."""
from __future__ import annotations

from typing import Mapping, Type

from bot_core.config.models import RiskProfileConfig
from bot_core.risk.base import RiskProfile
from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile
from bot_core.risk.profiles.manual import ManualProfile


_PROFILE_CLASS_BY_NAME: Mapping[str, Type[RiskProfile]] = {
    "conservative": ConservativeProfile,
    "balanced": BalancedProfile,
    "aggressive": AggressiveProfile,
}


def build_risk_profile_from_config(config: RiskProfileConfig) -> RiskProfile:
    """Tworzy instancję profilu ryzyka na podstawie konfiguracji."""

    profile_key = config.name.lower()
    profile_cls = _PROFILE_CLASS_BY_NAME.get(profile_key)
    if profile_cls is not None:
        return profile_cls()

    return ManualProfile(
        name=config.name,
        max_positions=config.max_open_positions,
        max_leverage=config.max_leverage,
        drawdown_limit=config.hard_drawdown_pct,
        daily_loss_limit=config.max_daily_loss_pct,
        max_position_pct=config.max_position_pct,
        target_volatility=config.target_volatility,
        stop_loss_atr_multiple=config.stop_loss_atr_multiple,
    )


__all__ = ["build_risk_profile_from_config"]
