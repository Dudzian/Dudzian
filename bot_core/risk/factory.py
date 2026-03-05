"""Funkcje pomocnicze do budowy profili ryzyka z konfiguracji."""

from __future__ import annotations

from bot_core.config.models import RiskProfileConfig
from bot_core.risk.base import RiskProfile
from bot_core.risk.profiles.loader import RiskProfileLoader


def build_risk_profile_from_config(config: RiskProfileConfig) -> RiskProfile:
    """Tworzy instancję profilu ryzyka na podstawie konfiguracji."""

    return RiskProfileLoader().build_from_config(config)


__all__ = ["build_risk_profile_from_config"]
