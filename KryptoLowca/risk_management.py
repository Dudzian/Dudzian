"""Kompatybilna warstwa zgodności dla historycznych importów.

Moduł ``KryptoLowca.risk_management`` był dawniej pełnoprawnym systemem
zarządzania ryzykiem. W nowej architekturze zastąpiono go silnikiem
``bot_core.risk.engine.ThresholdRiskEngine``. Plik pozostaje jedynie jako
warstwa zgodności – eksportuje klasy z ``bot_core.risk`` i emituje ostrzeżenie
o przestarzałym API podczas korzystania z ``create_risk_manager``.
"""

from __future__ import annotations

import warnings
from typing import Mapping

from bot_core.config.models import RiskProfileConfig
from bot_core.risk import *  # noqa: F401,F403 - kompatybilność publicznego API
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.factory import build_risk_profile_from_config


def create_risk_manager(config: Mapping[str, object] | RiskProfileConfig) -> ThresholdRiskEngine:
    warnings.warn(
        "KryptoLowca.risk_management.create_risk_manager jest przestarzałe – "
        "użyj bezpośrednio bot_core.risk.engine.ThresholdRiskEngine.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(config, RiskProfileConfig):
        profile_config = config
    elif isinstance(config, Mapping):
        profile_config = RiskProfileConfig(
            name=str(config.get("risk_profile_name") or config.get("name") or "manual"),
            max_daily_loss_pct=float(config.get("max_daily_loss_pct", 0.1) or 0.1),
            max_position_pct=float(config.get("max_position_pct", config.get("max_risk_per_trade", 0.02)) or 0.02),
            target_volatility=float(config.get("target_volatility", 0.0) or 0.0),
            max_leverage=float(config.get("max_leverage", 3.0) or 3.0),
            stop_loss_atr_multiple=float(config.get("stop_loss_atr_multiple", 2.0) or 2.0),
            max_open_positions=int(config.get("max_open_positions", config.get("max_positions", 5)) or 5),
            hard_drawdown_pct=float(config.get("hard_drawdown_pct", config.get("max_drawdown_pct", 0.2)) or 0.2),
        )
    else:
        raise TypeError("Oczekiwano Mapping lub RiskProfileConfig")

    profile = build_risk_profile_from_config(profile_config)
    engine = ThresholdRiskEngine()
    engine.register_profile(profile)
    return engine


__all__ = [name for name in globals().keys() if not name.startswith("_")]

