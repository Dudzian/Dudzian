"""Ustawienia menedżera ryzyka wykorzystywane przez runtime."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Mapping

_CANON = "bot_core.risk.settings"
if __name__ != _CANON:
    sys.modules[_CANON] = sys.modules[__name__]


@dataclass(frozen=True, slots=True)
class RiskManagerSettings:
    """Zestaw wartości konfiguracyjnych dla adaptera menedżera ryzyka."""

    max_risk_per_trade: float
    max_daily_loss_pct: float
    max_portfolio_risk: float
    max_positions: int
    emergency_stop_drawdown: float
    confidence_level: float | None = None
    target_volatility: float | None = None
    profile_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializuje ustawienia do słownika kompatybilnego z adapterem GUI."""

        payload: dict[str, Any] = {
            "max_risk_per_trade": float(self.max_risk_per_trade),
            "max_portfolio_risk": float(self.max_portfolio_risk),
            "max_positions": int(self.max_positions),
            "emergency_stop_drawdown": float(self.emergency_stop_drawdown),
        }
        if self.confidence_level is not None:
            payload["confidence_level"] = float(self.confidence_level)
        return payload

    def risk_service_kwargs(self) -> dict[str, float | int]:
        """Zwraca parametry oczekiwane przez ``RiskService`` AutoTradera."""

        return {
            "max_position_notional_pct": float(self.max_risk_per_trade),
            "max_daily_loss_pct": float(self.max_daily_loss_pct),
            "max_portfolio_risk_pct": float(self.max_portfolio_risk),
            "emergency_stop_drawdown_pct": float(self.emergency_stop_drawdown),
            "max_positions": int(self.max_positions),
        }


_DEFAULT_RISK_SETTINGS = RiskManagerSettings(
    max_risk_per_trade=0.02,
    max_daily_loss_pct=0.10,
    max_portfolio_risk=0.10,
    max_positions=10,
    emergency_stop_drawdown=0.15,
)


def derive_risk_manager_settings(
    profile: object | None,
    *,
    profile_name: str | None = None,
    defaults: RiskManagerSettings | Mapping[str, Any] | None = None,
) -> RiskManagerSettings:
    """Buduje ustawienia menedżera ryzyka na podstawie profilu runtime."""

    base = {
        "max_risk_per_trade": _DEFAULT_RISK_SETTINGS.max_risk_per_trade,
        "max_daily_loss_pct": _DEFAULT_RISK_SETTINGS.max_daily_loss_pct,
        "max_portfolio_risk": _DEFAULT_RISK_SETTINGS.max_portfolio_risk,
        "max_positions": _DEFAULT_RISK_SETTINGS.max_positions,
        "emergency_stop_drawdown": _DEFAULT_RISK_SETTINGS.emergency_stop_drawdown,
        "confidence_level": _DEFAULT_RISK_SETTINGS.confidence_level,
        "target_volatility": _DEFAULT_RISK_SETTINGS.target_volatility,
    }

    if isinstance(defaults, RiskManagerSettings):
        base.update(
            max_risk_per_trade=defaults.max_risk_per_trade,
            max_daily_loss_pct=defaults.max_daily_loss_pct,
            max_portfolio_risk=defaults.max_portfolio_risk,
            max_positions=defaults.max_positions,
            emergency_stop_drawdown=defaults.emergency_stop_drawdown,
        )
        base["confidence_level"] = defaults.confidence_level
        base["target_volatility"] = defaults.target_volatility
    elif isinstance(defaults, Mapping):
        for key in base:
            if key in defaults and defaults[key] is not None:
                value = defaults[key]
                if key == "max_positions":
                    try:
                        base[key] = int(value)  # type: ignore[assignment]
                    except Exception:
                        continue
                else:
                    try:
                        base[key] = float(value)  # type: ignore[assignment]
                    except Exception:
                        continue

    def _read_value(name: str) -> Any:
        if profile is None:
            return None
        if hasattr(profile, name):
            return getattr(profile, name)
        if isinstance(profile, Mapping):
            return profile.get(name)
        return None

    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            coerced = float(value)
        except Exception:
            return fallback
        if not coerced and coerced != 0.0:
            return fallback
        return coerced

    per_trade = _coerce_float(_read_value("max_position_pct"), float(base["max_risk_per_trade"]))
    if per_trade <= 0:
        per_trade = float(base["max_risk_per_trade"])
    max_risk_per_trade = float(max(0.0, min(1.0, per_trade)))
    base["max_risk_per_trade"] = max_risk_per_trade

    daily_loss = _coerce_float(_read_value("max_daily_loss_pct"), base["max_daily_loss_pct"])
    if daily_loss <= 0:
        daily_loss = base["max_daily_loss_pct"]
    base["max_daily_loss_pct"] = max(0.0, min(1.0, daily_loss))

    portfolio_limit = _coerce_float(
        _read_value("max_portfolio_risk"),
        base["max_portfolio_risk"] if base["max_portfolio_risk"] else daily_loss,
    )
    if portfolio_limit <= 0:
        portfolio_limit = base["max_portfolio_risk"] if base["max_portfolio_risk"] else daily_loss
    portfolio_limit = max(max_risk_per_trade + 1e-4, portfolio_limit)
    base["max_portfolio_risk"] = min(1.0, portfolio_limit)

    positions = _read_value("max_open_positions")
    try:
        positions_int = int(positions) if positions is not None else base["max_positions"]
    except Exception:
        positions_int = base["max_positions"]
    if positions_int > 0:
        base["max_positions"] = positions_int

    drawdown = _coerce_float(_read_value("hard_drawdown_pct"), base["emergency_stop_drawdown"])
    if drawdown > 0:
        base["emergency_stop_drawdown"] = min(1.0, max(drawdown, max_risk_per_trade))

    target_vol = _coerce_float(_read_value("target_volatility"), base["target_volatility"] or 0.0)
    if target_vol > 0:
        base["target_volatility"] = min(1.0, target_vol)
        base["confidence_level"] = max(0.5, min(0.99, 1.0 - min(target_vol, 0.5) / 2))

    return RiskManagerSettings(
        max_risk_per_trade=float(base["max_risk_per_trade"]),
        max_daily_loss_pct=float(base["max_daily_loss_pct"]),
        max_portfolio_risk=float(base["max_portfolio_risk"]),
        max_positions=int(base["max_positions"]),
        emergency_stop_drawdown=float(base["emergency_stop_drawdown"]),
        confidence_level=None
        if base["confidence_level"] is None
        else float(base["confidence_level"]),
        target_volatility=None
        if base["target_volatility"] is None
        else float(base["target_volatility"]),
        profile_name=profile_name,
    )


__all__ = [
    "RiskManagerSettings",
    "derive_risk_manager_settings",
]
