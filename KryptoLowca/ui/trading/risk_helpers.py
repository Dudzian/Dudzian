"""Pomocnicze funkcje do obsługi profili ryzyka w modułach Trading GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from bot_core.runtime.metadata import RiskManagerSettings

__all__ = [
    "RiskSnapshot",
    "snapshot_from_app",
    "compute_default_notional",
    "format_decimal",
    "format_notional",
    "build_risk_profile_hint",
    "build_risk_limits_summary",
]


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """Wyciągnięte informacje o profilu ryzyka z aplikacji GUI."""

    paper_balance: float
    settings: RiskManagerSettings | None
    profile_name: Optional[str]


def snapshot_from_app(app: Any) -> RiskSnapshot:
    """Buduje ``RiskSnapshot`` z obiektu aplikacji TradingGUI."""

    balance = 0.0
    try:
        balance = float(getattr(app, "paper_balance", 0.0) or 0.0)
    except Exception:
        balance = 0.0

    raw_settings = getattr(app, "risk_manager_settings", None)
    settings: RiskManagerSettings | None
    if isinstance(raw_settings, RiskManagerSettings):
        settings = raw_settings
    else:
        settings = None

    profile_name_obj = getattr(app, "risk_profile_name", None)
    profile_name = str(profile_name_obj) if profile_name_obj else None

    return RiskSnapshot(balance, settings, profile_name)


def compute_default_notional(snapshot: RiskSnapshot, *, default_notional: float) -> float:
    """Wylicza domyślną kwotę zlecenia na podstawie profilu ryzyka."""

    balance = snapshot.paper_balance
    settings = snapshot.settings

    if balance > 0 and settings is not None:
        limit_value = _resolve_risk_limit(balance, settings)
        if limit_value > 0:
            if default_notional <= 0:
                return limit_value
            return min(default_notional, limit_value)

    return default_notional


def _resolve_risk_limit(balance: float, settings: RiskManagerSettings) -> float:
    """Zwraca limit notional wynikający z profilu ryzyka."""

    limits: list[float] = []

    try:
        per_trade = float(settings.max_risk_per_trade)
    except Exception:
        per_trade = 0.0
    if per_trade > 0:
        limits.append(float(balance) * per_trade)

    try:
        portfolio_risk = float(settings.max_portfolio_risk)
    except Exception:
        portfolio_risk = 0.0
    if portfolio_risk > 0:
        limits.append(float(balance) * portfolio_risk)

    if not limits:
        return 0.0

    return _fmt_float(min(limits), 2)


def format_decimal(value: float, *, decimals: int = 2, fallback: str = "0") -> str:
    """Formatuje liczbę do wpisu w polu tekstowym GUI."""

    try:
        formatted = f"{float(value):.{decimals}f}"
    except Exception:
        return fallback
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or fallback


def format_notional(value: float) -> str:
    """Formatowanie wartości notional (domyślnie 2 miejsca po przecinku)."""

    return format_decimal(value, decimals=2, fallback="0")


def build_risk_profile_hint(snapshot: RiskSnapshot) -> Optional[str]:
    """Zwraca podpowiedź tekstową o aktywnym profilu ryzyka."""

    if not snapshot.profile_name or not snapshot.settings:
        return None

    try:
        per_trade_pct = float(snapshot.settings.max_risk_per_trade) * 100.0
    except Exception:
        per_trade_pct = 0.0

    if per_trade_pct <= 0:
        return f"Profil ryzyka: {snapshot.profile_name}"

    return f"Profil ryzyka: {snapshot.profile_name} (≈{per_trade_pct:.2f}% / trade)"


def build_risk_limits_summary(snapshot: RiskSnapshot) -> Optional[str]:
    """Buduje zwięzły opis limitów ryzyka z aktywnego profilu."""

    settings = snapshot.settings
    if settings is None:
        return None

    parts: list[str] = []

    try:
        portfolio_pct = float(settings.max_portfolio_risk) * 100.0
    except Exception:
        portfolio_pct = 0.0
    if portfolio_pct > 0:
        parts.append(f"Ekspozycja: {portfolio_pct:.1f}%")

    try:
        daily_pct = float(settings.max_daily_loss_pct) * 100.0
    except Exception:
        daily_pct = 0.0
    if daily_pct > 0:
        parts.append(f"Dzienna strata: {daily_pct:.1f}%")

    try:
        drawdown_pct = float(settings.emergency_stop_drawdown) * 100.0
    except Exception:
        drawdown_pct = 0.0
    if drawdown_pct > 0:
        parts.append(f"Stop awaryjny: {drawdown_pct:.1f}%")

    try:
        max_positions = int(settings.max_positions)
    except Exception:
        max_positions = 0
    if max_positions > 0:
        parts.append(f"Pozycje: {max_positions}")

    if not parts:
        return None

    return "Limity ryzyka: " + " | ".join(parts)


def _fmt_float(value: float, max_dec: int) -> float:
    representation = f"{float(value):.{max_dec}f}"
    if "." in representation:
        representation = representation.rstrip("0").rstrip(".")
    return float(representation) if representation else 0.0

