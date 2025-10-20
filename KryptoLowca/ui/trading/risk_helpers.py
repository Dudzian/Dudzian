"""Pomocnicze funkcje do obsługi profili ryzyka w modułach Trading GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from bot_core.runtime.metadata import RiskManagerSettings

try:  # pragma: no cover - środowiska bez bootstrapu runtime
    from bot_core.runtime.metadata import load_risk_manager_settings
except Exception:  # pragma: no cover - defensywne fallbacki
    load_risk_manager_settings = None  # type: ignore[assignment]

__all__ = [
    "RiskSnapshot",
    "snapshot_from_app",
    "compute_default_notional",
    "format_decimal",
    "format_notional",
    "build_risk_profile_hint",
    "build_risk_limits_summary",
    "apply_runtime_risk_context",
    "refresh_runtime_risk_context",
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


def apply_runtime_risk_context(
    gui: Any,
    *,
    entrypoint: str,
    default_notional: float,
    config_path: str | None = None,
    logger: logging.Logger | None = None,
) -> RiskSnapshot:
    """Uzupełnia obiekt GUI o ustawienia profilu ryzyka z runtime bootstrapu.

    Funkcja jest defensywna – brak loadera lub wyjątek podczas ładowania
    ustawień nie przerywa konfiguracji, a jedynie loguje szczegóły (jeśli
    przekazano logger). Zwraca zaktualizowany ``RiskSnapshot`` tak, by testy i
    wywołujący mogli łatwo pobrać wynikowe limity.
    """

    loader = load_risk_manager_settings if callable(load_risk_manager_settings) else None

    current_profile_name = getattr(gui, "risk_profile_name", None)
    current_settings = getattr(gui, "risk_manager_settings", None)
    resolved_name: str | None = current_profile_name
    profile_payload: Any | None = getattr(gui, "risk_profile_config", None)
    settings = current_settings if isinstance(current_settings, RiskManagerSettings) else None

    if loader is not None:
        try:
            loaded_name, loaded_payload, loaded_settings = loader(  # type: ignore[misc]
                entrypoint,
                profile_name=current_profile_name,
                config_path=config_path,
                logger=logger,
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            if logger:
                logger.debug(
                    "Nie udało się wczytać ustawień risk managera dla %s",
                    entrypoint,
                    exc_info=True,
                )
        else:
            if loaded_name:
                resolved_name = loaded_name
            if loaded_payload is not None:
                profile_payload = loaded_payload
            if isinstance(loaded_settings, RiskManagerSettings):
                settings = loaded_settings

    if resolved_name and hasattr(gui, "risk_profile_name"):
        try:
            setattr(gui, "risk_profile_name", resolved_name)
        except Exception:  # pragma: no cover - defensywne logowanie
            if logger:
                logger.debug("Nie udało się ustawić nazwy profilu ryzyka.", exc_info=True)

    if profile_payload is not None and hasattr(gui, "risk_profile_config"):
        try:
            setattr(gui, "risk_profile_config", profile_payload)
        except Exception:  # pragma: no cover - defensywne logowanie
            if logger:
                logger.debug(
                    "Nie udało się ustawić konfiguracji profilu ryzyka.",
                    exc_info=True,
                )

    if settings is not None and hasattr(gui, "risk_manager_settings"):
        try:
            setattr(gui, "risk_manager_settings", settings)
        except Exception:  # pragma: no cover - defensywne logowanie
            if logger:
                logger.debug(
                    "Nie udało się zastosować obiektu RiskManagerSettings.",
                    exc_info=True,
                )

    snapshot = snapshot_from_app(gui)
    return _synchronize_gui_view(
        gui,
        snapshot,
        default_notional=default_notional,
        logger=logger,
    )


def refresh_runtime_risk_context(
    gui: Any,
    *,
    default_notional: float,
    logger: logging.Logger | None = None,
) -> RiskSnapshot:
    """Aktualizuje ``default_paper_notional`` i banery po zmianie limitów ryzyka.

    Funkcja nie korzysta z loadera runtime – zakłada, że ``gui`` zawiera już
    aktualne ustawienia ``RiskManagerSettings`` i służy jedynie do
    przeliczenia pochodnych wartości (podpowiedzi, tytułu okna, domyślnego
    notionala). Przydaje się po zmianach profilu w trakcie działania aplikacji.
    """

    snapshot = snapshot_from_app(gui)
    return _synchronize_gui_view(
        gui,
        snapshot,
        default_notional=default_notional,
        logger=logger,
    )


def _synchronize_gui_view(
    gui: Any,
    snapshot: RiskSnapshot,
    *,
    default_notional: float,
    logger: logging.Logger | None,
) -> RiskSnapshot:
    notional = compute_default_notional(snapshot, default_notional=default_notional)
    if snapshot.settings is not None:
        try:
            limit_value = _resolve_risk_limit(snapshot.paper_balance, snapshot.settings)
        except Exception:  # pragma: no cover - defensywne logowanie
            limit_value = 0.0
        if limit_value > notional:
            notional = limit_value
    hint = build_risk_profile_hint(snapshot)

    if hint and logger:
        logger.info("%s", hint)
    if logger:
        logger.info(
            "Domyślny notional (paper): %s USDT",
            format_notional(notional),
        )

    try:
        setattr(gui, "default_paper_notional", notional)
    except Exception:  # pragma: no cover - defensywne logowanie
        if logger:
            logger.debug("Nie udało się zapisać domyślnego notional.", exc_info=True)

    root = getattr(gui, "root", None)
    if hint and root is not None:
        try:
            current_title = root.title() if hasattr(root, "title") else None
            if current_title:
                root.title(f"{current_title} — {hint}")
        except Exception:  # pragma: no cover - defensywne logowanie
            if logger:
                logger.debug("Nie udało się zaktualizować tytułu okna.", exc_info=True)

    update_banner = getattr(gui, "_update_risk_banner", None)
    if callable(update_banner):
        try:
            update_banner()
        except Exception:  # pragma: no cover - defensywne logowanie
            if logger:
                logger.debug("Nie udało się odświeżyć banera ryzyka.", exc_info=True)

    return snapshot

