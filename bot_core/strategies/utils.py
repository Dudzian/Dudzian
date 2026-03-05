"""Wspólne helpery strategii redukujące duplikację logiki sygnałów i backtestów."""

from __future__ import annotations

from typing import Any, Dict


def compute_change_ratio(previous_price: float | None, current_price: float) -> float | None:
    """Zwraca relatywną zmianę ceny, ignorując brak lub zerowe wartości.

    Parametry
    ---------
    previous_price: float | None
        Ostatnia znana cena zamknięcia lub ``None`` gdy brak historii.
    current_price: float
        Bieżąca cena zamknięcia.

    Zwraca
    -------
    float | None
        Stosunek zmiany (current/previous - 1) lub ``None`` gdy brak danych wejściowych.
    """

    if previous_price is None or previous_price <= 0:
        return None
    if current_price <= 0:
        return None
    return (current_price - previous_price) / previous_price


def build_signal_metadata(
    *,
    strategy_type: str,
    profile: str,
    risk_label: str,
    position: str,
    exit_reason: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Buduje spójny blok metadanych sygnału z informacjami o strategii.

    Parametry
    ---------
    strategy_type: str
        Nazwa typu strategii (np. ``"scalping"``).
    profile: str
        Profil opisujący wariant strategii.
    risk_label: str
        Oznaczenie poziomu ryzyka.
    position: str
        Kierunek pozycji (``"long"``, ``"short"`` lub ``"flat"``).
    exit_reason: str | None
        Opcjonalny powód zamknięcia pozycji.
    extra: Dict[str, Any] | None
        Dodatkowe pola metadanych dodawane do wyniku.
    """

    metadata: Dict[str, Any] = {
        "strategy": {
            "type": strategy_type,
            "profile": profile,
            "risk_label": risk_label,
        },
        "position": position,
    }
    if exit_reason:
        metadata["exit_reason"] = exit_reason
    if extra:
        metadata.update(extra)
    return metadata


__all__ = ["compute_change_ratio", "build_signal_metadata"]
