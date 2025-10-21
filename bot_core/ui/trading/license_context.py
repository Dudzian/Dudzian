from dataclasses import dataclass
from datetime import date
from typing import Iterable

from bot_core.security.capabilities import LicenseCapabilities

__all__ = ["COMMUNITY_NOTICE", "LicenseUiContext", "build_license_ui_context"]

COMMUNITY_NOTICE = "Wersja Community: dostępny tylko backtest i Paper Labs."


@dataclass(frozen=True)
class LicenseUiContext:
    """Zestaw informacji używanych przez GUI do prezentacji licencji."""

    summary: str
    notice: str
    live_enabled: bool
    futures_enabled: bool
    auto_trader_enabled: bool
    maintenance_active: bool


def _format_date(value: date | None) -> str:
    if value is None:
        return "?"
    return value.isoformat()


def _format_summary(capabilities: LicenseCapabilities) -> str:
    holder_name = ""
    try:
        holder_name = str(capabilities.holder.get("name", "")).strip()
    except Exception:
        holder_name = ""

    parts: list[str] = [f"Licencja: {capabilities.edition.title()}"]
    if capabilities.license_id:
        parts.append(f"ID: {capabilities.license_id}")
    if capabilities.maintenance_until:
        parts.append(f"Maintenance do {capabilities.maintenance_until.isoformat()}")
    if capabilities.seats:
        parts.append(f"Stanowiska: {capabilities.seats}")
    if holder_name:
        parts.append(f"Klient: {holder_name}")
    return " • ".join(parts)


def _merge_messages(messages: Iterable[str]) -> str:
    merged: list[str] = []
    for message in messages:
        text = (message or "").strip()
        if not text:
            continue
        if text not in merged:
            merged.append(text)
    return " ".join(merged)


def build_license_ui_context(
    capabilities: LicenseCapabilities | None,
) -> LicenseUiContext:
    """Buduje opis licencji na potrzeby GUI."""

    if capabilities is None:
        return LicenseUiContext(
            summary="Licencja: Community (tryb ograniczony)",
            notice=COMMUNITY_NOTICE,
            live_enabled=False,
            futures_enabled=False,
            auto_trader_enabled=False,
            maintenance_active=False,
        )

    maintenance_active = capabilities.is_maintenance_active()
    live_enabled = capabilities.is_environment_allowed("live")
    futures_enabled = capabilities.is_module_enabled("futures")
    auto_trader_enabled = capabilities.is_runtime_enabled("auto_trader")

    messages: list[str] = []

    if not maintenance_active:
        messages.append(
            "Licencja wygasła "
            f"{_format_date(capabilities.maintenance_until)} – aktualizacje i integracje live są zablokowane. "
            "Skontaktuj się z zespołem OEM."
        )

    if capabilities.trial.enabled and not capabilities.is_trial_active():
        messages.append("Okres trial licencji wygasł. Skontaktuj się z opiekunem licencji.")

    if not live_enabled:
        messages.append("Tryb live wymaga edycji Pro. Skontaktuj się z opiekunem licencji.")

    if not futures_enabled:
        messages.append("Dodaj moduł Futures, aby aktywować handel kontraktami.")

    if not auto_trader_enabled:
        messages.append("Licencja nie obejmuje modułu AutoTrader. Skontaktuj się z opiekunem licencji.")

    if not capabilities.is_module_enabled("observability_ui"):
        messages.append("Moduł telemetry UI jest niedostępny w tej licencji.")

    if not messages:
        messages.append("Licencja aktywna.")

    summary = _format_summary(capabilities)
    notice = _merge_messages(messages)

    return LicenseUiContext(
        summary=summary,
        notice=notice,
        live_enabled=live_enabled,
        futures_enabled=futures_enabled,
        auto_trader_enabled=auto_trader_enabled,
        maintenance_active=maintenance_active,
    )
