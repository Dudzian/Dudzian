"""Predefiniowane konfiguracje lokalnych dostawców SMS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True, slots=True)
class SmsProviderConfig:
    """Opisuje parametry integracji z dostawcą SMS."""

    provider_id: str
    display_name: str
    api_base_url: str
    iso_country_code: str
    supports_alphanumeric_sender: bool
    notes: str | None = None
    max_sender_length: int = 11


#: Zestaw standardowych konfiguracji dla operatorów wymaganych w pierwszym etapie.
DEFAULT_SMS_PROVIDERS: Dict[str, SmsProviderConfig] = {
    "orange_pl": SmsProviderConfig(
        provider_id="orange_pl",
        display_name="Orange Polska",
        api_base_url="https://api.orange.pl/sms/v1",
        iso_country_code="PL",
        supports_alphanumeric_sender=True,
        notes="Wymaga whitelisting IP i zgłoszenia alfanumerycznego nadawcy do Orange.",
    ),
    "tmobile_pl": SmsProviderConfig(
        provider_id="tmobile_pl",
        display_name="T-Mobile Polska",
        api_base_url="https://api.t-mobile.pl/sms/v1",
        iso_country_code="PL",
        supports_alphanumeric_sender=True,
        notes="Obsługa w trybie REST, dostęp po podpisaniu umowy A2P.",
    ),
    "plus_pl": SmsProviderConfig(
        provider_id="plus_pl",
        display_name="Plus Polska",
        api_base_url="https://api.plus.pl/messaging/v1",
        iso_country_code="PL",
        supports_alphanumeric_sender=False,
        notes="Domyślnie preferuje numeryczny nadpis nadawcy, alfanumeryczny wymaga dodatkowej zgody.",
    ),
    "play_pl": SmsProviderConfig(
        provider_id="play_pl",
        display_name="Play Polska",
        api_base_url="https://api.play.pl/sms/v1",
        iso_country_code="PL",
        supports_alphanumeric_sender=True,
        notes="Wymaga wcześniejszego zgłoszenia puli IP oraz rejestracji brandu.",
    ),
    "nova_is": SmsProviderConfig(
        provider_id="nova_is",
        display_name="Nova Islandia",
        api_base_url="https://api.nova.is/sms/v1",
        iso_country_code="IS",
        supports_alphanumeric_sender=True,
        notes="Islandzki operator z obsługą webhooków statusowych i fallbackiem numerowym.",
    ),
}


def get_sms_provider(provider_key: str) -> SmsProviderConfig:
    """Zwraca konfigurację dostawcy na podstawie klucza."""

    try:
        return DEFAULT_SMS_PROVIDERS[provider_key]
    except KeyError as exc:  # pragma: no cover - błąd sygnalizujemy w miejscu użycia
        raise KeyError(f"Nieznany dostawca SMS: {provider_key}") from exc


__all__ = ["SmsProviderConfig", "DEFAULT_SMS_PROVIDERS", "get_sms_provider"]
