"""Walidacja spójności konfiguracji CoreConfig."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from bot_core.config.models import CoreConfig


@dataclass(slots=True)
class ConfigValidationResult:
    """Wynik walidacji konfiguracji."""

    errors: list[str]
    warnings: list[str]

    def is_valid(self) -> bool:
        """Zwraca True, jeśli nie znaleziono błędów."""

        return not self.errors


class ConfigValidationError(RuntimeError):
    """Wyjątek rzucany przy krytycznych błędach konfiguracji."""

    def __init__(self, result: ConfigValidationResult):
        self.result = result
        message = "\n".join(result.errors)
        super().__init__(message or "Nieznany błąd walidacji konfiguracji")


def validate_core_config(config: CoreConfig) -> ConfigValidationResult:
    """Waliduje spójność konfiguracji i zwraca listę błędów/ostrzeżeń."""

    errors: list[str] = []
    warnings: list[str] = []

    _validate_risk_profiles(config, errors, warnings)
    _validate_environments(config, errors, warnings)

    return ConfigValidationResult(errors=errors, warnings=warnings)


def assert_core_config_valid(config: CoreConfig) -> ConfigValidationResult:
    """Waliduje konfigurację i rzuca wyjątek przy błędach."""

    result = validate_core_config(config)
    if result.errors:
        raise ConfigValidationError(result)
    return result


def _validate_risk_profiles(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    for name, profile in config.risk_profiles.items():
        context = f"profil ryzyka '{name}'"
        if profile.max_daily_loss_pct < 0:
            errors.append(f"{context}: max_daily_loss_pct nie może być ujemne")
        if profile.max_position_pct < 0:
            errors.append(f"{context}: max_position_pct nie może być ujemne")
        if profile.target_volatility < 0:
            errors.append(f"{context}: target_volatility nie może być ujemne")
        if profile.max_leverage < 0:
            errors.append(f"{context}: max_leverage nie może być ujemne")
        if profile.stop_loss_atr_multiple < 0:
            errors.append(f"{context}: stop_loss_atr_multiple nie może być ujemne")
        if profile.max_open_positions < 0:
            errors.append(f"{context}: max_open_positions nie może być ujemne")
        if profile.hard_drawdown_pct < 0:
            errors.append(f"{context}: hard_drawdown_pct nie może być ujemne")

        if name.lower() != profile.name.lower():
            warnings.append(
                f"profil ryzyka '{name}' ma nazwę '{profile.name}' – zalecana spójność"
            )


def _validate_environments(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    risk_profiles = set(config.risk_profiles)
    universes = set(config.instrument_universes)

    for name, environment in config.environments.items():
        context = f"środowisko '{name}'"
        if environment.risk_profile not in risk_profiles:
            errors.append(
                f"{context}: profil ryzyka '{environment.risk_profile}' nie istnieje w konfiguracji"
            )

        if environment.instrument_universe and environment.instrument_universe not in universes:
            errors.append(
                f"{context}: uniwersum instrumentów '{environment.instrument_universe}' nie istnieje"
            )

        _validate_alert_channels(config, environment.alert_channels, context, errors)
        _validate_permissions(environment.required_permissions, environment.forbidden_permissions, context, errors)


def _validate_permissions(
    required: Mapping | tuple | list | set,
    forbidden: Mapping | tuple | list | set,
    context: str,
    errors: list[str],
) -> None:
    required_set = {str(value).lower() for value in required}
    forbidden_set = {str(value).lower() for value in forbidden}
    overlap = required_set & forbidden_set
    if overlap:
        overlap_list = ", ".join(sorted(overlap))
        errors.append(
            f"{context}: uprawnienia {overlap_list} nie mogą jednocześnie znajdować się w required i forbidden"
        )


def _validate_alert_channels(
    config: CoreConfig, alert_channels: tuple[str, ...], context: str, errors: list[str]
) -> None:
    registry: Mapping[str, Mapping[str, object]] = {
        "telegram": config.telegram_channels,
        "email": config.email_channels,
        "sms": config.sms_providers,
        "signal": config.signal_channels,
        "whatsapp": config.whatsapp_channels,
        "messenger": config.messenger_channels,
    }

    for channel in alert_channels:
        if ":" not in channel:
            errors.append(f"{context}: kanał alertowy '{channel}' ma nieprawidłowy format")
            continue
        backend, key = channel.split(":", 1)
        backend = backend.strip().lower()
        key = key.strip()
        if not backend or not key:
            errors.append(f"{context}: kanał alertowy '{channel}' ma nieprawidłowy format")
            continue
        if backend not in registry:
            errors.append(
                f"{context}: kanał alertowy '{channel}' wskazuje nieobsługiwany typ '{backend}'"
            )
            continue
        mapping = registry[backend]
        if key not in mapping:
            errors.append(
                f"{context}: kanał alertowy '{channel}' nie istnieje w sekcji alerts"
            )

