"""Walidacja spójności konfiguracji CoreConfig."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_INTERVAL_SUFFIX_TO_SECONDS: Mapping[str, int] = {
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
    "M": 30 * 24 * 60 * 60,
}


def _interval_seconds(interval: str) -> int:
    """Zwraca długość interwału w sekundach.

    Akceptuje wartości w formacie ``<liczba><jednostka>``, gdzie jednostka należy do
    zestawu {m, h, d, w, M}. Wielkość liter jest znacząca jedynie dla miesięcy
    (``1M``). Przy błędnym formacie zgłaszamy :class:`ValueError`.
    """

    text = interval.strip()
    if not text:
        raise ValueError("interwał nie może być pusty")

    number_part = []
    suffix = None
    for char in text:
        if char.isdigit():
            if suffix is not None:
                raise ValueError(f"niepoprawny format interwału '{interval}'")
            number_part.append(char)
        else:
            if suffix is not None:
                raise ValueError(f"niepoprawny format interwału '{interval}'")
            suffix = char

    if not number_part or suffix is None:
        raise ValueError(f"niepoprawny format interwału '{interval}'")

    seconds_per_unit = _INTERVAL_SUFFIX_TO_SECONDS.get(suffix)
    if seconds_per_unit is None:
        raise ValueError(f"nieobsługiwany sufiks interwału '{suffix}'")

    value = int("".join(number_part))
    if value <= 0:
        raise ValueError("interwał musi być dodatni")

    return value * seconds_per_unit


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
    _validate_strategies(config, errors, warnings)
    _validate_runtime_controllers(config, errors, warnings)
    _validate_instrument_universes(config, errors, warnings)
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


def _validate_strategies(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    for name, strategy in config.strategies.items():
        context = f"strategia '{name}'"
        if strategy.fast_ma <= 0:
            errors.append(f"{context}: fast_ma musi być dodatnie")
        if strategy.slow_ma <= 0:
            errors.append(f"{context}: slow_ma musi być dodatnie")
        if strategy.fast_ma >= strategy.slow_ma:
            errors.append(
                f"{context}: fast_ma musi być mniejsze od slow_ma, otrzymano {strategy.fast_ma} >= {strategy.slow_ma}"
            )
        if strategy.breakout_lookback <= 0:
            errors.append(f"{context}: breakout_lookback musi być dodatnie")
        if strategy.momentum_window <= 0:
            errors.append(f"{context}: momentum_window musi być dodatnie")
        if strategy.atr_window <= 0:
            errors.append(f"{context}: atr_window musi być dodatnie")
        if strategy.atr_multiplier <= 0:
            errors.append(f"{context}: atr_multiplier musi być dodatnie")
        if strategy.min_trend_strength < 0:
            warnings.append(
                f"{context}: min_trend_strength ma wartość ujemną ({strategy.min_trend_strength})"
            )
        if strategy.min_momentum < 0:
            warnings.append(
                f"{context}: min_momentum ma wartość ujemną ({strategy.min_momentum})"
            )


def _validate_runtime_controllers(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    for name, controller in config.runtime_controllers.items():
        context = f"kontroler runtime '{name}'"
        if controller.tick_seconds <= 0:
            errors.append(f"{context}: tick_seconds musi być dodatnie")
        interval = controller.interval.strip()
        if not interval:
            errors.append(f"{context}: interval nie może być pusty")
            continue
        try:
            expected_seconds = _interval_seconds(interval)
        except ValueError as exc:
            errors.append(f"{context}: {exc}")
            continue

        delta = abs(controller.tick_seconds - expected_seconds)
        tolerance = max(1.0, expected_seconds * 0.1)
        if delta > tolerance:
            warnings.append(
                f"{context}: tick_seconds={controller.tick_seconds} znacząco różni się od interwału {interval} (~{expected_seconds}s)"
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
            continue

        if environment.instrument_universe:
            universe = config.instrument_universes[environment.instrument_universe]
            exchange = environment.exchange
            matching_instruments = [
                instrument
                for instrument in universe.instruments
                if exchange in instrument.exchange_symbols
            ]
            if not matching_instruments:
                errors.append(
                    f"{context}: uniwersum instrumentów '{environment.instrument_universe}' nie zawiera powiązań dla giełdy '{exchange}'"
                )
            else:
                intervals_available = {
                    window.interval.strip().lower()
                    for instrument in matching_instruments
                    for window in instrument.backfill_windows
                    if window.interval.strip()
                }
                if intervals_available and config.runtime_controllers:
                    controller_intervals = {
                        controller.interval.strip().lower()
                        for controller in config.runtime_controllers.values()
                        if controller.interval.strip()
                    }
                    if controller_intervals and not (
                        intervals_available & controller_intervals
                    ):
                        warnings.append(
                            f"{context}: brak wspólnego interwału między oknami backfill ({', '.join(sorted(intervals_available)) or 'brak'}) a kontrolerami runtime ({', '.join(sorted(controller_intervals))})"
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


def _validate_instrument_universes(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    known_exchanges = {env.exchange for env in config.environments.values()}

    for name, universe in config.instrument_universes.items():
        context = f"uniwersum instrumentów '{name}'"
        if not universe.instruments:
            errors.append(f"{context}: musi zawierać co najmniej jeden instrument")
            continue

        seen_instruments: set[str] = set()
        for instrument in universe.instruments:
            inst_context = f"{context}: instrument '{instrument.name}'"
            if instrument.name in seen_instruments:
                errors.append(
                    f"{context}: instrument '{instrument.name}' został zdefiniowany wielokrotnie"
                )
            else:
                seen_instruments.add(instrument.name)

            if not instrument.base_asset or not instrument.quote_asset:
                errors.append(
                    f"{inst_context}: base_asset i quote_asset muszą być ustawione"
                )

            if not instrument.categories:
                errors.append(f"{inst_context}: lista kategorii nie może być pusta")
            elif len(set(cat.lower() for cat in instrument.categories)) != len(
                instrument.categories
            ):
                warnings.append(f"{inst_context}: wykryto zduplikowane kategorie")

            if not instrument.exchange_symbols:
                errors.append(
                    f"{inst_context}: musi mieć przypisane co najmniej jedno powiązanie giełdowe"
                )
            else:
                for exchange, symbol in instrument.exchange_symbols.items():
                    if not symbol:
                        errors.append(
                            f"{inst_context}: symbol dla giełdy '{exchange}' nie może być pusty"
                        )
                    if exchange not in known_exchanges:
                        warnings.append(
                            f"{inst_context}: giełda '{exchange}' nie jest zdefiniowana w sekcji environments"
                        )

            if not instrument.backfill_windows:
                warnings.append(
                    f"{inst_context}: brak zdefiniowanych okien backfill – pipeline nie pobierze danych"
                )
                continue

            intervals_seen: set[str] = set()
            for window in instrument.backfill_windows:
                interval = window.interval.strip()
                if not interval:
                    errors.append(
                        f"{inst_context}: interwał w sekcji backfill nie może być pusty"
                    )
                    continue
                try:
                    _interval_seconds(interval)
                except ValueError as exc:
                    errors.append(f"{inst_context}: {exc}")
                    continue
                if window.lookback_days <= 0:
                    errors.append(
                        f"{inst_context}: lookback_days dla interwału '{window.interval}' musi być dodatni"
                    )
                interval_key = interval.lower()
                if interval_key in intervals_seen:
                    warnings.append(
                        f"{inst_context}: interwał '{window.interval}' zdefiniowano wielokrotnie"
                    )
                else:
                    intervals_seen.add(interval_key)
