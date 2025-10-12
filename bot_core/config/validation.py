"""Walidacja spójności konfiguracji CoreConfig."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from bot_core.config.models import CoreConfig

_UI_ALERT_AUDIT_BACKEND_ALLOWED = {"auto", "file", "memory"}
_SUPPORTED_TOKEN_HASH_ALGORITHMS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
}

# Mapowanie sufiksów interwałów na sekundy.
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
    zestawu {m, h, d, w, M}. Wielkość liter jest znacząca jedynie dla miesięcy (``1M``).
    Przy błędnym formacie zgłaszamy :class:`ValueError`.
    """
    text = interval.strip()
    if not text:
        raise ValueError("interwał nie może być pusty")

    number_part: list[str] = []
    suffix: str | None = None
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
    _validate_metrics_service(config, errors, warnings)
    _validate_risk_service(config, errors, warnings)
    _validate_risk_decision_log(config, errors, warnings)

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

        # tolerancja 10% lub min. 1 sekunda
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
    strategies = set(config.strategies)
    controllers = set(config.runtime_controllers)

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

        intervals_available: set[str] = set()
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
                    if controller_intervals and not (intervals_available & controller_intervals):
                        warnings.append(
                            f"{context}: brak wspólnego interwału między oknami backfill ({', '.join(sorted(intervals_available)) or 'brak'}) a kontrolerami runtime ({', '.join(sorted(controller_intervals))})"
                        )

        # Spójność default_* względem zdefiniowanych sekcji
        default_strategy = getattr(environment, "default_strategy", None)
        if strategies:
            if not default_strategy:
                errors.append(
                    f"{context}: default_strategy nie jest ustawione mimo zdefiniowanych strategii"
                )
            elif default_strategy not in strategies:
                errors.append(
                    f"{context}: domyślna strategia '{default_strategy}' nie istnieje w sekcji strategies"
                )
        elif default_strategy:
            errors.append(
                f"{context}: domyślna strategia '{default_strategy}' wskazana bez dostępnych strategii"
            )

        default_controller = getattr(environment, "default_controller", None)
        default_controller_interval: str | None = None
        if controllers:
            if not default_controller:
                errors.append(
                    f"{context}: default_controller nie jest ustawione mimo zdefiniowanych kontrolerów runtime"
                )
            elif default_controller not in controllers:
                errors.append(
                    f"{context}: domyślny kontroler '{default_controller}' nie istnieje w sekcji runtime.controllers"
                )
            else:
                controller_cfg = config.runtime_controllers[default_controller]
                interval_text = controller_cfg.interval.strip()
                if interval_text:
                    default_controller_interval = interval_text.lower()
        elif default_controller:
            errors.append(
                f"{context}: domyślny kontroler '{default_controller}' wskazany bez dostępnych kontrolerów runtime"
            )

        _validate_alert_channels(config, environment.alert_channels, context, errors)
        _validate_permissions(
            environment.required_permissions,
            environment.forbidden_permissions,
            context,
            errors,
        )

        # Twardy błąd, jeśli domyślny kontroler ma interwał niedostępny w backfillu uniwersum
        if (
            default_controller
            and default_controller_interval
            and environment.instrument_universe
            and intervals_available
            and default_controller_interval not in intervals_available
        ):
            controller_cfg = config.runtime_controllers[default_controller]
            errors.append(
                f"{context}: domyślny kontroler '{default_controller}' używa interwału '{controller_cfg.interval}'"
                f" niedostępnego w oknach backfill uniwersum '{environment.instrument_universe}'"
                f" dla giełdy '{environment.exchange}'"
            )


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
            errors.append(f"{context}: kanał alertowy '{channel}' nie istnieje w sekcji alerts")


def _validate_metrics_service(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    metrics = getattr(config, "metrics_service", None)
    if metrics is None:
        return

    context = "runtime.metrics_service"

    tokens = tuple(getattr(metrics, "rbac_tokens", ()) or ())
    if tokens:
        _validate_service_tokens_block(
            tokens=tokens,
            context=f"{context}.rbac_tokens",
            errors=errors,
            warnings=warnings,
        )

    tls = getattr(metrics, "tls", None)
    if tls is not None and getattr(tls, "enabled", False):
        certificate = getattr(tls, "certificate_path", None)
        private_key = getattr(tls, "private_key_path", None)
        if not certificate or not str(certificate).strip():
            errors.append(
                f"{context}: TLS wymaga certificate_path przy włączonym szyfrowaniu"
            )
        if not private_key or not str(private_key).strip():
            errors.append(
                f"{context}: TLS wymaga private_key_path przy włączonym szyfrowaniu"
            )
        if getattr(tls, "require_client_auth", False):
            client_ca = getattr(tls, "client_ca_path", None)
            if not client_ca or not str(client_ca).strip():
                errors.append(
                    f"{context}: TLS z require_client_auth wymaga client_ca_path"
                )
    if tls is not None:
        pinned = tuple(getattr(tls, "pinned_fingerprints", ()) or ())
        if pinned and not getattr(tls, "enabled", False):
            warnings.append(
                f"{context}: tls.pinned_fingerprints ustawione, ale TLS jest wyłączone"
            )
        seen_pins: set[str] = set()
        for entry in pinned:
            normalized = str(entry).strip().lower()
            if not normalized or ":" not in normalized:
                errors.append(
                    f"{context}: wpis tls.pinned_fingerprints='{entry}' ma nieprawidłowy format"
                )
                continue
            if normalized in seen_pins:
                warnings.append(
                    f"{context}: tls.pinned_fingerprints zawiera duplikat '{entry}'"
                )
            seen_pins.add(normalized)
        password_env = getattr(tls, "private_key_password_env", None)
        if password_env is not None:
            normalized_env = str(password_env).strip()
            if not normalized_env:
                errors.append(
                    f"{context}: tls.private_key_password_env nie może być puste"
                )
            elif normalized_env and not normalized_env.isupper():
                warnings.append(
                    f"{context}: nazwa zmiennej tls.private_key_password_env powinna być wielkimi literami"
                )

    _validate_ui_alert_block(
        context=context,
        block_name="reduce_motion",
        enabled_flag=bool(getattr(metrics, "reduce_motion_alerts", False)),
        mode_value=getattr(metrics, "reduce_motion_mode", None),
        category_value=getattr(metrics, "reduce_motion_category", ""),
        required_severities={
            "reduce_motion_severity_active": getattr(
                metrics, "reduce_motion_severity_active", ""
            ),
            "reduce_motion_severity_recovered": getattr(
                metrics, "reduce_motion_severity_recovered", ""
            ),
        },
        optional_severities={},
        threshold_value=None,
        threshold_label=None,
        errors=errors,
        warnings=warnings,
    )

    _validate_ui_alert_block(
        context=context,
        block_name="overlay",
        enabled_flag=bool(getattr(metrics, "overlay_alerts", False)),
        mode_value=getattr(metrics, "overlay_alert_mode", None),
        category_value=getattr(metrics, "overlay_alert_category", ""),
        required_severities={
            "overlay_alert_severity_exceeded": getattr(
                metrics, "overlay_alert_severity_exceeded", ""
            ),
            "overlay_alert_severity_recovered": getattr(
                metrics, "overlay_alert_severity_recovered", ""
            ),
        },
        optional_severities={
            "overlay_alert_severity_critical": getattr(
                metrics, "overlay_alert_severity_critical", None
            )
        },
        threshold_value=getattr(metrics, "overlay_alert_critical_threshold", None),
        threshold_label="overlay_alert_critical_threshold",
        errors=errors,
        warnings=warnings,
    )

    _validate_ui_alert_block(
        context=context,
        block_name="jank",
        enabled_flag=bool(getattr(metrics, "jank_alerts", False)),
        mode_value=getattr(metrics, "jank_alert_mode", None),
        category_value=getattr(metrics, "jank_alert_category", ""),
        required_severities={
            "jank_alert_severity_spike": getattr(
                metrics, "jank_alert_severity_spike", ""
            ),
        },
        optional_severities={
            "jank_alert_severity_critical": getattr(
                metrics, "jank_alert_severity_critical", None
            )
        },
        threshold_value=getattr(metrics, "jank_alert_critical_over_ms", None),
        threshold_label="jank_alert_critical_over_ms",
        errors=errors,
        warnings=warnings,
    )

    backend_value = getattr(metrics, "ui_alerts_audit_backend", None)
    if backend_value is not None:
        normalized = str(backend_value).strip().lower()
        if normalized and normalized not in _UI_ALERT_AUDIT_BACKEND_ALLOWED:
            errors.append(
                f"{context}: ui_alerts_audit_backend musi należeć do {{auto,file,memory}} (otrzymano '{backend_value}')"
            )

    profile_value = getattr(metrics, "ui_alerts_risk_profile", None)
    if profile_value:
        normalized_profile = str(profile_value).strip().lower()
        available_profiles = getattr(config, "risk_profiles", {}) or {}
        if normalized_profile not in available_profiles:
            errors.append(
                f"{context}: ui_alerts_risk_profile '{profile_value}' nie istnieje w sekcji risk_profiles"
            )

    profiles_file_value = getattr(metrics, "ui_alerts_risk_profiles_file", None)
    if profiles_file_value:
        profiles_path = Path(str(profiles_file_value)).expanduser()
        if not profiles_path.exists():
            errors.append(
                f"{context}: ui_alerts_risk_profiles_file '{profiles_path}' nie istnieje"
            )


def _validate_risk_service(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    risk_service = getattr(config, "risk_service", None)
    if risk_service is None:
        return

    context = "runtime.risk_service"

    tokens = tuple(getattr(risk_service, "rbac_tokens", ()) or ())
    if tokens:
        _validate_service_tokens_block(
            tokens=tokens,
            context=f"{context}.rbac_tokens",
            errors=errors,
            warnings=warnings,
        )

    history_size = getattr(risk_service, "history_size", 0)
    if history_size is None or int(history_size) <= 0:
        errors.append(f"{context}: history_size musi być dodatnie")

    port = getattr(risk_service, "port", 0)
    if port is None or int(port) < 0:
        errors.append(f"{context}: port nie może być ujemny")

    interval_raw = getattr(risk_service, "publish_interval_seconds", None)
    interval_value: float | None = None
    if interval_raw is not None:
        try:
            interval_value = float(interval_raw)
        except (TypeError, ValueError):
            errors.append(f"{context}: publish_interval_seconds musi być liczbą dodatnią")
    if interval_value is None:
        errors.append(f"{context}: publish_interval_seconds musi być dodatnie")
    elif interval_value <= 0:
        errors.append(f"{context}: publish_interval_seconds musi być dodatnie")

    profiles = getattr(risk_service, "profiles", None)
    if profiles:
        seen: set[str] = set()
        for idx, profile in enumerate(profiles, start=1):
            normalized = str(profile).strip()
            if not normalized:
                errors.append(f"{context}: profiles[{idx}] nie może być puste")
                break
            if normalized in seen:
                warnings.append(
                    f"{context}: profiles zawiera zduplikowany wpis '{normalized}' – zostanie zignorowany"
                )
                continue
            seen.add(normalized)

    tls = getattr(risk_service, "tls", None)
    if tls is not None and getattr(tls, "enabled", False):
        certificate = getattr(tls, "certificate_path", None)
        private_key = getattr(tls, "private_key_path", None)
        if not certificate or not str(certificate).strip():
            errors.append(
                f"{context}: TLS wymaga certificate_path przy włączonym szyfrowaniu"
            )
        if not private_key or not str(private_key).strip():
            errors.append(
                f"{context}: TLS wymaga private_key_path przy włączonym szyfrowaniu"
            )
        if getattr(tls, "require_client_auth", False):
            client_ca = getattr(tls, "client_ca_path", None)
            if not client_ca or not str(client_ca).strip():
                errors.append(
                    f"{context}: TLS z require_client_auth wymaga client_ca_path"
                )
    if tls is not None:
        pinned = tuple(getattr(tls, "pinned_fingerprints", ()) or ())
        if pinned and not getattr(tls, "enabled", False):
            warnings.append(
                f"{context}: tls.pinned_fingerprints ustawione, ale TLS jest wyłączone"
            )
        seen_pins: set[str] = set()
        for entry in pinned:
            normalized = str(entry).strip().lower()
            if not normalized or ":" not in normalized:
                errors.append(
                    f"{context}: wpis tls.pinned_fingerprints='{entry}' ma nieprawidłowy format"
                )
                continue
            if normalized in seen_pins:
                warnings.append(
                    f"{context}: tls.pinned_fingerprints zawiera duplikat '{entry}'"
                )
            seen_pins.add(normalized)
        password_env = getattr(tls, "private_key_password_env", None)
        if password_env is not None:
            normalized_env = str(password_env).strip()
            if not normalized_env:
                errors.append(
                    f"{context}: tls.private_key_password_env nie może być puste"
                )
            elif normalized_env and not normalized_env.isupper():
                warnings.append(
                    f"{context}: nazwa zmiennej tls.private_key_password_env powinna być wielkimi literami"
                )


def _validate_risk_decision_log(
    config: CoreConfig, errors: list[str], warnings: list[str]
) -> None:
    log_config = getattr(config, "risk_decision_log", None)
    if log_config is None or not getattr(log_config, "enabled", True):
        return

    context = "runtime.risk_decision_log"

    max_entries = getattr(log_config, "max_entries", 0)
    if max_entries is None or int(max_entries) <= 0:
        errors.append(f"{context}: max_entries musi być dodatnie")

    path_value = getattr(log_config, "path", None)
    if path_value is not None and not str(path_value).strip():
        errors.append(f"{context}: path nie może być puste")

    key_sources: list[str] = []
    for source in ("signing_key_env", "signing_key_path", "signing_key_value"):
        value = getattr(log_config, source, None)
        if value not in (None, ""):
            key_sources.append(source)

    if len(key_sources) > 1:
        errors.append(
            f"{context}: skonfiguruj tylko jedno źródło klucza podpisu (env/path/value)"
        )

    if not key_sources:
        warnings.append(
            f"{context}: brak klucza podpisu – podpisy HMAC nie będą generowane"
        )

    if getattr(log_config, "signing_key_value", None):
        warnings.append(
            f"{context}: signing_key_value w pliku YAML może naruszać politykę bezpieczeństwa"
        )


def _validate_ui_alert_block(
    *,
    context: str,
    block_name: str,
    enabled_flag: bool,
    mode_value: str | None,
    category_value: str | None,
    required_severities: Mapping[str, str | None],
    optional_severities: Mapping[str, str | None],
    threshold_value: float | int | None,
    threshold_label: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    allowed_modes = {"enable", "jsonl", "disable"}
    normalized_mode: str | None = None
    if mode_value is not None:
        normalized_mode = str(mode_value).strip().lower()
        if not normalized_mode:
            normalized_mode = None
        elif normalized_mode not in allowed_modes:
            errors.append(
                f"{context}: {block_name}_mode musi należeć do {{enable,jsonl,disable}} (otrzymano '{mode_value}')"
            )

    if normalized_mode is None:
        normalized_mode = "enable" if enabled_flag else "disable"
    else:
        if normalized_mode == "disable" and enabled_flag:
            warnings.append(
                f"{context}: {block_name}_alerts ustawione na True, ale tryb '{normalized_mode}' je wyciszy"
            )
        if normalized_mode != "disable" and not enabled_flag:
            warnings.append(
                f"{context}: {block_name}_alerts ustawione na False, tryb '{normalized_mode}' je jednak aktywuje"
            )

    if normalized_mode == "disable":
        return

    if not category_value or not str(category_value).strip():
        errors.append(
            f"{context}: {block_name}_alert_category nie może być puste przy aktywnych alertach"
        )

    for field_name, severity in required_severities.items():
        if not severity or not str(severity).strip():
            errors.append(
                f"{context}: {field_name} nie może być puste przy aktywnych alertach"
            )

    for field_name, severity in optional_severities.items():
        if severity is not None and not str(severity).strip():
            errors.append(f"{context}: {field_name} nie może być puste jeśli jest ustawione")

    if threshold_label is not None and threshold_value is not None:
        try:
            numeric = float(threshold_value)
        except (TypeError, ValueError):  # pragma: no cover - defensywne logowanie
            errors.append(
                f"{context}: {threshold_label} musi być wartością liczbową"
            )
            return
        if numeric <= 0:
            errors.append(
                f"{context}: {threshold_label} musi być dodatnie (otrzymano {threshold_value})"
            )


def _validate_service_tokens_block(
    *,
    tokens,
    context: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    seen_ids: set[str] = set()
    for idx, token in enumerate(tokens, start=1):
        token_context = f"{context}[{idx}]"
        token_id = str(getattr(token, "token_id", "")).strip()
        if not token_id:
            errors.append(f"{token_context}: token_id nie może być puste")
            continue
        if token_id in seen_ids:
            warnings.append(
                f"{context}: token_id '{token_id}' występuje wielokrotnie – kolejne wpisy będą nadpisywać wcześniejsze"
            )
        seen_ids.add(token_id)

        token_env = getattr(token, "token_env", None)
        token_value = getattr(token, "token_value", None)
        token_hash = getattr(token, "token_hash", None)

        if not token_env and not token_value and not token_hash:
            errors.append(
                f"{token_context}: wymagane token_value, token_env lub token_hash dla definicji tokenu"
            )

        if token_env is not None:
            env_name = str(token_env).strip()
            if not env_name:
                errors.append(f"{token_context}.token_env nie może być puste")
            elif not env_name.isupper():
                warnings.append(
                    f"{token_context}.token_env powinno być zapisane wielkimi literami (otrzymano '{token_env}')"
                )

        if token_hash:
            normalized = str(token_hash).strip().lower()
            if ":" in normalized:
                algorithm, digest = normalized.split(":", 1)
            else:
                algorithm, digest = "sha256", normalized
            algorithm = algorithm.strip()
            digest = digest.strip()
            if algorithm and algorithm not in _SUPPORTED_TOKEN_HASH_ALGORITHMS:
                errors.append(
                    f"{token_context}.token_hash wykorzystuje nieobsługiwany algorytm '{algorithm}'"
                )
            if not digest:
                errors.append(f"{token_context}.token_hash wymaga wartości skrótu hex")
            else:
                try:
                    bytes.fromhex(digest)
                except ValueError:
                    errors.append(
                        f"{token_context}.token_hash musi zawierać poprawny zapis hex (otrzymano '{token_hash}')"
                    )

        scopes = tuple(getattr(token, "scopes", ()) or ())
        seen_scopes: set[str] = set()
        for scope_idx, scope in enumerate(scopes, start=1):
            normalized_scope = str(scope).strip().lower()
            if not normalized_scope:
                warnings.append(
                    f"{token_context}.scopes[{scope_idx}] jest puste i zostanie pominięte"
                )
                continue
            if normalized_scope in seen_scopes:
                warnings.append(
                    f"{token_context}.scopes[{scope_idx}] duplikuje wpis '{normalized_scope}'"
                )
                continue
            seen_scopes.add(normalized_scope)

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
                errors.append(f"{inst_context}: base_asset i quote_asset muszą być ustawione")

            if not instrument.categories:
                errors.append(f"{inst_context}: lista kategorii nie może być pusta")
            elif len(set(cat.lower() for cat in instrument.categories)) != len(instrument.categories):
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
                    errors.append(f"{inst_context}: interwał w sekcji backfill nie może być pusty")
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
                    warnings.append(f"{inst_context}: interwał '{window.interval}' zdefiniowano wielokrotnie")
                else:
                    intervals_seen.add(interval_key)
