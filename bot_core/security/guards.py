"""Mechanizmy sprawdzające uprawnienia wynikające z licencji offline."""
from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from bot_core.security.capabilities import LicenseCapabilities
from bot_core.security.fingerprint import (
    FingerprintError,
    SecuritySignals,
    collect_security_signals,
)


LOGGER = logging.getLogger(__name__)


class LicenseCapabilityError(PermissionError):
    """Wyjątek rzucany, gdy moduł lub funkcja nie jest dozwolona."""

    def __init__(self, message: str, *, capability: str | None = None) -> None:
        super().__init__(message)
        self.capability = capability


_EDITION_ORDER = ("community", "standard", "pro", "commercial")
_LICENSE_TIER_TO_EDITION = {
    "community": "community",
    "standard": "standard",
    "professional": "pro",
    "pro": "pro",
    "enterprise": "commercial",
    "commercial": "commercial",
}
_GLOBAL_GUARD: "CapabilityGuard | None" = None


def _edition_rank(value: str) -> int:
    normalized = value.strip().lower()
    try:
        return _EDITION_ORDER.index(normalized)
    except ValueError:
        LOGGER.warning("Nieznana edycja licencji: %s", value)
        return -1
_LIMIT_FIELD_MAP = {
    "paper_controller": "max_paper_controllers",
    "live_controller": "max_live_controllers",
    "bot": "max_concurrent_bots",
    "alert_channel": "max_alert_channels",
}


@dataclass(slots=True)
class CapabilityGuard:
    """Stan licznika i zestaw metod do kontroli funkcji runtime."""

    capabilities: LicenseCapabilities
    _slots: Counter[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_slots", Counter())

    # --- metody wymagające flag -------------------------------------------------
    def require_module(self, module: str, message: str | None = None) -> None:
        if not self.capabilities.is_module_enabled(module):
            raise LicenseCapabilityError(
                message or f"Moduł '{module}' nie jest dostępny w tej licencji.",
                capability=module,
            )

    def require_runtime(self, runtime: str, message: str | None = None) -> None:
        if not self.capabilities.is_runtime_enabled(runtime):
            raise LicenseCapabilityError(
                message or f"Runtime '{runtime}' nie jest dostępny w tej licencji.",
                capability=runtime,
            )

    def require_exchange(self, exchange_id: str, message: str | None = None) -> None:
        if not self.capabilities.is_exchange_enabled(exchange_id):
            raise LicenseCapabilityError(
                message or f"Giełda '{exchange_id}' nie jest aktywna w licencji.",
                capability=exchange_id,
            )

    def require_strategy(self, strategy_id: str, message: str | None = None) -> None:
        if not self.capabilities.is_strategy_enabled(strategy_id):
            raise LicenseCapabilityError(
                message or f"Strategia '{strategy_id}' nie jest dostępna.",
                capability=strategy_id,
            )

    def require_environment(self, environment: str, message: str | None = None) -> None:
        if not self.capabilities.is_environment_allowed(environment):
            raise LicenseCapabilityError(
                message or f"Środowisko '{environment}' nie jest dozwolone.",
                capability=environment,
            )

    def require_edition(self, min_edition: str, message: str | None = None) -> None:
        current = self.capabilities.edition
        required_rank = _edition_rank(min_edition)
        if required_rank < 0:
            raise LicenseCapabilityError(
                f"Nieznana edycja referencyjna '{min_edition}'.",
                capability="edition",
            )
        if _edition_rank(current) < required_rank:
            raise LicenseCapabilityError(
                message or f"Licencja '{current}' nie spełnia wymagań edycji '{min_edition}'.",
                capability="edition",
            )

    def require_license_tier(self, license_tier: str, message: str | None = None) -> None:
        normalized = (license_tier or "").strip().lower()
        if not normalized or normalized in {"unspecified", "none"}:
            return
        mapped = _LICENSE_TIER_TO_EDITION.get(normalized)
        if mapped is None:
            raise LicenseCapabilityError(
                f"Nieznany poziom licencyjny '{license_tier}'.",
                capability="license_tier",
            )
        default_message = (
            f"Licencja '{self.capabilities.edition}' nie spełnia wymaganej klasy '{normalized}'."
        )
        self.require_edition(mapped, message=message or default_message)

    def require_maintenance(self) -> None:
        if not self.capabilities.is_maintenance_active():
            raise LicenseCapabilityError(
                "Okres utrzymania licencji wygasł.",
                capability="maintenance",
            )

    def require_trial_active(self) -> None:
        if not self.capabilities.is_trial_active():
            raise LicenseCapabilityError(
                "Okres trial licencji wygasł.",
                capability="trial",
            )

    # --- limity -----------------------------------------------------------------
    def reserve_slot(self, kind: str, *, count: int = 1) -> None:
        if count <= 0:
            return
        field = _LIMIT_FIELD_MAP.get(kind)
        limit = getattr(self.capabilities.limits, field) if field else None
        if limit is not None and self._slots[kind] + count > limit:
            raise LicenseCapabilityError(
                f"Przekroczono limit '{kind}' ({limit}).",
                capability=kind,
            )
        self._slots[kind] += count

    def release_slot(self, kind: str, *, count: int = 1) -> None:
        if count <= 0:
            return
        self._slots[kind] = max(0, self._slots[kind] - count)

    # --- metody pomocnicze ------------------------------------------------------
    def preload_slots(self, mapping: dict[str, int]) -> None:
        for key, value in mapping.items():
            if value > 0:
                self._slots[key] = value

    def describe_limits(self) -> dict[str, int | None]:
        return {key: getattr(self.capabilities.limits, attr) for key, attr in _LIMIT_FIELD_MAP.items()}

    def describe_missing_modules(self, required: Iterable[str]) -> list[str]:
        missing: list[str] = []
        for module in required:
            if not self.capabilities.is_module_enabled(module):
                missing.append(module)
        return missing


def set_capability_guard(guard: "CapabilityGuard | None") -> None:
    """Rejestruje globalnego strażnika licencyjnego."""

    global _GLOBAL_GUARD
    _GLOBAL_GUARD = guard


def reset_capability_guard() -> None:
    """Czyści zarejestrowanego strażnika (używane w testach)."""

    set_capability_guard(None)


def get_capability_guard() -> "CapabilityGuard | None":
    """Zwraca obecnie zainstalowanego strażnika możliwości."""

    return _GLOBAL_GUARD


def build_capability_guard(
    capabilities: LicenseCapabilities,
    *,
    signals: SecuritySignals | None = None,
    **signal_kwargs: Any,
) -> CapabilityGuard:
    """Buduje strażnika i uruchamia heurystyki bezpieczeństwa."""

    if signals is None:
        skip_env = os.environ.get("DUDZIAN_SECURITY_SKIP", "").strip().lower()
        if skip_env in {"1", "true", "yes", "on"}:
            LOGGER.warning("Pominięto heurystyki bezpieczeństwa (DUDZIAN_SECURITY_SKIP).")
            return CapabilityGuard(capabilities)
        signals = collect_security_signals(**signal_kwargs)
    if signals.should_block:
        summary_items = signals.summary()
        summary_text = "; ".join(summary_items) if summary_items else "nieokreślone sygnały"
        LOGGER.error("Zablokowano utworzenie strażnika licencji: %s", summary_text)
        raise FingerprintError(
            "Wykryto podejrzane sygnały środowiska: " + summary_text
        )
    return CapabilityGuard(capabilities)


def install_capability_guard(
    capabilities: LicenseCapabilities, **signal_kwargs: Any
) -> CapabilityGuard:
    """Buduje strażnika na podstawie capabilities i rejestruje go globalnie."""

    guard = build_capability_guard(capabilities, **signal_kwargs)
    set_capability_guard(guard)
    return guard


__all__ = [
    "CapabilityGuard",
    "LicenseCapabilityError",
    "build_capability_guard",
    "get_capability_guard",
    "install_capability_guard",
    "reset_capability_guard",
    "set_capability_guard",
]
